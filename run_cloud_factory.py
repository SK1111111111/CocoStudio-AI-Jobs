#!/usr/bin/env python3
"""Process the next CocoStudio cloud job on a CUDA machine."""
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def write_report(output_dir, **updates):
    path = output_dir / "report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report.update(updates)
    report["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(report, indent=2))
    return report


def next_job():
    for job_file in sorted((ROOT / "input").glob("*/job.json")):
        output = ROOT / "output" / job_file.parent.name
        report = output / "report.json"
        if not report.exists() or json.loads(report.read_text()).get("status") not in ("animation_ready", "rig_pending", "failed"):
            return job_file
    return None


def generate_hunyuan(job_file, output_dir):
    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    job = json.loads(job_file.read_text())
    input_dir = job_file.parent
    images = {view: str(input_dir / filename) for view, filename in job["views"].items() if view in ("front", "left", "right", "back")}
    write_report(output_dir, status="generating_mesh", viewsReceived=sorted(images), engine="Hunyuan3D-2mv")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mv",
        subfolder="hunyuan3d-dit-v2-mv",
        device="cuda",
        dtype=torch.float16,
    )
    mesh = pipeline(image=images, num_inference_steps=30, octree_resolution=384)[0]
    raw_path = output_dir / "character.glb"
    mesh.export(raw_path)
    write_report(output_dir, status="draft_mesh_generated", mesh=True, textureApplied=False, rigReady=False, animationReady=False)
    return raw_path


def optional_rig(mesh_path, output_dir):
    command = os.environ.get("COCOSTUDIO_CLOUD_RIG_COMMAND", "").strip()
    if not command:
        write_report(output_dir, status="rig_pending", rigReady=False, animationReady=False, note="No cloud rig command configured.")
        return
    final_path = output_dir / "character_animation_ready.glb"
    env = {**os.environ, "INPUT_MODEL": str(mesh_path), "OUTPUT_MODEL": str(final_path)}
    result = subprocess.run(command, shell=True, env=env, capture_output=True, text=True, timeout=3600)
    (output_dir / "rig.log").write_text(result.stdout + "\n" + result.stderr)
    ready = result.returncode == 0 and final_path.exists() and final_path.stat().st_size > 1024
    write_report(output_dir, status="animation_ready" if ready else "rig_pending", rigReady=ready, animationReady=ready)


def main():
    job_file = next_job()
    if not job_file:
        print("No pending CocoStudio cloud jobs.")
        return
    output_dir = ROOT / "output" / job_file.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        mesh = generate_hunyuan(job_file, output_dir)
        optional_rig(mesh, output_dir)
    except Exception as exc:
        write_report(output_dir, status="failed", error=f"{type(exc).__name__}: {exc}", animationReady=False)
        raise
    subprocess.run(["git", "add", f"output/{job_file.parent.name}"], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", f"Cloud result: {job_file.parent.name}"], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", os.environ.get("COCOSTUDIO_CLOUD_BRANCH", "main")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
