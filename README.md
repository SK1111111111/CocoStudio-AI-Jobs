# CocoStudio Cloud GPU Character Factory

This repository is synchronized by CocoStudio. Each job appears under
`input/<job-id>/`; cloud results must be written under `output/<job-id>/`.

Open `CocoStudio_Cloud_GPU.ipynb` in Google Colab, set the GitHub repository URL,
and run its cells with a GPU runtime. The notebook uses the official
`Tencent-Hunyuan/Hunyuan3D-2` project and its multiview model.

The mesh generator does not automatically provide an animation rig. Unless
`COCOSTUDIO_CLOUD_RIG_COMMAND` is configured, the report correctly returns
`rig_pending` and CocoStudio will not label the model animation-ready.
