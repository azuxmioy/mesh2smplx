# SMPL Registration

Source-only SMPL/SMPL-H/SMPL-X fitting from public data interfaces. The main
starter path fits a body model to a textured mesh sequence plus precomputed 3D
keypoints, with optional mesh alignment and AITviewer visualization.

This repository does not include SMPL-family model files, OpenPose weights,
datasets, generated outputs, or environment-specific configs.

## Folder Layout

```text
registration_oss_draft/
  README.md
  pyproject.toml
  docs/
    data_format.md
  examples/
    configs/textured_mesh.yaml
  scripts/
    run_starter.sh
  src/smpl_registration/
  tests/
```

## Install

Create an environment from the package folder:

```bash
cd registration_oss_draft
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For CUDA runs, install the PyTorch wheel that matches your driver and CUDA
version before installing the package. Optional extras:

```bash
python -m pip install -e ".[viewer]"      # AITviewer support
python -m pip install -e ".[openpose135]" # optional OpenPose-135 wrapper
```

## Prepare Data

Prepare a mesh sequence and a 3D-keypoint file:

```text
my_sequence/
  meshes/
    frame_000000.obj
    frame_000001.obj
  textures/
    frame_000000.png
    frame_000001.png
  keypoints_3d.npy
```

`keypoints_3d.npy` should have shape `(num_frames, num_joints, 5)`, with XYZ in
the same units as the meshes and confidence in the last column. Set
`input.scale_to_meters` in the config to convert those units for fitting.

Download SMPL, SMPL-H, or SMPL-X model files from the official providers and
store them outside this repository, for example `/path/to/body_models`.

More detail is in [docs/data_format.md](docs/data_format.md).

## Configure

Copy the starter config into an ignored local config folder:

```bash
mkdir -p local_configs
cp examples/configs/textured_mesh.yaml local_configs/my_sequence.yaml
```

Update at least these fields:

```yaml
input:
  root: /path/to/my_sequence
  meshes: /path/to/my_sequence/meshes
  textures: /path/to/my_sequence/textures
  scale_to_meters: 0.001

body_model:
  model_path: /path/to/body_models

fitting:
  device: cuda
  output_dir: outputs/my_sequence
```

Use `device: cpu` for a small smoke test when CUDA is unavailable.

## Launch

Run the starter script after editing the config:

```bash
scripts/run_starter.sh local_configs/my_sequence.yaml /path/to/my_sequence/keypoints_3d.npy 0
```

Arguments:

```text
scripts/run_starter.sh <config.yaml> <keypoints_3d.npy> [frame_indices]
```

`frame_indices` indexes into the sorted mesh/keypoint sequence. Use values such
as `0`, `0,5,10`, or `0-30`.

The starter script first runs:

```bash
python -m smpl_registration inspect --config local_configs/my_sequence.yaml
```

Then it launches:

```bash
python -m smpl_registration fit-full \
  --config local_configs/my_sequence.yaml \
  --keypoints3d /path/to/my_sequence/keypoints_3d.npy \
  --frame-indices 0
```

By default the starter script caps optimizer work with
`MAX_STEPS_PER_STAGE=5`. For a full schedule:

```bash
MAX_STEPS_PER_STAGE= scripts/run_starter.sh local_configs/my_sequence.yaml /path/to/keypoints_3d.npy 0-30
```

## Useful Commands

Inspect mesh discovery without fitting:

```bash
python -m smpl_registration inspect --config local_configs/my_sequence.yaml
```

Run the mesh and keypoint fitting command directly:

```bash
python -m smpl_registration fit-full \
  --config local_configs/my_sequence.yaml \
  --keypoints3d /path/to/keypoints_3d.npy \
  --frame-indices 0-10 \
  --tracking
```

Open saved outputs in AITviewer:

```bash
python -m smpl_registration view \
  --config local_configs/my_sequence.yaml \
  --outputs outputs/my_sequence/fit_full
```

## Release Notes

Keep these out of the public repository:

- local virtual environments
- `outputs/`, logs, screenshots, and generated meshes
- local configs containing private paths
- SMPL-family model files and converted model pickles
- OpenPose or other detector weights
- private datasets and capture-system conversion scripts
