# mesh2smplx

Fit SMPL, SMPL-H, or SMPL-X bodies to textured mesh sequences and precomputed
3D keypoints. The repository is source-only: it does not include SMPL-family
model files, OpenPose weights, datasets, generated outputs, or local machine
configs.

## Folder Layout

```text
mesh2smplx/
  main.py                  Repository-root entry point
  configs/                 One public starter config
  scripts/                 Convenience launch and visualization scripts
  mesh2smplx/
    main.py                Package entry point
    core/                  Config, data loading, rendering plan, triangulation
    fitting/               SMPL/SMPL-H/SMPL-X fitting losses and schedules
    openpose/              Self-contained OpenPose-135 migration and JSON format
    visualization/         Optional AITviewer integration
```

## Install

Create an environment from the repository root:

```bash
cd mesh2smplx
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For CUDA runs, install the PyTorch wheel that matches your driver and CUDA
version before installing the package. Optional extras:

```bash
python -m pip install -e ".[viewer]"      # AITviewer support
python -m pip install -e ".[openpose]"    # optional OpenPose-135 wrapper
```

## Prepare Data

Prepare one mesh per frame. OBJ is the main tested format.

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

Textures are optional when the mesh material already points to an image or when
you are fitting from precomputed 3D keypoints only.

`keypoints_3d.npy` should have shape:

```text
(num_frames, num_joints, 5)
```

The first three values are XYZ in the same unit system as the mesh before
`scale_to_meters` is applied. The last value is confidence. The fourth value is
kept for compatibility with triangulation outputs.

The default joint order is OpenPose-135:

```text
BODY_25 + left hand + right hand + face inner + face contour
```

If you use a different detector or joint set, update the joint mapping in
`mesh2smplx/fitting/joints.py`.

Download SMPL, SMPL-H, or SMPL-X model files from the official providers and
store them outside this repository, for example `/path/to/body_models`.

## Configure

Copy the starter config into an ignored local config folder:

```bash
mkdir -p local_configs
cp configs/textured_mesh.yaml local_configs/my_sequence.yaml
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
python -m mesh2smplx inspect --config local_configs/my_sequence.yaml
```

Then it launches:

```bash
python -m mesh2smplx fit-full \
  --config local_configs/my_sequence.yaml \
  --keypoints3d /path/to/my_sequence/keypoints_3d.npy \
  --frame-indices 0
```

By default the starter script caps optimizer work with
`MAX_STEPS_PER_STAGE=5`. For a full schedule:

```bash
MAX_STEPS_PER_STAGE= scripts/run_starter.sh local_configs/my_sequence.yaml /path/to/keypoints_3d.npy 0-30
```

You can also call the root entry point directly:

```bash
python main.py inspect --config local_configs/my_sequence.yaml
```

## Useful Commands

Inspect mesh discovery without fitting:

```bash
python -m mesh2smplx inspect --config local_configs/my_sequence.yaml
```

Run the mesh and keypoint fitting command directly:

```bash
python -m mesh2smplx fit-full \
  --config local_configs/my_sequence.yaml \
  --keypoints3d /path/to/keypoints_3d.npy \
  --frame-indices 0-10 \
  --tracking
```

Open saved outputs in AITviewer:

```bash
python -m mesh2smplx view \
  --config local_configs/my_sequence.yaml \
  --outputs outputs/my_sequence/fit_full
```

Run the optional OpenPose-135 wrapper directly:

```bash
python -m mesh2smplx.openpose --help
mesh2smplx-openpose --help
```

## Licensing And External Assets

The original code and documentation in this repository are marked as
CC-BY-NC-4.0. This does not grant rights to third-party assets or upstream code
with separate license terms.

Users must download SMPL-family models from the official providers and accept
their licenses. OpenPose model weights are not shipped with this repository.
If code is derived from MPI/SMPLify-X fitting components, keep the upstream
non-commercial research license terms and attribution intact.

Keep these out of the public repository:

- local virtual environments
- `outputs/`, logs, screenshots, and generated meshes
- local configs containing private paths
- SMPL-family model files and converted model pickles
- OpenPose or other detector weights
- private datasets and capture-system conversion scripts
