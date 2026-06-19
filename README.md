# mesh2smplx

Fit SMPL, SMPL-H, or SMPL-X bodies to mesh sequences using real camera images,
rendered virtual views, or precomputed 3D keypoints, then optionally convert
fitted outputs between SMPL-family model types. The repository is source-only:
it does not include SMPL-family model files, OpenPose weights, datasets,
generated outputs, or local machine configs.

## Folder Layout

```text
mesh2smplx/
  main.py                  Repository-root entry point
  configs/                 Public server and local starter configs
  scripts/                 Two public launchers: GPU full pipeline and local AITviewer debug
  mesh2smplx/
    main.py                CLI implementation used by the root entry point
    core/                  Config, data loading, rendering plan, triangulation
    fitting/               SMPL/SMPL-H/SMPL-X fitting, losses, conversion
    openpose/              Self-contained OpenPose-135 migration and JSON format
    visualization/         Optional AITviewer integration
```

## Install

Create one environment from the repository root. Python 3.10 or newer is
required; Python 3.11 is recommended for the local AITviewer UI.

```bash
cd mesh2smplx
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file installs the full feature set by default: fitting,
OpenPose-135, conversion, virtual-camera rendering, and AITviewer `v1.14.2`.
For CUDA runs, keep the PyTorch and Kaolin lines aligned with the server's CUDA
stack. The comments in `requirements.txt` mark the optional pieces that can be
disabled when a user only needs part of the pipeline.

## Prepare Data

Create one data folder per sequence. Meshes are always required; images and
calibration are optional depending on how you want to produce 2D keypoints.

```text
data/
  meshes/
    000000.obj
    000001.obj
    # or nested frame folders such as 000000/mesh.obj
  images/                    # optional: real camera images for OpenPose
    cam_000/
      frame_000000.png
      frame_000001.png
    cam_001/
      frame_000000.png
      frame_000001.png
  calibration/               # required only when images/ is used
    cameras.json
  keypoints_2d/              # generated automatically when missing
  keypoints_3d.npy           # generated automatically, then used by fitting
  body_shape.npy             # optional fixed SMPL-family betas
```

`meshes/` is required and should contain one mesh per fitted frame. OBJ is the
main tested mesh format, and common mesh extensions such as `.ply`, `.stl`,
`.glb`, `.gltf`, and `.off` are also discovered. Frame ids are parsed from the
mesh filename first and then from the parent folder name, so both `000080.obj`
and `000080/mesh.obj` are valid layouts.

`images/` is optional. If `input.images` is configured, those images are used as
OpenPose inputs and `calibration/` must also be configured. The expected layout is
one subfolder per camera id.

`calibration/` is required only when real images are provided. It may be a
directory containing `cameras.json` or a direct path to a camera JSON file. The
camera ids must match the image subfolder names:

```json
{
  "cam_000": {
    "intrinsics": [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
    "extrinsics": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.5]],
    "image_size": [720, 1280]
  }
}
```

`extrinsics` is world-to-camera and can be either `3x4` or `4x4`. `image_size`
is `[height, width]`; if omitted, the loader infers it from the first image.
`dist_coeffs` is optional.

If `images/` is not configured, the pipeline renders images from the meshes. When
calibration is provided, those cameras are used for rendering. When only
`meshes/` is provided, virtual cameras are sampled uniformly over the upper
semi-sphere around the mesh center.

You can also skip 2D keypoint extraction and triangulation by providing
precomputed 3D keypoints with `--keypoints3d`.

By default the pipeline reads/writes 2D keypoints at `data/keypoints_2d/` and
writes triangulated 3D keypoints to `data/keypoints_3d.npy`. Override
`input.keypoints_2d` or `input.keypoints_3d` only when you want a different
location. Rendered virtual views default to `data/rendered/`, and fitting writes
registration outputs to `data/fitting/`.

Fitting outputs are grouped only by output model type and then kept flat. For a
source mesh named `mesh-f00080.obj`, SMPL-X fitting writes:

```text
data/fitting/smplx/
  mesh-f00080_smplx_params.json  # fitted parameters for this frame
  mesh-f00080_smplx.obj          # fitted body mesh
  mesh-f00080_scan.obj           # original source mesh scaled by input.scale_to_meters
```

`body_shape.npy` is optional. If present, it should contain shape betas with shape
`(num_betas,)` or `(1, num_betas)`. The fitter copies these betas into the body
model and keeps them fixed, so shape is not optimized during fitting. You can also
set `input.body_shape` explicitly; `.npz` with key `betas` and JSON files are
accepted by the same loader.

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

## Prepare Body Models

The repository includes an ignored `body_models/` folder for local third-party
assets. Model files are not redistributable with this codebase, so each user must
register, accept the upstream licenses, and download the files directly from the
official providers:

- SMPL-X: <https://smpl-x.is.tue.mpg.de>
- SMPL: <https://smpl.is.tue.mpg.de>
- neutral SMPL model used by some SMPLify workflows: <https://smplify.is.tue.mpg.de>
- SMPL-H / MANO assets: <https://mano.is.tue.mpg.de>

The `smplx` Python package accepts `body_model.model_path` as either a direct
model file or a directory with this layout. We use the directory layout:

```text
body_models/
  smpl/
    SMPL_NEUTRAL.pkl
    SMPL_MALE.pkl
    SMPL_FEMALE.pkl
  smplh/
    SMPLH_NEUTRAL.pkl
    SMPLH_MALE.pkl
    SMPLH_FEMALE.pkl
  smplx/
    SMPLX_NEUTRAL.npz
    SMPLX_MALE.npz
    SMPLX_FEMALE.npz
```

Only the model type and gender selected in the config are required. The default
config uses neutral SMPL-X:

```yaml
body_model:
  type: smplx
  gender: neutral
  model_path: body_models
```

For the default setup, place `body_models/smplx/SMPLX_NEUTRAL.npz`. If you set
`gender: male` or `gender: female`, provide the matching `SMPLX_MALE.npz` or
`SMPLX_FEMALE.npz` file instead. If you fit with
`body_model.type: smpl` or `smplh`, prepare the corresponding `body_models/smpl/`
or `body_models/smplh/` files.

SMPL and SMPL-H `.pkl` files may need the cleanup/merge steps documented by the
upstream `smplx` package before PyTorch loading:
<https://github.com/vchoutas/smplx/blob/main/tools/README.md>. In particular,
SMPL-H often requires merging SMPL-H and MANO files into `SMPLH_<GENDER>.pkl`.

For SMPL-X<->SMPL output conversion, also download the SMPL-X model
correspondences from the SMPL-X downloads page and place the transfer matrices
under `body_models/transfer/`:

```text
body_models/
  transfer/
    smplx2smpl_deftrafo_setup.pkl  # SMPL-X output -> SMPL output
    smpl2smplx_deftrafo_setup.pkl  # SMPL output -> SMPL-X output
```

Conversion also needs both source and target model files. For example,
SMPL-X-to-SMPL conversion needs a SMPL-X model in `body_models/smplx/`, a SMPL
model in `body_models/smpl/`, and `body_models/transfer/smplx2smpl_deftrafo_setup.pkl`.
The upstream conversion notes are here:
<https://github.com/vchoutas/smplx/blob/main/transfer_model/README.md>.
Conversion uses `scipy`, which is included in `requirements.txt`.

## Prepare OpenPose Checkpoints

OpenPose-135 weights are not shipped with this repository. The pipeline uses the
local checkpoint folder below by default and auto-fetches missing files the first
time OpenPose runs:

```text
checkpoints/openpose135/
  body_pose_model_25.pth
  hand_pose_model.pth
  facenet.pth
```

The starter config makes this explicit:

```yaml
keypoints:
  provider: auto
  weights_dir: checkpoints/openpose135
```

All three files are fetched from the project Hugging Face mirror by default:
<https://huggingface.co/hohs/openpose135-weights>. If automatic download fails,
manually place the three `.pth` files in `checkpoints/openpose135/`.

## Configure

Two starter configs are included:

```text
configs/server_full_pipeline.yaml  # CUDA, full optimizer schedule, full pipeline
configs/local_aitviewer.yaml       # CPU/debug defaults, AITviewer-friendly smoke run
```

Update at least these fields in the config you want to run:

```yaml
input:
  root: data
  meshes: data/meshes
  images: null
  calibration: null
  keypoints_2d: null    # null uses data/keypoints_2d
  keypoints_3d: null    # null uses data/keypoints_3d.npy
  body_shape: null      # null auto-detects data/body_shape.npy when present
  scale_to_meters: 0.001

body_model:
  model_path: body_models

virtual_cameras:
  background_color: green
  output_dir: data/rendered

keypoints:
  provider: auto
  weights_dir: checkpoints/openpose135
  hf_repo: hohs/openpose135-weights
  crop_to_mask: true
  crop_padding_pixels: 30
  crop_aspect_height: 1200
  crop_aspect_width: 900
  max_input_size: 1280

fitting:
  device: cuda
  output_dir: data      # fitting writes data/fitting
  scan_surface_samples: 2000
  body_vertex_samples: 2000
  max_steps_per_stage: null
  max_total_steps: null
  tracking: true        # fit all meshes sequentially with warm starts

conversion:
  output_types:
    - smplx
  transfer_dir: null   # set to body_models/transfer for SMPL-X<->SMPL conversion
```

`meshes`, `virtual_cameras.output_dir`, and `fitting.output_dir` may also be
omitted or set to `null`; they default to `data/meshes`, `data/rendered`, and
`data` from `input.root`.

`virtual_cameras.background_color` controls the rendered image background when
the pipeline must render mesh views before OpenPose. It defaults to green
because a black background is less reliable for detection and cropping. Valid
values are color names such as `green`, `white`, `gray`, `black`, RGB lists such
as `[0, 255, 0]`, or comma strings such as `"0,1,0"`.

Use `device: cpu` only when you already have images/keypoints and are not using
virtual mesh rendering. Rendering virtual OpenPose images from meshes requires
CUDA because it uses Kaolin rasterization.

The OpenPose-135 detector uses CUDA automatically when PyTorch sees a GPU. For
large calibrated renders, the pipeline follows the old `copy_frame_v2.py` path:
it crops each OpenPose input around the rendered mesh mask with a 1200x900 body
crop, resizes the detector input to at most `keypoints.max_input_size`, and maps
the detected keypoints back to the original camera image coordinates before
writing JSON.

## Launch

For GPU/full-sequence fitting, run:

```bash
scripts/run_gpu_full_pipeline.sh configs/server_full_pipeline.yaml
```

This launches the single fitting entry point without frame overrides:

```bash
python main.py --config configs/server_full_pipeline.yaml
```

By default this loads every mesh in `data/meshes/` in sorted filename order. If
`fitting.tracking: true`, frames are fitted sequentially and each frame is
warm-started from the previous result. Set `fitting.tracking: false` only when
you intentionally want to fit the selected frames as one batch.

For local debugging with AITviewer, run:

```bash
scripts/run_local_debug_aitviewer.sh configs/local_aitviewer.yaml
```

The debug script defaults to frame index `0` and launches AITviewer. You can pass
a small subset such as `0-5`:

```bash
scripts/run_local_debug_aitviewer.sh configs/local_aitviewer.yaml 0-5
```

Both starter scripts accept an optional precomputed 3D keypoint file:

```bash
scripts/run_gpu_full_pipeline.sh configs/server_full_pipeline.yaml data/keypoints_3d.npy
scripts/run_local_debug_aitviewer.sh configs/local_aitviewer.yaml data/keypoints_3d.npy 0
```

If `data/keypoints_3d.npy` is missing, the command first generates 2D/3D
keypoints from the configured images or rendered mesh views, then continues into
fitting.

The local AITviewer config caps optimizer work with
`fitting.max_steps_per_stage: 5`. For SMPL-X this is 9 fitting stages, so the
live viewer reports 45 optimizer steps. Set it to `null` for the full fitting
schedule.

When AITviewer is launched from `main.py`, it initializes the viewport from the
first configured calibration camera by default. Pick a camera id or disable this
with:

```bash
python main.py --config configs/local_aitviewer.yaml --frame-indices 0 \
  --aitviewer-launch --aitviewer-initial-camera 0

python main.py --config configs/local_aitviewer.yaml --frame-indices 0 \
  --aitviewer-launch --aitviewer-initial-camera none
```

The launched viewer uses the same Python environment as `main.py`. The default
viewer settings disable shadows and use `znear=0.05`, `zfar=50.0` to avoid
floor/origin depth noise in AITviewer.

To request SMPL output, edit the config instead of passing shell variables:

```yaml
conversion:
  output_types:
    - smplx
    - smpl
  transfer_dir: body_models/transfer
```

## Useful Commands

Run the mesh and keypoint fitting command directly:

```bash
python main.py --config configs/server_full_pipeline.yaml
```

For a small debug subset, pass indices into the sorted mesh/keypoint sequence:

```bash
python main.py --config configs/local_aitviewer.yaml --frame-indices 0-10
```

Fit with SMPL-X and request SMPL-X plus converted SMPL outputs by setting:

```yaml
conversion:
  output_types:
    - smplx
    - smpl
  transfer_dir: body_models/transfer
```

`body_model.type` controls the model used during fitting. `conversion.output_types`
controls what gets written. If a requested output type differs from the fitted
type, the converter runs automatically. Results are always written under
`output_dir/<model_type>` so SMPL-X and converted SMPL outputs stay separate.

The converter first maps source vertices to the target topology with the sparse
transfer matrix, then optimizes the target model parameters to those vertices.
Set `conversion.transfer_matrix` instead of `conversion.transfer_dir` when the
matrix file has a custom name or location.

Stream progress to AITviewer during fitting:

```bash
python main.py \
  --config configs/local_aitviewer.yaml \
  --aitviewer-launch
```

Run the optional OpenPose-135 wrapper directly:

```bash
python -m mesh2smplx.openpose --help
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
- config edits containing private paths
- SMPL-family model files, transfer matrices, and converted model pickles
- OpenPose or other detector weights
- private datasets and capture-system conversion scripts
