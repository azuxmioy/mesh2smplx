# Data Folder

This folder is intentionally empty in git except for this README. Put one
sequence here when running the starter configs, or replace `input.root` in the
config with another sequence folder.

## Example Data

Download the example data archive from Hugging Face and extract it from the
repository root:

```bash
scripts/download_example_data.sh
```

The script downloads `mesh2smplx_example_data.zip` from
<https://huggingface.co/datasets/hohs/mesh2smplx> and extracts it to `data/`.
The archive contains a small textured mesh sequence, camera calibration, and
optional fixed body-shape betas.

If you prefer to download manually:

```bash
hf download hohs/mesh2smplx mesh2smplx_example_data.zip \
  --repo-type dataset \
  --local-dir .
unzip -o mesh2smplx_example_data.zip
```

## Required Input

```text
data/
  meshes/
    mesh-f00001.obj
    matlib-f00001.mtl
    atlas-f00001.png
    mesh-f00140.obj
    matlib-f00140.mtl
    atlas-f00140.png
```

`meshes/` is required and should contain one mesh per fitted frame. OBJ is the
main tested mesh format; `.ply`, `.stl`, `.glb`, `.gltf`, and `.off` are also
discovered. Textures are optional, but textured OBJ sequences should keep their
`.mtl` files and texture images next to the meshes.

Frame ids are parsed from the mesh filename first, then from the parent folder
name. For example, `mesh-f00080.obj`, `000080.obj`, and `000080/mesh.obj` all
resolve to frame id `80`.

## Optional Real Images

```text
data/
  images/
    cam_000/
      000001.png
      000140.png
    cam_001/
      000001.png
      000140.png
  calibration/
    cameras.json
```

If `images/` is provided, OpenPose runs on those images and `calibration/` is
required. The expected layout is one subfolder per camera id.

`calibration/` may contain `cameras.json`, `rgb_cameras.json`,
`calibration.json`, `camera.json`, or a single JSON file. Camera ids in the JSON
must match the image subfolder names.

Each camera entry should provide intrinsics, world-to-camera extrinsics, and
optionally image size and distortion coefficients:

```json
{
  "cam_000": {
    "intrinsics": [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
    "extrinsics": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.5]],
    "image_size": [720, 1280]
  }
}
```

`extrinsics` can be `3x4` or `4x4`. `image_size` is `[height, width]`; if it is
missing, the loader infers it from the first image.

## Rendering Without Images

If `images/` is missing, the pipeline renders images from the meshes before
OpenPose. With calibration present, `rendering.mode: auto` renders from the
provided cameras. Without calibration, the renderer samples heuristic cameras
from the upper semi-sphere around the mesh center.

Rendered images and masks are written to:

```text
data/rendered/
  images/
  masks/
```

Virtual rendering uses Kaolin and requires CUDA.

## Optional Fixed Body Shape

```text
data/body_shape.npy
```

If this file exists, the fitter uses it as fixed SMPL-family betas and does not
optimize body shape. The array should have shape `(num_betas,)` or
`(1, num_betas)`.

You can also pass a shape file explicitly:

```bash
scripts/run_gpu.sh configs/gpu.yaml --betas data/body_shape.npy
```

## Generated Files

The pipeline writes intermediate and final results back under `data/`:

```text
data/
  keypoints_2d/              # OpenPose JSON per camera/frame
  keypoints_3d.npy           # triangulated 3D keypoints
  fitting/
    smplx/
      mesh-f00001_smplx_params.json
      mesh-f00001_smplx.obj
      mesh-f00001_scan.obj
```

`keypoints_3d.npy` has shape:

```text
(num_frames, num_joints, 5)
```

The first three values are XYZ. The last value is confidence; the fourth value
is kept for compatibility with triangulation outputs. The default joint order is
OpenPose-135:

```text
BODY_25 + left hand + right hand + face inner + face contour
```

Fitting outputs are grouped by model type. The `*_scan.obj` file is the source
mesh scaled by `input.scale_to_meters`, which is useful for overlay debugging.
