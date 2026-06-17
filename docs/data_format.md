# Data Format

This package starts from files that can be produced by any reconstruction or
keypoint pipeline. Keep conversion from private capture systems outside this
repository.

## Mesh Sequence

Prepare one mesh per frame. OBJ is the main tested format.

```text
my_sequence/
  meshes/
    frame_000000.obj
    frame_000001.obj
  textures/
    frame_000000.png
    frame_000001.png
```

Textures are optional when the mesh material already points to an image or when
you are fitting from precomputed 3D keypoints only.

The example config discovers frames with:

```yaml
input:
  meshes: /path/to/my_sequence/meshes
  textures: /path/to/my_sequence/textures
  mesh_glob: "frame_*.obj"
  texture_glob: "frame_*.png"
  frame_id_regex: ".*?(\\d+)$"
  scale_to_meters: 0.001
```

Set `scale_to_meters` to `0.001` for millimeter-space meshes and keypoints, or
`1.0` when the input is already in meters.

## Body Models

Download SMPL, SMPL-H, or SMPL-X from the official model providers and place
them outside the repository, for example:

```text
body_models/
  smplx/
  smplh/
  smpl/
```

Point `body_model.model_path` in the config at that folder. Do not commit model
files, converted model pickles, or license-restricted weights.

## 3D Keypoints

The starter path expects a NumPy file:

```text
keypoints_3d.npy
```

Shape:

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
`src/smpl_registration/fitting/joints.py`.

## Optional 2D Keypoints

The pipeline can also triangulate OpenPose-format 2D detections laid out as:

```text
keypoints_2d/
  virtual_000/
    000000_keypoints.json
  virtual_001/
    000000_keypoints.json
```

Each JSON file follows the CMU OpenPose `people[0]` layout. The package does not
ship OpenPose model weights.
