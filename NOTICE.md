# Notices

This repository is source-only. It does not include SMPL-family model files,
OpenPose weights, datasets, generated meshes, or private capture-system assets.

## SMPL / SMPL-H / SMPL-X

Users must obtain SMPL, SMPL-H, or SMPL-X models from the official providers and
agree to the corresponding model licenses before using them with this package.
The local `body_model.model_path` config should point to those files outside
this repository.

Useful official pages:

- https://smpl.is.tue.mpg.de/
- https://smpl-x.is.tue.mpg.de/

## MPI / SMPLify-X

If parts of the fitting code are derived from SMPLify-X or other Max Planck
Institute / Max-Planck-Gesellschaft research code, those parts remain subject to
the applicable upstream non-commercial research license terms and attribution
requirements. Do not remove upstream notices from derived files.

Reference:

- https://github.com/vchoutas/smplify-x

## OpenPose

The optional `mesh2smplx.openpose` package is a code wrapper/migration for
OpenPose-style 135-keypoint detection and JSON parsing. OpenPose model weights
are not shipped here. Users are responsible for obtaining weights under terms
they are allowed to use.
