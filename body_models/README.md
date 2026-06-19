# Body Model Assets

This folder is intentionally empty in git. Downloaded SMPL-family model files
are restricted third-party assets, so keep them local and do not commit them.

The starter config points `body_model.model_path` here:

```yaml
body_model:
  model_path: body_models
```

Expected layout:

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
  transfer/
    smplx2smpl_deftrafo_setup.pkl
    smpl2smplx_deftrafo_setup.pkl
```

You only need the model type and gender you actually use. The default config
fits neutral SMPL-X, so the minimal default setup is one neutral SMPL-X model in
`body_models/smplx/SMPLX_NEUTRAL.npz`.

Download sources:

- SMPL-X: https://smpl-x.is.tue.mpg.de
- SMPL: https://smpl.is.tue.mpg.de
- neutral SMPL model used by some SMPLify workflows: https://smplify.is.tue.mpg.de
- SMPL-H / MANO assets: https://mano.is.tue.mpg.de

For conversion between SMPL-X and SMPL, also download the SMPL-X model
correspondences from the SMPL-X downloads page and place the needed transfer
files in `body_models/transfer/`.

## Removing chumpy from SMPL `.pkl` files

The downloaded SMPL and SMPL-H `.pkl` files (e.g. `SMPL_NEUTRAL.pkl`,
`basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`) store some fields as `chumpy`
arrays, so loading them raises `ModuleNotFoundError: No module named 'chumpy'`
(and `chumpy` itself breaks on modern numpy). Strip chumpy out once with the
helper script — values are preserved exactly, only the container type changes,
and the result loads with no chumpy dependency:

```bash
# convert a single file in place
python scripts/dechumpify_models.py body_models/smpl/SMPL_NEUTRAL.pkl

# or convert every .pkl under the folder in one pass
python scripts/dechumpify_models.py body_models/
```

Use `--dry-run` to see which fields are chumpy without writing. The `.npz`
SMPL-X models do not need this step.
