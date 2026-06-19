# Checkpoints

This folder is intentionally empty in git. Runtime model checkpoints are
restricted or large third-party assets and should stay local.

The default OpenPose-135 checkpoint folder is:

```text
checkpoints/openpose135/
  body_pose_model_25.pth
  hand_pose_model.pth
  facenet.pth
```

The main pipeline auto-fetches missing OpenPose-135 files from
<https://huggingface.co/hohs/openpose135-weights> into that folder. You can
override the location with:

```yaml
keypoints:
  weights_dir: /path/to/openpose135
```
