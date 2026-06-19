# Checkpoints

This folder is intentionally empty in git. Runtime checkpoints are large
third-party assets and should not be committed.

The default OpenPose-135 checkpoint folder is:

```text
checkpoints/openpose135/
  body_pose_model_25.pth
  hand_pose_model.pth
  facenet.pth
```

The main pipeline downloads missing OpenPose-135 files from
<https://huggingface.co/hohs/openpose135-weights> into that folder the first time
OpenPose runs. If the download fails, manually place the three `.pth` files
there.

To use a different location, set:

```yaml
keypoints:
  weights_dir: /path/to/openpose135
```
