from pathlib import Path

from smpl_registration.config import load_config
from smpl_registration.frame_selection import parse_frame_range


def test_parse_frame_range():
    assert parse_frame_range("1,3-5") == [1, 3, 4, 5]


def test_load_textured_mesh_config():
    config = load_config(Path("examples/configs/textured_mesh.yaml"))
    assert config.input.mode == "textured_mesh"
    assert config.body_model.type == "smplx"
    assert config.fitting.frames == "0"
    assert config.fitting.keypoint_loss_weight == 1000.0
    assert config.input.mesh_glob == "frame_*.obj"
