import torch

from ultralytics.utils.torch_utils import strip_optimizer, torch_load_compat


def test_strip_optimizer_loads_a_trusted_legacy_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": torch.nn.Linear(2, 1), "train_args": {}}, checkpoint)

    strip_optimizer(checkpoint)
    loaded = torch_load_compat(checkpoint, map_location="cpu", weights_only=False)

    assert loaded["optimizer"] is None
    assert loaded["epoch"] == -1
