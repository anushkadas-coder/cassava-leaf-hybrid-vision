import pytest
import torch

from cassava.models.cnn import SmallCNN
from cassava.models.factory import build_model, count_parameters
from cassava.models.fusion import FusionNet
from cassava.models.transfer import TransferNet
from cassava.config import DEFAULTS
import copy


@pytest.mark.parametrize("ch", [3, 4, 6])
def test_cnn_output_and_size(ch):
    m = SmallCNN(ch, 5)
    assert m(torch.randn(2, ch, 48, 48)).shape == (2, 5)
    assert count_parameters(m) < 200_000


@pytest.mark.parametrize("arch", ["resnet18", "efficientnet_b0"])
def test_transfer_non_rgb_input(arch):
    m = TransferNet(6, 5, arch, pretrained=False)
    assert m(torch.randn(2, 6, 64, 64)).shape == (2, 5)


def test_transfer_freezing():
    m = TransferNet(3, 5, "resnet18", pretrained=False)
    m.set_backbone_frozen(True)
    assert not any(p.requires_grad for p in m.body.parameters()) and all(p.requires_grad for p in m.head.parameters())
    m.train()
    assert not m.body.training  # BN stats frozen with the backbone
    m.set_backbone_frozen(False)
    assert all(p.requires_grad for p in m.body.parameters())


@pytest.mark.parametrize("control", ["none", "shuffle", "zeros"])
def test_fusion_shapes_and_controls(control):
    m = FusionNet(SmallCNN(3, 5), 20, 5, control=control).eval()
    x, f = torch.randn(4, 3, 32, 32), torch.randn(4, 20)
    assert m(x, f).shape == (4, 5)
    if control == "zeros":
        assert torch.allclose(m(x, f), m(x, f * 100))  # frequency input is ignored


def test_fusion_requires_frequency_features():
    m = FusionNet(SmallCNN(3, 5), 20, 5)
    with pytest.raises(ValueError):
        m(torch.randn(2, 3, 32, 32), torch.zeros(2, 0))
    with pytest.raises(ValueError):
        FusionNet(SmallCNN(3, 5), 20, 5, control="bogus")


def test_factory_variants():
    cfg = copy.deepcopy(DEFAULTS)
    cfg["model"]["transfer"]["pretrained"] = False
    cfg["data"]["representation"] = "lab"
    for kind in ("cnn", "transfer", "fusion"):
        cfg["model"]["kind"] = kind
        assert build_model(cfg, 5) is not None
    cfg["model"]["kind"] = "classical"
    with pytest.raises(ValueError):
        build_model(cfg, 5)
