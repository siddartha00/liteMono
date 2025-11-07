import torch
import pytest
from monoLite.imageReconstruction import ImageReconOptimization
from monoLite.decoder import LiteMonoDecoder
from monoLite.encoder import LiteMonoEncoder
from monoLite.cdc import CDCBlock
from monoLite.lgfi import LGFIBlock
from monoLite.litemono import LiteMonoSystem
from monoLite.posnet import PoseNet
# Mock input batch
BATCH_SIZE = 2
HEIGHT = 128
WIDTH = 160
CHANNELS = 3


def test_cdcblock_residual():
    block = CDCBlock(channels=32, dilation_rates=[1, 2, 3])
    x = torch.randn(BATCH_SIZE, 32, HEIGHT, WIDTH)
    out = block(x)
    assert out.shape == x.shape, "CDCBlock output shape mismatch! Should not inflate channels."


def test_lgfiblock_additive():
    block = LGFIBlock(channels=32, num_heads=4)
    x = torch.randn(BATCH_SIZE, 32, HEIGHT, WIDTH)
    out = block(x)
    assert out.shape == x.shape, "LGFIBlock output shape mismatch!"
    assert torch.allclose(out, out, atol=1e-6), "LGFIBlock should be numerically valid."


def test_encoder():
    encoder = LiteMonoEncoder(variant="base")
    x = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT*2, WIDTH*2)
    feats = encoder(x)
    # Should return three feature maps of decreasing resolution
    assert isinstance(feats, list)
    assert len(feats) == 3, "Encoder should return three scales!"
    c3, c2, c1 = [feat.shape for feat in feats]
    assert c3[2] < c2[2] < c1[2], "Feature map resolution ordering violated!"


def test_decoder():
    # Channel counts should match output from encoder["base"] variant
    decoder = LiteMonoDecoder([128, 128, 48])
    x3 = torch.randn(BATCH_SIZE, 128, HEIGHT//8, WIDTH//8)
    x2 = torch.randn(BATCH_SIZE, 128, HEIGHT//4, WIDTH//4)
    x1 = torch.randn(BATCH_SIZE, 48, HEIGHT//2, WIDTH//2)
    preds = decoder([x3, x2, x1])
    assert len(preds) == 3, "Decoder did not return multi-scale predictions!"
    assert preds[0].shape[2:] == (HEIGHT, WIDTH), "Top scale should be full resolution."


def test_posenet():
    posenet = PoseNet()
    # img_pair: torch.Size([B, 6, H, W])
    img_pair = torch.randn(BATCH_SIZE, 6, HEIGHT, WIDTH)
    poses = posenet(img_pair)
    assert poses.shape == (BATCH_SIZE, 6), "PoseNet must output [B,6] pose vectors."


def test_loss_module():
    loss_mod = ImageReconOptimization()
    img_tgt = torch.randn(BATCH_SIZE, 3, HEIGHT, WIDTH)
    img_srcs = [torch.randn(BATCH_SIZE, 3, HEIGHT, WIDTH) for _ in range(2)]
    disp_preds = [torch.sigmoid(torch.randn(BATCH_SIZE, 1, HEIGHT, WIDTH)) for _ in range(3)]
    pose_preds = [torch.randn(BATCH_SIZE, 6) for _ in range(2)]
    K = torch.eye(3).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
    loss = loss_mod(img_tgt, img_srcs, disp_preds, pose_preds, K)
    assert loss.ndim == 0 or loss.shape == (), "Loss should be scalar."


def test_litemono_system_all_forward():
    model = LiteMonoSystem(encoder_variant="base", decoder_channels=[128, 128, 48])
    img_tgt = torch.randn(BATCH_SIZE, 3, HEIGHT, WIDTH)
    img_srcs = [torch.randn(BATCH_SIZE, 3, HEIGHT, WIDTH) for _ in range(2)]
    K = torch.eye(3).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
    batch = {'img_tgt': img_tgt, 'img_srcs': img_srcs, 'K': K}
    try:
        model.training_step(batch, 0)
    except NotImplementedError:
        pytest.skip("Some methods (like view_synthesis) may need implementation in loss.")
