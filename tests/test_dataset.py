import pytest
from dataset.kittiEigenSplit import LiteMonoDataModule
from pathlib import Path
import torch

# Paths relative to your project root
ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data/splits/train_selfsup.csv"
VAL_CSV = ROOT / "data/splits/val_selfsup.csv"

@pytest.fixture(scope="module")
def data_module():
    dm = LiteMonoDataModule(
        train_csv=str(TRAIN_CSV),
        val_csv=str(VAL_CSV),
        batch_size=2,
        num_workers=0,  # 0 for Windows testing
        img_size=(128, 416),
        load_depth=True
    )
    dm.setup()
    return dm


def test_train_dataloader_shapes(data_module):
    loader = data_module.train_dataloader()
    batch = next(iter(loader))

    # Target image
    assert isinstance(batch['img_tgt'], torch.Tensor)
    assert batch['img_tgt'].ndim == 4  # [B,3,H,W]
    assert batch['img_tgt'].shape[1] == 3

    # Source images
    for src_list in batch['img_srcs']:
        for src in src_list:
            assert isinstance(src, torch.Tensor)
            assert src.ndim == 3  # [C,H,W]
            assert src.shape[0] == 3

    # Intrinsics
    assert isinstance(batch['K'], torch.Tensor)
    assert batch['K'].shape[1:] == (3, 3)

    # Depth (if available)
    if 'gt_depth' in batch:
        assert batch['gt_depth'].ndim == 4
        assert batch['gt_depth'].shape[1] == 1


def test_val_dataloader_shapes(data_module):
    loader = data_module.val_dataloader()
    batch = next(iter(loader))

    # Target image
    assert isinstance(batch['img_tgt'], torch.Tensor)
    assert batch['img_tgt'].ndim == 4
    assert batch['img_tgt'].shape[1] == 3

    # Source images
    for src_list in batch['img_srcs']:
        for src in src_list:
            assert isinstance(src, torch.Tensor)
            assert src.ndim == 3
            assert src.shape[0] == 3

    # Intrinsics
    assert isinstance(batch['K'], torch.Tensor)
    assert batch['K'].shape[1:] == (3, 3)

    # Depth
    if 'gt_depth' in batch:
        assert batch['gt_depth'].ndim == 4
        assert batch['gt_depth'].shape[1] == 1
