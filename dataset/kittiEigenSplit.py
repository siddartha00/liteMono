import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pytorch_lightning as pl


# ---------------------------
# Helper Functions
# ---------------------------

def read_image(path, resize=(640, 192)):
    """Load image, resize, and convert to tensor."""
    img = Image.open(path).convert('RGB')
    img = img.resize(resize, Image.BILINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45],
                             std=[0.225, 0.225, 0.225])
    ])
    return transform(img)


def read_intrinsics(path, img_size=(640, 192)):
    """
    Parse KITTI camera calibration file (.txt) and return 3x3 intrinsic matrix.
    This reads 'P_rect_02' (left color camera) and rescales intrinsics 
    to match resized image size.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    P_rect_02 = None
    for l in lines:
        if l.startswith('P_rect_02:'):
            vals = [float(x) for x in l.strip().split()[1:]]
            P_rect_02 = np.reshape(vals, (3, 4))[:, :3]
            break
    if P_rect_02 is None:
        raise ValueError(f"Couldn't find P_rect_02 in {path}")

    # Original KITTI images are (1242x375)
    orig_w, orig_h = 1242, 375
    new_w, new_h = img_size
    sx, sy = new_w / orig_w, new_h / orig_h

    # Scale intrinsics
    K = np.copy(P_rect_02)
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return torch.tensor(K, dtype=torch.float32)


# ---------------------------
# Dataset
# ---------------------------

class KITTISelfSupDataset(Dataset):
    """
    Reads rows from train_selfsup.csv or val_selfsup.csv, returning:
    {
      'img_tgt': Tensor [3,H,W],
      'img_srcs': list of Tensor [3,H,W],
      'K': Tensor [3,3],
      'gt_depth': Optional Tensor [1,H,W]
    }
    """
    def __init__(self, csv_path, load_depth=False, resize=(640, 192)):
        self.samples = []
        self.load_depth = load_depth
        self.resize = resize

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or len(row) < 4:
                    continue
                target = row['target_filepath']
                src1 = row['src1_filepath']
                src2 = row['src2_filepath']
                srcs = [src1, src2]
                intrinsics = row['camera_intrinsics_path']
                depth = row['depth_filepath']
                self.samples.append({
                    'target': target,
                    'srcs': srcs,
                    'intrinsics': intrinsics,
                    'depth': depth
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_tgt = read_image(s['target'], self.resize)
        img_srcs = [read_image(p, self.resize) for p in s['srcs']]
        K = read_intrinsics(s['intrinsics'], self.resize)

        sample = {
            'img_tgt': img_tgt,
            'img_srcs': img_srcs,
            'K': K
        }

        # optional ground-truth depth
        if self.load_depth and s['depth'] is not None and os.path.exists(s['depth']):
            depth = np.array(Image.open(s['depth'])).astype(np.float32)
            depth = Image.fromarray(depth)
            depth = depth.resize(self.resize, Image.NEAREST)
            depth = torch.from_numpy(np.array(depth)).unsqueeze(0)
            sample['gt_depth'] = depth

        return sample


# ---------------------------
# DataModule
# ---------------------------

class KITTISelfSupDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='data/splits', batch_size=4, num_workers=4, load_depth=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_depth = load_depth

    def setup(self, stage=None):
        train_csv = os.path.join(self.data_dir, 'train_selfsup.csv')
        val_csv = os.path.join(self.data_dir, 'val_selfsup.csv')
        test_csv = os.path.join(self.data_dir, 'test_selfsup.csv')

        self.train_dataset = KITTISelfSupDataset(train_csv, load_depth=self.load_depth)
        self.val_dataset = KITTISelfSupDataset(val_csv, load_depth=self.load_depth)
        self.test_dataset = KITTISelfSupDataset(test_csv, load_depth=self.load_depth)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          collate_fn=self._collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          collate_fn=self._collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          collate_fn=self._collate_fn)

    @staticmethod
    def _collate_fn(batch):
        """
        Handles variable number of src frames per sample.
        Returns list of dicts stacked appropriately.
        """
        imgs_tgt = torch.stack([b['img_tgt'] for b in batch])
        Ks = torch.stack([b['K'] for b in batch])
        # Transpose list-of-lists: [[src1_i, src2_i], ...] -> per-source list
        num_srcs = len(batch[0]['img_srcs'])
        img_srcs = [
            torch.stack([b['img_srcs'][i] for b in batch])
            for i in range(num_srcs)
        ]
        collated = {'img_tgt': imgs_tgt, 'img_srcs': img_srcs, 'K': Ks}
        if 'gt_depth' in batch[0]:
            collated['gt_depth'] = torch.stack([b['gt_depth'] for b in batch])
        return collated
