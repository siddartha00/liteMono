import pytorch_lightning as pl
import torch
from encoder import LiteMonoEncoder
from decoder import LiteMonoDecoder
from posnet import PoseNet
from imageReconstruction import ImageReconOptimization

class LiteMonoSystem(pl.LightningModule):
    def __init__(self, encoder_variant="base", decoder_channels=None, lr=1e-4):
        super().__init__()
        self.encoder = LiteMonoEncoder(variant=encoder_variant)
        # decoder_channels: list matching your final encoder outputs
        if decoder_channels is None:
            decoder_channels = [128, 128, 48]  # example for 'base', adapt to your encoder
        self.decoder = LiteMonoDecoder(decoder_channels)
        self.posenet = PoseNet()
        self.loss_fn = ImageReconOptimization()
        self.lr = lr

    def forward(self, img_tgt):
        features = self.encoder(img_tgt)
        disp_preds = self.decoder(features)
        return disp_preds

    def training_step(self, batch, batch_idx):
        img_tgt = batch['img_tgt']         # [B,3,H,W]
        img_srcs = batch['img_srcs']       # e.g. list of [B,3,H,W] frames before/after
        intrinsics = batch['K']            # [B,3,3], camera matrix for all frames
        # Predict depth/disparity for target
        disp_preds = self(img_tgt)         # list of multi-scale outputs
        # Predict pose for each src (from tgt to src)
        pose_preds = []
        for img_src in img_srcs:
            pose_input = torch.cat([img_tgt, img_src], dim=1)  # [B,6,H,W]
            pose = self.posenet(pose_input)
            pose_preds.append(pose)
        # Calculate loss using the loss module
        loss = self.loss_fn(img_tgt, img_srcs, disp_preds, pose_preds, intrinsics)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img_tgt = batch['img_tgt']
        img_srcs = batch['img_srcs']
        intrinsics = batch['K']
        disp_preds = self(img_tgt)
        pose_preds = [self.posenet(torch.cat([img_tgt, src], dim=1)) for src in img_srcs]
        loss = self.loss_fn(img_tgt, img_srcs, disp_preds, pose_preds, intrinsics)
        self.log('val_loss', loss)
        # For depth metrics: (if batch has gt)
        if 'gt_depth' in batch:
            depth_pred = 1/(disp_preds[0]+1e-8)
            gt_depth = batch['gt_depth']
            mask = gt_depth > 0
            absrel = torch.mean(torch.abs(depth_pred[mask] - gt_depth[mask]) / gt_depth[mask])
            self.log('val_absrel', absrel)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
