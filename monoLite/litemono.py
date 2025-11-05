import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from encoder import LiteMonoEncoder
from decoder import LiteMonoDecoder
from posnet import PoseNet

class LiteMono(pl.LightningModule):
    def __init__(self, encoder_variant="base", decoder_channels=None, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = LiteMonoEncoder(variant=encoder_variant)
        # decoder_channels: e.g., [C3, C2, C1] matching encoder's outputs
        if decoder_channels is None:
            # You should adjust depending on encoder variant specifics
            decoder_channels = [128, 128, 48]
        self.decoder = LiteMonoDecoder(decoder_channels)
        self.posenet = PoseNet()
        self.lr = lr

    def forward(self, x):
        feats = self.encoder(x)
        preds = self.decoder(feats)
        return preds  # list: full, 1/2, 1/4 inverse depth

    def training_step(self, batch, batch_idx):
        # batch: monocular triplet (img_tgt, img_src, camera intrinsics)
        img_tgt, img_src, K = batch["img_tgt"], batch["img_src"], batch["K"]  # adjust keys as needed
        # Depth prediction (target frame)
        feats = self.encoder(img_tgt)
        inv_depth_preds = self.decoder(feats)
        # Pose prediction (between pairs)
        pose_inputs = torch.cat([img_tgt, img_src], dim=1)  # [B,6,H,W]
        pose = self.posenet(pose_inputs)
        # Photometric reconstruction loss (e.g. as in Monodepth2) —
        # synthesize target from source using predicted depth, predicted pose, and K
        photometric_loss = self.compute_photometric_loss(img_tgt, img_src, inv_depth_preds[0], pose, K)
        # Edge-aware smoothness loss for depth map
        smooth_loss = self.compute_smoothness_loss(inv_depth_preds[0], img_tgt)
        loss = photometric_loss + 0.1 * smooth_loss  # weight smoothness as in paper
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img_tgt, img_src, K = batch["img_tgt"], batch["img_src"], batch["K"]
        feats = self.encoder(img_tgt)
        inv_depth_preds = self.decoder(feats)
        # Calculate RMSE or AbsRel or any depth metric against ground truth if available
        # Here, let's assume batch["gt_depth"] is ground-truth depth
        if "gt_depth" in batch:
            depth_pred = 1 / (inv_depth_preds[0] + 1e-8)  # inverse depth to depth
            gt_depth = batch["gt_depth"]
            mask = gt_depth > 0
            abs_rel = torch.mean(torch.abs(depth_pred[mask] - gt_depth[mask]) / gt_depth[mask])
            self.log("val_absrel", abs_rel)
        return

    def test_step(self, batch, batch_idx):
        # Similar to validation_step — call self.forward and report any test metrics
        img_tgt = batch["img_tgt"]
        feats = self.encoder(img_tgt)
        inv_depth_preds = self.decoder(feats)
        # Optionally return predictions for evaluation
        return {"inv_depth": inv_depth_preds[0]}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def compute_photometric_loss(self, img_tgt, img_src, inv_depth, pose, K):
        # Placeholder for real implementation: warp img_src to img_tgt using depth & pose
        # Calculate photometric error; see Monodepth2/LiteMono repo for details
        # Return scalar photometric loss
        return torch.tensor(0.0, device=img_tgt.device)  # Replace with real loss

    def compute_smoothness_loss(self, inv_depth, img):
        # Edge-aware depth smoothness loss, e.g.:
        grad_inv_depth_x = torch.abs(inv_depth[:, :, :, :-1] - inv_depth[:, :, :, 1:])
        grad_inv_depth_y = torch.abs(inv_depth[:, :, :-1, :] - inv_depth[:, :, 1:, :])
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        weight_x = torch.exp(-grad_img_x)
        weight_y = torch.exp(-grad_img_y)
        smooth_x = grad_inv_depth_x * weight_x
        smooth_y = grad_inv_depth_y * weight_y
        return torch.mean(smooth_x) + torch.mean(smooth_y)
