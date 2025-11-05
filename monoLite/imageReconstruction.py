import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageReconOptimization(nn.Module):
    def __init__(self, alpha=0.85, smoothness_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.smoothness_weight = smoothness_weight

    def ssim(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
        ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        ssim = ssim_n / (ssim_d + 1e-7)
        return torch.clamp((1 - ssim) / 2, 0, 1)

    def photometric_loss(self, recon, target):
        l1 = torch.abs(recon - target).mean(1, keepdim=True)
        ssim_loss = self.ssim(recon, target)
        return self.alpha * ssim_loss + (1 - self.alpha) * l1

    def smoothness_loss(self, disp, img):
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        smooth_x = grad_disp_x * torch.exp(-grad_img_x)
        smooth_y = grad_disp_y * torch.exp(-grad_img_y)
        return smooth_x.mean() + smooth_y.mean()

    def forward(self, img_tgt, img_srcs, disp_preds, pose_preds, intrinsics):
        # disp_preds: List of predicted inverse depth (multi-scale)
        # img_srcs : List of source images (e.g., [prev, next])
        # pose_preds: List of predicted poses: tgt->src (one per src)
        # intrinsics: Camera matrix
        #
        # For each scale...
        total_loss = 0
        for scale, disp in enumerate(disp_preds):
            tgt_scaled = F.interpolate(img_tgt, size=disp.shape[-2:], mode="bilinear", align_corners=False)
            srcs_scaled = [F.interpolate(src, size=disp.shape[-2:], mode="bilinear", align_corners=False) for src in img_srcs]

            reproj_losses = []
            for src, pose in zip(srcs_scaled, pose_preds):
                recon = self.view_synthesis(src, disp, pose, intrinsics)  # Fill this method!
                reproj_losses.append(self.photometric_loss(recon, tgt_scaled))
            reproj_losses = torch.stack(reproj_losses, dim=0)  # (N_src, B, 1, H, W)
            min_loss, _ = torch.min(reproj_losses, dim=0)

            # Auto-masking: ignore pixels where copying source beats prediction
            identity_reproj = [self.photometric_loss(src, tgt_scaled) for src in srcs_scaled]
            identity_reproj = torch.stack(identity_reproj, dim=0)
            mask = (min_loss < identity_reproj.min(dim=0)[0]).float()
            min_loss = mask * min_loss

            # Smoothness
            smooth = self.smoothness_loss(disp, tgt_scaled)

            total_loss += min_loss.mean() + self.smoothness_weight * smooth

        return total_loss / len(disp_preds)

    def view_synthesis(self, src, disp, pose, K):
        # Implement pixel-to-pixel warping using depth, pose, and camera intrinsics
        # Follows Monodepth2's project_3d/backproject methods or equivalent
        # You must provide this depending on your codebase
        # If not implemented, return src for shape correctness
        return src 
