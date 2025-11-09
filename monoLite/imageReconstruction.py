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
        ref = nn.ReflectionPad2d(1)
        x = ref(x)
        y = ref(y)
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
        ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        ssim = ssim_n / (ssim_d + 1e-9)
        assert not torch.isnan(ssim).any(), "NaN value in  SSIM"
        assert not torch.isinf(ssim).any(), 'Inf value in SSIM'
        ssim = torch.clamp((1 - ssim) / 2, 0, 1)
        return ssim

    def photometric_loss(self, recon, target):
        l1 = torch.abs(recon - target).mean(1, keepdim=True)
        assert not torch.isnan(l1).any(), 'NaN value in l1 Photometric loss'
        assert not torch.isinf(l1).any(), 'Inf value in l1 Photometric loss'
        ssim_loss = self.ssim(recon, target)
        return self.alpha * ssim_loss + (1 - self.alpha) * l1

    def smoothness_loss(self, disp, img):
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        assert not torch.isnan(grad_disp_x).any(), 'NaN value in grad_disp_x smoothness_loss'
        assert not torch.isinf(grad_disp_x).any(), 'InF value in grad_disp_x smoothness_loss'
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        assert not torch.isnan(grad_disp_y).any(), 'NaN value in grad_disp_y smoothness_loss'
        assert not torch.isinf(grad_disp_y).any(), 'InF value in grad_disp_y smoothness_loss'
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        assert not torch.isnan(grad_img_x).any(), 'NaN value in grad_img_x smoothness_loss'
        assert not torch.isinf(grad_img_x).any(), 'InF value in grad_img_x smoothness_loss'
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        assert not torch.isnan(grad_img_y).any(), 'NaN value in grad_img_y smoothness_loss'
        assert not torch.isinf(grad_img_y).any(), 'InF value in grad_img_y smoothness_loss'
        smooth_x = grad_disp_x * torch.exp(-grad_img_x)
        assert not torch.isnan(smooth_x).any(), 'NaN value in smooth_x smoothness_loss'
        assert not torch.isinf(smooth_x).any(), 'InF value in smooth_x smoothness_loss'
        smooth_y = grad_disp_y * torch.exp(-grad_img_y)
        assert not torch.isnan(smooth_y).any(), 'NaN value in smooth_y smoothness_loss'
        assert not torch.isinf(smooth_y).any(), 'InF value in smooth_y smoothness_loss'
        return smooth_x.mean() + smooth_y.mean()

    def pose_vec2mat(self,vec):
        """
        Converts 6DoF pose vectors to 4x4 transformation matrices.
        Args:
            vec: Tensor of shape [B, 6], where the first 3 elements represent the axis-angle rotation vector
                and the last 3 elements represent translation.
        Returns:
            Tensor of shape [B, 4, 4] representing homogeneous transformation matrices.
        """
        B = vec.shape[0]
        rot_vec = vec[:, :3]
        trans = vec[:, 3:].unsqueeze(-1)  # [B, 3, 1]

        # Compute rotation angle and axis
        angle = torch.norm(rot_vec, dim=1, keepdim=True)  # [B,1]
        axis = rot_vec / (angle + 1e-7)                   # [B,3]
        assert not torch.isnan(angle).any(), 'NaN angle in pose_vec2mat'
        assert not torch.isinf(angle).any(), 'InF angle in pose_vec2mat'
        assert not torch.isnan(axis).any(), 'NaN axis in pose_vec2mat'
        assert not torch.isinf(axis).any(), 'InF axis in pose_vec2mat'

        cos = torch.cos(angle).unsqueeze(-1)              # [B,1,1]
        sin = torch.sin(angle).unsqueeze(-1)              # [B,1,1]
        assert not torch.isinf(cos).any(), 'InF cos in pose_vec2mat'
        assert not torch.isnan(cos).any(), 'NaN cos in pose_vec2mat'
        assert not torch.isinf(sin).any(), 'InF sin in pose_vec2mat'
        assert not torch.isnan(sin).any(), 'NaN sin in pose_vec2mat'

        # Identity matrix
        I = torch.eye(3, device=vec.device).unsqueeze(0).repeat(B, 1, 1)  # [B,3,3]
        assert not torch.isinf(I).any(), 'InF I in pose_vec2mat'
        assert not torch.isnan(I).any(), 'NaN I in pose_vec2mat'

        # Outer product of axis vector
        outer = axis.unsqueeze(2) @ axis.unsqueeze(1)   # [B,3,3]

        # Skew-symmetric cross-product matrix for axis
        skew = torch.zeros(B, 3, 3, device=vec.device)
        skew[:, 0, 1] = -axis[:, 2]
        skew[:, 0, 2] = axis[:, 1]
        skew[:, 1, 0] = axis[:, 2]
        skew[:, 1, 2] = -axis[:, 0]
        skew[:, 2, 0] = -axis[:, 1]
        skew[:, 2, 1] = axis[:, 0]

        # Rodrigues' rotation formula
        R = cos * I + (1 - cos) * outer + sin * skew  # [B,3,3]
        assert not torch.isinf(R).any(), 'InF R in pose_vec2mat'
        assert not torch.isnan(R).any(), 'NaN R in pose_vec2mat'

        # Combine rotation and translation into homogeneous transform
        Rt = torch.cat([R, trans], dim=2)           # [B,3,4]
        assert not torch.isinf(Rt).any(), 'InF Rt in pose_vec2mat'
        assert not torch.isnan(Rt).any(), 'NaN Rt in pose_vec2mat'

        bottom_row = torch.tensor([0, 0, 0, 1], device=vec.device, dtype=vec.dtype).view(1,1,4).repeat(B,1,1)
        assert not torch.isinf(bottom_row).any(), 'InF bottom_row in pose_vec2mat'
        assert not torch.isnan(bottom_row).any(), 'NaN bottom_row in pose_vec2mat'
        T = torch.cat([Rt, bottom_row], dim=1)      # [B,4,4]
        assert not torch.isinf(T).any(), 'InF T in pose_vec2mat'
        assert not torch.isnan(T).any(), 'NaN T in pose_vec2mat'
        return T


    def forward(self, img_tgt, img_srcs, disp_preds, pose_preds, intrinsics):
        # disp_preds: List of predicted inverse depth (multi-scale)
        # img_srcs : List of source images (e.g., [prev, next])
        # pose_preds: List of predicted poses: tgt->src (one per src)
        # intrinsics: Camera matrix
        #
        # For each scale...
        for i, src in enumerate(img_srcs):
            assert not torch.isnan(src).any(), f'NaN in img_srcs[{i}]'
            assert not torch.isinf(src).any(), f'Inf in img_srcs[{i}]'
        for i, disp in enumerate(disp_preds):
            assert not torch.isnan(disp).any(), f'NaN in disp_preds[{i}]'
            assert not torch.isinf(disp).any(), f'Inf in disp_preds[{i}]'
        for i, pose in enumerate(pose_preds):
            assert not torch.isnan(pose).any(), f'NaN in pose_preds[{i}]'
            assert not torch.isinf(pose).any(), f'Inf in pose_preds[{i}]'

        for scale, disp in enumerate(disp_preds):
            tgt_scaled = F.interpolate(img_tgt, size=disp.shape[-2:], mode="bilinear", align_corners=False)
            srcs_scaled = \
                [F.interpolate(src, size=disp.shape[-2:], mode="bilinear", align_corners=False) for src in img_srcs]

            reproj_losses = []
            pose_matrices = [self.pose_vec2mat(pose) for pose in pose_preds]
            for src, pose in zip(srcs_scaled, pose_matrices):
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

    def backproject(self,depth, K_inv):
        """
        Backproject depth image to 3D camera coordinates
        depth: [B, 1, H, W]
        K_inv: [B, 3, 3]
        returns: [B, 3, H*W]
        """
        B, _, H, W = depth.shape
        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device=depth.device),
            torch.arange(H, device=depth.device),
            indexing='xy'
        )
        grid = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=0).float()  # [3, H, W]
        assert not torch.isinf(grid).any(), 'InF grid in backproject'
        assert not torch.isnan(grid).any(), 'NaN grid in backproject'
        grid = grid.view(3, -1).unsqueeze(0).repeat(B, 1, 1)  # [B, 3, H*W]
        cam_points = K_inv.bmm(grid) * depth.view(B, 1, -1)  # scale rays by depth
        assert not torch.isinf(cam_points).any(), 'InF cam_points in backproject'
        assert not torch.isnan(cam_points).any(), 'NaN cam_points in backproject'
        return cam_points


    def project(self,cam_points, K, T, H, W):
        """
        Project 3D camera points into a camera frame with pose T and intrinsics K.
        cam_points: [B, 3, N]
        K: [B, 3, 3]
        T: [B, 4, 4]  (relative pose from tgt to src)
        Returns normalized grid [B, H, W, 2] for grid_sample
        """
        B, _, N = cam_points.shape

        # Convert cam_points [B,3,N] to homogeneous [B,4,N]
        ones = torch.ones(B, 1, N, device=cam_points.device)
        homo_points = torch.cat([cam_points, ones], dim=1)

        # Apply transformation T: src_points = T * tgt_points
        ''' print(f"T shape: {T.shape}")  # Should be [B,4,4]
            print(f"homo_points shape: {homo_points.shape}")  # Should be [B,4,N] '''
        if T.dim() == 2:
            T = T.unsqueeze(0).repeat(B, 1, 1)  # Repeat for batch
        assert T.dim() == 3, "Pose tensor T must be 3D to batch-matrix multiply"


        src_points = T.bmm(homo_points)[:, :3, :]  # [B, 3, N]
        assert not torch.isnan(src_points).any(), "NaN in src_points in project"
        assert not torch.isinf(src_points).any(), "Inf in src_points in project"


        # Project to camera coordinates (scale by Z)
        pix_coords = K.bmm(src_points.float())  # [B, 3, N]
        # print("Min Z =", pix_coords[:, 2, :].min())
        # print("Max Z =", pix_coords[:, 2, :].max())
        assert not torch.isinf(pix_coords).any(), 'InF pix_coords in project'
        assert not torch.isnan(pix_coords).any(), 'NaN pix_coords in project'
        pix_xy = pix_coords[:, :2, :] / (pix_coords[:, 2:3, :] + 1e-4)

        # Normalize to [-1, 1] for grid sample
        x_norm = 2 * (pix_xy[:, 0, :] / (W - 1)) - 1
        y_norm = 2 * (pix_xy[:, 1, :] / (H - 1)) - 1

        grid = torch.stack([x_norm, y_norm], dim=2).view(B, H, W, 2)
        grid = torch.clamp(grid, -1e3, 1e3)
        assert not torch.isinf(grid).any(), f"Grid stats: min={grid.min().item()}, max={grid.max().item()}, mean={grid.mean().item()}"
        assert not torch.isnan(grid).any(), f"Grid stats: min={grid.min().item()}, max={grid.max().item()}, mean={grid.mean().item()}"
        return grid


    def view_synthesis(self, src, disp, pose, K):
        disp = torch.clamp(disp, min=1e-4, max=1.0)
        B, _, H, W = disp.shape
        depth = 1.0 / disp
        depth = torch.clamp(depth, max=100)
        K_inv = torch.inverse(K)

        cam_points = self.backproject(depth, K_inv)  # [B,3,H*W]

        # Check pose tensor dims; expand if needed (for single pose under batch)
        if pose.dim() == 2:
            pose = pose.unsqueeze(0).repeat(B, 1, 1)  # [B,4,4]

        grid = self.project(cam_points, K, pose, H, W)
        grid = torch.clamp(grid, -1, 1)
        warped_src = F.grid_sample(src, grid, padding_mode='border', align_corners=True)

        return warped_src
