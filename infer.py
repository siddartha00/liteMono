#!/usr/bin/env python3
"""
Inference script for LiteMono depth estimation.
Usage:
    python inference_litemono.py \
        --checkpoint ./checkpoints/litemono.ckpt \
        --input ./images/sample.png \
        --output ./outputs/depth.png \
        --device cuda
"""

import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

from monoLite.litemono import LiteMonoSystem  # ← your model class (adjust import if needed)

# ------------------------------
# Utility Functions
# ------------------------------
def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return img

def preprocess(img):
    """Convert PIL image to normalized tensor [1,3,H,W]."""
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0,1]
    ])
    return transform(img).unsqueeze(0)


def postprocess_colormap(depth):
    """
    Convert depth tensor to a color-mapped PIL image.
    """
    depth = depth.squeeze().cpu().numpy()  # [H, W]
    depth = depth - depth.min()
    depth = depth / (depth.max() + 1e-8)

    # Apply matplotlib colormap
    cmap = plt.get_cmap("plasma")  # spectral-like colormap
    depth_colored = cmap(depth)[:, :, :3]  # drop alpha channel
    depth_colored = (depth_colored * 255).astype(np.uint8)
    
    return Image.fromarray(depth_colored)

# ------------------------------
# Main Inference Function
# ------------------------------
def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.checkpoint}...")
    model = LiteMonoSystem.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval().to(device)

    # Support both single image and directory
    input_paths = []
    if os.path.isdir(args.input):
        input_paths = [os.path.join(args.input, f)
                       for f in os.listdir(args.input)
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    else:
        input_paths = [args.input]

    os.makedirs(args.output, exist_ok=True) if os.path.isdir(args.input) else None

    for path in input_paths:
        print(f"Inferencing: {path}")
        img = load_image(path)
        tensor = preprocess(img).to(device)

        with torch.no_grad():
            preds = model(tensor)

            # Take the highest-resolution output if multiple scales are returned
            if isinstance(preds, list):
                pred_depth = preds[0]
            else:
                pred_depth = preds

            # Convert disparity to depth (if applicable)
            pred_depth = 1.0 / (pred_depth + 1e-6)

            if pred_depth.ndim == 3:
                pred_depth = pred_depth.unsqueeze(1)

            pred_depth = F.interpolate(
                pred_depth, size=img.size[::-1], mode='bilinear', align_corners=False
            )


        # Old: depth_vis = postprocess(pred_depth)
        depth_vis = postprocess_colormap(pred_depth)  # use color mapping

        # Save
        if os.path.isdir(args.input):
            fname = os.path.basename(path)
            save_path = os.path.join(args.output, f"depth_{fname}")
        else:
            save_path = args.output

        depth_vis.save(save_path)
        print(f"Saved colored depth map → {save_path}")


# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiteMono Depth Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt checkpoint file")
    parser.add_argument("--input", type=str, required=True, help="Path to image or folder")
    parser.add_argument("--output", type=str, default="./outputs/depth.png", help="Output file or directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    run_inference(args)
