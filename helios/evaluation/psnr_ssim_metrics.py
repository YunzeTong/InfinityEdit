"""PSNR and SSIM metrics for edit adapter validation.

Provides:
  - compute_psnr: per-frame PSNR between edited video and ground truth
  - compute_ssim: per-frame SSIM between edited video and ground truth

All functions accept uint8 numpy arrays (T, H, W, C) in [0, 255] RGB.
No external model downloads needed — pure PyTorch implementation.
"""

from math import exp

import torch
import torch.nn.functional as F


def compute_psnr(edited_video, gt_video):
    """Per-frame PSNR between edited and GT videos, returns mean.

    Args:
        edited_video: uint8 numpy (T1, H, W, C) [0, 255]
        gt_video: uint8 numpy (T2, H, W, C) [0, 255]
    Returns:
        float — mean PSNR in dB across min(T1, T2) frames (higher = more similar)
    """
    T = min(len(edited_video), len(gt_video))
    ed = torch.from_numpy(edited_video[:T]).float() / 255.0
    gt = torch.from_numpy(gt_video[:T]).float() / 255.0
    mse = ((ed - gt) ** 2).mean(dim=(1, 2, 3))  # per-frame MSE
    psnr_per_frame = -10.0 * torch.log10(mse + 1e-10)
    return psnr_per_frame.mean().item()


def compute_ssim(edited_video, gt_video, device, window_size=11):
    """Per-frame SSIM between edited and GT videos, returns mean.

    Args:
        edited_video: uint8 numpy (T1, H, W, C) [0, 255]
        gt_video: uint8 numpy (T2, H, W, C) [0, 255]
        device: torch device
        window_size: Gaussian window size for SSIM computation
    Returns:
        float — mean SSIM across min(T1, T2) frames (higher = more similar, max 1.0)
    """
    T = min(len(edited_video), len(gt_video))
    # (T, C, H, W) float [0, 1]
    ed = torch.from_numpy(edited_video[:T]).permute(0, 3, 1, 2).float().to(device) / 255.0
    gt = torch.from_numpy(gt_video[:T]).permute(0, 3, 1, 2).float().to(device) / 255.0

    C = ed.shape[1]
    window = _create_window(window_size, C).to(device)

    ssim_vals = []
    for t in range(T):
        s = _ssim_single(ed[t : t + 1], gt[t : t + 1], window, window_size, C)
        ssim_vals.append(s)
    return sum(ssim_vals) / len(ssim_vals)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-((x - window_size // 2) ** 2) / (2.0 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    _1D = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D.expand(channel, 1, window_size, window_size).contiguous()


def _ssim_single(img1, img2, window, window_size, channel):
    """SSIM for a single (1, C, H, W) pair. Returns float."""
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    # Constants for L=1.0 (inputs normalised to [0, 1])
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()
