"""Shared helpers for the Edit Adapter training script.

Contains model unwrapping, video saving, validation-sample selection, and the
multi-stage pyramid DMD denoise used by the pyramid-continuation validation.
"""

import math

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger

from helios.utils.utils_base import calculate_shift

logger = get_logger(__name__)


def unwrap_model(model, accelerator=None):
    model = accelerator.unwrap_model(model) if accelerator else model
    from diffusers.utils.torch_utils import is_compiled_module
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_video(frames, path, fps=16):
    """Save video frames with correct full-range color.

    Args:
        frames: numpy array of shape (T, H, W, C), uint8, RGB, range [0, 255].
        path: output .mp4 path.
        fps: frames per second.
    """
    with imageio.get_writer(
        path, fps=fps, quality=None,
        output_params=["-pix_fmt", "yuvj420p", "-crf", "18"],
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))


def _select_validation_indices(train_dataset, args, mode="generation"):
    """Select validation sample indices that cover all edit instructions.

    When multiple data_dirs are used, picks N samples from each instruction
    (identified by dataset_name). Falls back to random selection when only one
    instruction is present.

    Args:
        mode: "generation" uses num_val_samples_per_instruction (fewer, expensive),
              "val_loss" uses num_val_loss_samples_per_instruction (more, cheap).

    Returns:
        List of dataset indices for validation.
    """
    from collections import defaultdict
    instruction_to_indices = defaultdict(list)
    for idx, sample in enumerate(train_dataset.samples):
        instruction_to_indices[sample["dataset_name"]].append(idx)

    num_instructions = len(instruction_to_indices)
    if mode == "val_loss":
        num_per_instruction = args.validation_config.num_val_loss_samples_per_instruction
    else:
        num_per_instruction = args.validation_config.num_val_samples_per_instruction

    if num_instructions <= 1:
        if mode == "val_loss":
            num_samples = min(num_per_instruction, len(train_dataset))
        else:
            num_samples = min(
                args.validation_config.num_validation_videos, len(train_dataset)
            )
        rng = torch.Generator().manual_seed(42)
        all_indices = torch.randperm(len(train_dataset), generator=rng).tolist()
        return all_indices[:num_samples]

    rng = torch.Generator().manual_seed(42)
    sample_indices = []
    for instruction_name in sorted(instruction_to_indices.keys()):
        indices = instruction_to_indices[instruction_name]
        perm = torch.randperm(len(indices), generator=rng).tolist()
        selected = [indices[p] for p in perm[:num_per_instruction]]
        sample_indices.extend(selected)

    logger.info(
        f"  Validation [{mode}] covers {num_instructions} instructions, "
        f"{num_per_instruction} samples each, {len(sample_indices)} total"
    )
    return sample_indices


def _sample_block_noise(
    batch_size, channel, num_frames, height, width,
    patch_size, gamma, device, generator,
):
    """Patch-correlated noise, matching pipeline.sample_block_noise."""
    _, ph, pw = patch_size
    block_size = ph * pw

    cov = (
        torch.eye(block_size, device=device) * (1 + gamma)
        - torch.ones(block_size, block_size, device=device) * gamma
    )
    cov += torch.eye(block_size, device=device) * 1e-8
    cov = cov.float()

    L = torch.linalg.cholesky(cov)
    block_number = batch_size * channel * num_frames * (height // ph) * (width // pw)
    z = torch.randn(block_number, block_size, generator=generator, device=device)
    noise = z @ L.T

    noise = noise.view(batch_size, channel, num_frames, height // ph, width // pw, ph, pw)
    noise = noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(batch_size, channel, num_frames, height, width)
    return noise


def _run_stage2_denoise(
    transformer, scheduler, latents,
    prompt_embeds, edit_instruction_embeds,
    indices_hidden_states,
    indices_latents_history_short,
    indices_latents_history_mid,
    indices_latents_history_long,
    latents_history_short,
    latents_history_mid,
    latents_history_long,
    pyramid_num_stages, pyramid_num_inference_steps_list,
    patch_size, device, weight_dtype, generator,
):
    """Multi-stage pyramid DMD denoising, matching pipeline stage2_sample."""
    batch_size, num_channel, num_frames, target_h, target_w = latents.shape
    gamma = scheduler.config.gamma
    p_h, p_w = patch_size[1], patch_size[2]

    height, width = target_h, target_w

    # ── Downsample to lowest resolution ──
    latents = latents.permute(0, 2, 1, 3, 4).reshape(
        batch_size * num_frames, num_channel, height, width
    )
    for _ in range(pyramid_num_stages - 1):
        height //= 2
        width //= 2
        latents = F.interpolate(latents, size=(height, width), mode="bilinear") * 2
    latents = latents.reshape(
        batch_size, num_frames, num_channel, height, width
    ).permute(0, 2, 1, 3, 4)

    # Pad to patch-aligned (Conv3d stride=patch_size truncates odd dims)
    pad_h = (p_h - height % p_h) % p_h
    pad_w = (p_w - width % p_w) % p_w
    if pad_h > 0 or pad_w > 0:
        latents = F.pad(latents, (0, pad_w, 0, pad_h))
        height += pad_h
        width += pad_w

    start_point_list = [latents.clone()]

    # ── Stage loop ──
    for i_s in range(pyramid_num_stages):
        image_seq_len = (latents.shape[-1] * latents.shape[-2] * latents.shape[-3]) // (
            patch_size[0] * p_h * p_w
        )
        mu = calculate_shift(image_seq_len)
        scheduler.set_timesteps(
            pyramid_num_inference_steps_list[i_s],
            i_s,
            device=device,
            mu=mu,
        )
        timesteps = scheduler.timesteps

        if i_s > 0:
            height *= 2
            width *= 2
            num_frames = latents.shape[2]
            latents = latents.permute(0, 2, 1, 3, 4).reshape(
                batch_size * num_frames, num_channel, height // 2, width // 2
            )
            latents = F.interpolate(latents, size=(height, width), mode="nearest")
            latents = latents.reshape(
                batch_size, num_frames, num_channel, height, width
            ).permute(0, 2, 1, 3, 4)

            # Pad to patch-aligned after upsample
            pad_h = (p_h - height % p_h) % p_h
            pad_w = (p_w - width % p_w) % p_w
            if pad_h > 0 or pad_w > 0:
                latents = F.pad(latents, (0, pad_w, 0, pad_h))
                height += pad_h
                width += pad_w

            # Gamma-corrected re-noising between stages
            ori_sigma = 1 - scheduler.ori_start_sigmas[i_s]
            alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
            beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

            bs, ch, nf, h, w = latents.shape
            noise = _sample_block_noise(bs, ch, nf, h, w, patch_size, gamma, device, generator)
            noise = noise.to(device=device, dtype=latents.dtype)
            latents = alpha * latents + beta * noise

            start_point_list.append(latents.clone())

        # The adapter is trained at full resolution; applying it at lower pyramid
        # stages produces out-of-distribution signals that corrupt color and
        # structure, so enable it only at the final stage.
        is_last_stage = (i_s == pyramid_num_stages - 1)
        stage_edit_embeds = edit_instruction_embeds if is_last_stage else None

        for idx, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(torch.int64)

            noise_pred = transformer(
                hidden_states=latents.to(weight_dtype),
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short.to(weight_dtype),
                latents_history_mid=latents_history_mid.to(weight_dtype),
                latents_history_long=latents_history_long.to(weight_dtype),
                edit_instruction_embeds=stage_edit_embeds,
            )[0]

            latents = scheduler.step(
                noise_pred,
                t,
                latents,
                generator=generator,
                return_dict=False,
                cur_sampling_step=idx,
                dmd_noisy_tensor=start_point_list[i_s],
                dmd_sigmas=scheduler.sigmas,
                dmd_timesteps=scheduler.timesteps,
                all_timesteps=timesteps,
            )[0]

    # Crop back to original full resolution
    latents = latents[:, :, :, :target_h, :target_w]

    return latents
