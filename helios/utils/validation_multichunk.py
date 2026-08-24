"""
Multi-chunk video generation with Edit Adapter for validation.

Generates videos by iteratively producing latent chunks and sliding the
history window, supporting two x0 anchor strategies:

  "fixed"   — x0 stays as the original source anchor (src first frame)
  "sliding" — x0 = first frame of the current 19-frame history window

Also provides a hybrid strategy where chunk 0 uses adapter (Euler) and
subsequent chunks use pyramid DMD without adapter, with the edited anchor
(first frame of chunk 0 output) as x0.
"""

import logging

import numpy as np
import torch

from helios.utils.utils_base import apply_schedule_shift
from helios.utils.utils_helios_base import prepare_stage1_clean_input_from_latents
from helios.utils.utils_helios_post import add_noise, convert_flow_pred_to_x0

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_multichunk_with_adapter(
    transformer,
    vae,
    noise_scheduler,
    history_latents,        # (1, C, 19, H, W) — initial src history
    x0_latents,             # (1, C, 1, H, W) — initial x0 anchor
    prompt_embeds,          # (1, 512, 4096) — scene caption
    edit_prompt_embeds,     # (1, 512, 4096) — edit instruction
    latents_mean,
    latents_std,
    args,
    device,
    weight_dtype,
    num_chunks=3,
    num_inference_steps=8,
    x0_mode="fixed",
    seed=42,
    sample_idx=0,
):
    """Generate multi-chunk video with adapter.

    Args:
        x0_mode: "fixed" keeps x0 as the original anchor across all chunks.
                 "sliding" uses the first frame of the current history window.

    Returns:
        uint8 numpy array of shape (T, H, W, C).
    """
    assert x0_mode in ("fixed", "sliding"), f"Unknown x0_mode={x0_mode!r}"

    latent_window_size = args.training_config.latent_window_size[0]  # 9
    history_sizes = sorted(args.training_config.history_sizes, reverse=True)  # [16, 2, 1]
    history_window_size = sum(history_sizes)  # 19

    B = history_latents.shape[0]  # 1
    C = history_latents.shape[1]
    H = history_latents.shape[3]
    W = history_latents.shape[4]

    current_history = history_latents.clone()  # (1, C, 19, H, W)
    current_x0 = x0_latents.clone()            # (1, C, 1, H, W)

    all_decoded_chunks = []

    for chunk_idx in range(num_chunks):
        logger.info(f"      Multi-chunk [{x0_mode}] chunk {chunk_idx+1}/{num_chunks}")

        # Update x0 for sliding mode
        if x0_mode == "sliding":
            current_x0 = current_history[:, :, :1, :, :]  # first frame of current window

        # Dummy target for prepare_stage1 (needed for shape/index calculation)
        dummy_target = torch.zeros(
            B, C, latent_window_size, H, W,
            device=device, dtype=weight_dtype,
        )

        # Split history into long/mid/short + x0
        (
            _model_input,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
        ) = prepare_stage1_clean_input_from_latents(
            history_latents=current_history,
            target_latents=dummy_target,
            x0_latents=current_x0,
            latent_window_size=latent_window_size,
            history_sizes=history_sizes,
            is_random_drop=False,
            is_keep_x0=True,
            dtype=weight_dtype,
            device=device,
        )

        # Random noise for this chunk (deterministic per chunk)
        chunk_seed = seed + sample_idx * num_chunks + chunk_idx
        noise_gen = torch.Generator(device=device).manual_seed(chunk_seed)
        latents = torch.randn(
            B, C, latent_window_size, H, W,
            generator=noise_gen, device=device, dtype=weight_dtype,
        )

        # Build the sigma schedule manually, mirroring _build_denoise_schedule
        # in train_edit_adapter.py (no set_timesteps call).
        sigmas = torch.linspace(
            0.999, 0.0, steps=num_inference_steps + 1,
            dtype=torch.float32, device=device,
        )[:-1]
        if args.training_config.use_dynamic_shifting:
            sigmas = apply_schedule_shift(
                sigmas=sigmas,
                noise=latents,
                base_seq_len=args.training_config.base_seq_len,
                max_seq_len=args.training_config.max_seq_len,
                base_shift=args.training_config.base_shift,
                max_shift=args.training_config.max_shift,
            )
        timesteps = sigmas * 1000.0
        dmd_sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
        dmd_timesteps = timesteps

        # Denoising loop
        for step_idx, t in enumerate(timesteps):
            noise_pred = transformer(
                hidden_states=latents,
                timestep=t.expand(B),
                encoder_hidden_states=prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                is_first_denoising_step=(step_idx == 0),
                edit_instruction_embeds=edit_prompt_embeds,
                return_dict=False,
            )[0]

            pred_x0 = convert_flow_pred_to_x0(
                flow_pred=noise_pred, xt=latents,
                timestep=t * torch.ones(B, dtype=torch.long, device=device),
                sigmas=dmd_sigmas, timesteps=dmd_timesteps,
            )
            if step_idx < len(timesteps) - 1:
                latents = add_noise(
                    pred_x0,
                    torch.randn(
                        pred_x0.shape, generator=noise_gen,
                        device=device, dtype=weight_dtype,
                    ),
                    timesteps[step_idx + 1] * torch.ones(
                        B, dtype=torch.long, device=device,
                    ),
                    sigmas=dmd_sigmas, timesteps=dmd_timesteps,
                )
            else:
                latents = pred_x0

        # Decode this chunk independently (matching pipeline behaviour:
        # each chunk is decoded separately to avoid 3D-VAE temporal-conv
        # artefacts at latent-space chunk boundaries).
        chunk_normed = (latents / latents_std + latents_mean).to(vae.dtype)
        chunk_pixels = vae.decode(chunk_normed)[0]  # (1, C, T_chunk, H, W)
        chunk_video = (chunk_pixels * 0.5 + 0.5).clamp(0, 1)
        all_decoded_chunks.append(chunk_video)

        # Slide history window: append generated, keep last 19 frames
        current_history = torch.cat(
            [current_history, latents], dim=2,
        )[:, :, -history_window_size:, :, :]

    # Concatenate in pixel space and convert to uint8
    full_video = torch.cat(all_decoded_chunks, dim=2)  # (1, C, T_total, H, W)
    video = (
        full_video[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255
    ).astype("uint8")  # (T, H, W, C)
    return video


@torch.no_grad()
def generate_multichunk_adapter_then_pyramid(
    transformer,
    vae,
    noise_scheduler,
    history_latents,        # (1, C, 19, H, W) — initial src history
    x0_latents,             # (1, C, 1, H, W) — initial x0 anchor
    prompt_embeds,          # (1, 512, 4096) — scene caption
    edit_prompt_embeds,     # (1, 512, 4096) — edit instruction
    latents_mean,
    latents_std,
    args,
    device,
    weight_dtype,
    # Pyramid DMD params (for chunk 1+)
    pyramid_denoise_fn,     # callable: _run_stage2_denoise from train_edit_adapter
    pyramid_scheduler,      # HeliosScheduler instance
    pyramid_num_stages,
    pyramid_num_inference_steps_list,
    patch_size,
    # General params
    num_chunks=4,
    num_inference_steps=8,
    seed=42,
    sample_idx=0,
):
    """Generate multi-chunk video: chunk 0 with adapter, chunk 1+ with pyramid DMD.

    Chunk 0 uses Euler stepping with adapter enabled to "ignite" the edit.
    Subsequent chunks use pyramid DMD (no adapter) with edit_prompt_embeds as
    text conditioning, relying on history context to propagate the edit style.

    x0 strategy: "edited anchor" — after chunk 0, x0 is fixed to the first
    frame of chunk 0's output (the edited anchor).

    Returns:
        uint8 numpy array of shape (T, H, W, C).
    """
    latent_window_size = args.training_config.latent_window_size[0]  # 9
    history_sizes = sorted(args.training_config.history_sizes, reverse=True)  # [16, 2, 1]
    history_window_size = sum(history_sizes)  # 19

    B = history_latents.shape[0]  # 1
    C = history_latents.shape[1]
    H = history_latents.shape[3]
    W = history_latents.shape[4]

    current_history = history_latents.clone()  # (1, C, 19, H, W)
    current_x0 = x0_latents.clone()            # (1, C, 1, H, W)

    all_decoded_chunks = []

    for chunk_idx in range(num_chunks):
        is_first_chunk = (chunk_idx == 0)
        mode_str = "adapter+Euler" if is_first_chunk else "pyramid-DMD"
        logger.info(
            f"      Multi-chunk [adapter→pyramid] chunk {chunk_idx+1}/{num_chunks} ({mode_str})"
        )

        # Dummy target for prepare_stage1 (needed for shape/index calculation)
        dummy_target = torch.zeros(
            B, C, latent_window_size, H, W,
            device=device, dtype=weight_dtype,
        )

        # Split history into long/mid/short + x0
        (
            _model_input,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
        ) = prepare_stage1_clean_input_from_latents(
            history_latents=current_history,
            target_latents=dummy_target,
            x0_latents=current_x0,
            latent_window_size=latent_window_size,
            history_sizes=history_sizes,
            is_random_drop=False,
            is_keep_x0=True,
            dtype=weight_dtype,
            device=device,
        )

        # Random noise for this chunk (deterministic per chunk)
        chunk_seed = seed + sample_idx * num_chunks + chunk_idx
        noise_gen = torch.Generator(device=device).manual_seed(chunk_seed)
        latents = torch.randn(
            B, C, latent_window_size, H, W,
            generator=noise_gen, device=device, dtype=weight_dtype,
        )

        if is_first_chunk:
            # ── Chunk 0: Euler stepping with adapter ──
            # Build sigma schedule manually (same as generate_multichunk_with_adapter)
            sigmas = torch.linspace(
                0.999, 0.0, steps=num_inference_steps + 1,
                dtype=torch.float32, device=device,
            )[:-1]
            if args.training_config.use_dynamic_shifting:
                sigmas = apply_schedule_shift(
                    sigmas=sigmas,
                    noise=latents,
                    base_seq_len=args.training_config.base_seq_len,
                    max_seq_len=args.training_config.max_seq_len,
                    base_shift=args.training_config.base_shift,
                    max_shift=args.training_config.max_shift,
                )
            timesteps = sigmas * 1000.0
            dmd_sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
            dmd_timesteps = timesteps

            for step_idx, t in enumerate(timesteps):
                noise_pred = transformer(
                    hidden_states=latents,
                    timestep=t.expand(B),
                    encoder_hidden_states=prompt_embeds,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=latents_history_short,
                    latents_history_mid=latents_history_mid,
                    latents_history_long=latents_history_long,
                    is_first_denoising_step=(step_idx == 0),
                    edit_instruction_embeds=edit_prompt_embeds,
                    return_dict=False,
                )[0]

                pred_x0 = convert_flow_pred_to_x0(
                    flow_pred=noise_pred, xt=latents,
                    timestep=t * torch.ones(B, dtype=torch.long, device=device),
                    sigmas=dmd_sigmas, timesteps=dmd_timesteps,
                )
                if step_idx < len(timesteps) - 1:
                    latents = add_noise(
                        pred_x0,
                        torch.randn(
                            pred_x0.shape, generator=noise_gen,
                            device=device, dtype=weight_dtype,
                        ),
                        timesteps[step_idx + 1] * torch.ones(
                            B, dtype=torch.long, device=device,
                        ),
                        sigmas=dmd_sigmas, timesteps=dmd_timesteps,
                    )
                else:
                    latents = pred_x0

            # Set edited anchor: first frame of chunk 0 output
            current_x0 = latents[:, :, :1, :, :].clone()

        else:
            # ── Chunk 1+: Pyramid DMD, no adapter, edit instruction as text ──
            latents = pyramid_denoise_fn(
                transformer=transformer,
                scheduler=pyramid_scheduler,
                latents=latents,
                prompt_embeds=edit_prompt_embeds,       # edit instruction as text
                edit_instruction_embeds=None,            # no adapter
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                pyramid_num_stages=pyramid_num_stages,
                pyramid_num_inference_steps_list=pyramid_num_inference_steps_list,
                patch_size=patch_size,
                device=device,
                weight_dtype=weight_dtype,
                generator=noise_gen,
            )

        # Decode this chunk independently (same as pipeline: per-chunk decode
        # avoids 3D-VAE temporal-conv artefacts at chunk boundaries).
        chunk_normed = (latents / latents_std + latents_mean).to(vae.dtype)
        chunk_pixels = vae.decode(chunk_normed)[0]  # (1, C, T_chunk, H, W)
        chunk_video = (chunk_pixels * 0.5 + 0.5).clamp(0, 1)
        all_decoded_chunks.append(chunk_video)

        # Slide history window: append generated, keep last 19 frames
        current_history = torch.cat(
            [current_history, latents], dim=2,
        )[:, :, -history_window_size:, :, :]

    # Concatenate in pixel space and convert to uint8
    full_video = torch.cat(all_decoded_chunks, dim=2)  # (1, C, T_total, H, W)
    video = (
        full_video[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255
    ).astype("uint8")  # (T, H, W, C)
    return video
