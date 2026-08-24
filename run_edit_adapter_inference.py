"""
Inference-only script for Edit Adapter.

Loads a base Helios-Distilled transformer + a trained adapter checkpoint,
walks the EditAdapterDataset, and writes edited videos (plus source/GT as
references) to disk. No training, no optimizer, no EMA update.

Two inference paths are supported via --inference_mode:

  single_stage : 4/8/16-step single-resolution Euler DMD. This is the path used
                 for the edit chunk in the paper.
  pyramid      : multi-stage pyramid DMD.

Temporal self-attn is auto-detected by peeking at the checkpoint state_dict for
any ``temporal_self_attn.*`` key; when present the EditAdapterCollection is
constructed with ``enable_temporal_self_attn=True``.

EMA weights ``edit_adapter_ema.pth`` are loaded when --use_ema is passed
and the file exists in the ckpt dir.

Output layout:
    <output_dir>/<dataset_basename>/edited_sample_XXXX.mp4
                                  /source_sample_XXXX.mp4    (once)
                                  /gt_sample_XXXX.mp4        (once)

Launch:
    python run_edit_adapter_inference.py \
        --ckpt   /path/to/checkpoint-1000 \
        --config configs/edit_adapter_phase2.yaml \
        --data_dirs path/to/simpsons_comic,path/to/zoom_in \
        --output_dir /path/to/output \
        --inference_mode single_stage \
        --num_inference_steps 16 \
        --num_samples 20
"""

import argparse
import math
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
import imageio
import numpy as np
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file as safetensors_load_file

from helios.dataset.dataloader_edit_adapter import EditAdapterDataset
from helios.modules.transformer_helios_edit import (
    HeliosTransformer3DModelWithEditAdapter,
)
from helios.utils.train_config import Args
from helios.utils.utils_base import apply_schedule_shift, calculate_shift
from helios.utils.utils_helios_base import prepare_stage1_clean_input_from_latents
from helios.utils.utils_helios_post import add_noise, convert_flow_pred_to_x0
from helios.utils.validation_multichunk import (
    generate_multichunk_with_adapter,
    generate_multichunk_adapter_then_pyramid,
)
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (copied minimally from train_edit_adapter to avoid heavy imports)
# ──────────────────────────────────────────────────────────────────────────────

def save_video(frames, path, fps=16):
    """Save uint8 RGB frames as mp4."""
    with imageio.get_writer(
        path, fps=fps, quality=None,
        output_params=["-pix_fmt", "yuvj420p", "-crf", "18"],
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))


def latents_to_uint8_video(latents, vae, latents_mean, latents_std):
    """Decode latents -> (T, H, W, 3) uint8 RGB."""
    pixels = vae.decode((latents / latents_std + latents_mean).to(vae.dtype))[0]
    video = (pixels * 0.5 + 0.5).clamp(0, 1)
    video = (
        video[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255
    ).astype("uint8")
    return video


def _sample_block_noise(
    batch_size, channel, num_frames, height, width,
    patch_size, gamma, device, generator,
):
    """Patch-correlated noise (matches pipeline.sample_block_noise)."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Inference paths
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_single_stage_denoise(
    transformer, latents, prompt_embeds, edit_prompt_embeds,
    indices_hidden_states,
    indices_latents_history_short, indices_latents_history_mid, indices_latents_history_long,
    latents_history_short, latents_history_mid, latents_history_long,
    args, num_inference_steps, device, weight_dtype, generator,
    final_sigma_floor=None, final_sigma_extra_step=None,
):
    """Single-resolution Euler-DMD denoise.

    final_sigma_floor: optionally replace the schedule's last σ with this
        smaller value (step count unchanged) so the final pred_x0 is from
        a much cleaner input.
    final_sigma_extra_step: optionally append one extra forward at this σ
        AFTER the standard schedule (step count + 1) as a refine pass.
    """
    base_sigmas = torch.linspace(
        0.999, 0.0, steps=num_inference_steps + 1,
        dtype=torch.float32, device=device,
    )[:-1]
    if args.training_config.use_dynamic_shifting:
        base_sigmas = apply_schedule_shift(
            sigmas=base_sigmas,
            noise=latents,
            base_seq_len=args.training_config.base_seq_len,
            max_seq_len=args.training_config.max_seq_len,
            base_shift=args.training_config.base_shift,
            max_shift=args.training_config.max_shift,
        )

    if final_sigma_floor is not None:
        floor_val = float(final_sigma_floor)
        if base_sigmas[-1].item() > floor_val:
            base_sigmas = torch.cat([
                base_sigmas[:-1],
                torch.tensor([floor_val], dtype=base_sigmas.dtype, device=device),
            ])
    if final_sigma_extra_step is not None:
        base_sigmas = torch.cat([
            base_sigmas,
            torch.tensor([float(final_sigma_extra_step)],
                         dtype=base_sigmas.dtype, device=device),
        ])

    timesteps = base_sigmas * 1000.0
    sigmas_with_zero = torch.cat([base_sigmas, torch.zeros(1, device=device)])
    B = latents.shape[0]

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
            timestep=t * torch.ones(B, dtype=torch.long, device=latents.device),
            sigmas=sigmas_with_zero, timesteps=timesteps,
        )
        if step_idx < len(timesteps) - 1:
            latents = add_noise(
                pred_x0,
                torch.randn(pred_x0.shape, generator=generator,
                            device=pred_x0.device, dtype=pred_x0.dtype),
                timesteps[step_idx + 1] * torch.ones(B, dtype=torch.long, device=pred_x0.device),
                sigmas=sigmas_with_zero, timesteps=timesteps,
            )
        else:
            latents = pred_x0

    return latents


@torch.no_grad()
def run_pyramid_denoise(
    transformer, scheduler, latents,
    prompt_embeds, edit_instruction_embeds,
    indices_hidden_states,
    indices_latents_history_short, indices_latents_history_mid, indices_latents_history_long,
    latents_history_short, latents_history_mid, latents_history_long,
    pyramid_num_stages, pyramid_num_inference_steps_list,
    patch_size, device, weight_dtype, generator,
    adapter_at_all_stages: bool,
    custom_last_stage_sigmas=None,
    use_plain_renoise: bool = False,
):
    """Multi-stage pyramid DMD denoise (matches log_validation_pyramid).

    ``adapter_at_all_stages`` controls whether edit_instruction_embeds is
    passed to the lower-resolution stages. The released adapter is trained at
    full resolution only, so it wants False.

    ``custom_last_stage_sigmas``: optional list/tuple of sigma values that
    override the last stage's DMD schedule. Pass N sigma values (no trailing
    zero — it is appended internally) to perform N DMD steps.

    ``use_plain_renoise``: when True the stage-to-stage re-noise uses plain
    ``torch.randn`` instead of the patch-correlated ``_sample_block_noise``.
    Default False matches the base Helios pipeline.
    """
    batch_size, num_channel, num_frames, target_h, target_w = latents.shape
    gamma = scheduler.config.gamma
    p_h, p_w = patch_size[1], patch_size[2]

    height, width = target_h, target_w

    # Downsample to lowest resolution
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

    # Pad to patch-aligned
    pad_h = (p_h - height % p_h) % p_h
    pad_w = (p_w - width % p_w) % p_w
    if pad_h > 0 or pad_w > 0:
        latents = F.pad(latents, (0, pad_w, 0, pad_h))
        height += pad_h
        width += pad_w

    start_point_list = [latents.clone()]

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

        # Optional override of the LAST stage's sigma schedule.
        # We do this *after* set_timesteps so the scheduler's internal state
        # (start_sigmas, timesteps_per_stage, ...) is valid; we only swap
        # sigmas + timesteps used by the per-step DMD loop. timesteps are
        # rebuilt from the same dynamic-shift mapping that set_timesteps uses
        # (matches scheduling_helios_diffusers.py:253-256).
        if (
            custom_last_stage_sigmas is not None
            and i_s == pyramid_num_stages - 1
        ):
            sigmas_t = torch.tensor(
                list(custom_last_stage_sigmas) + [0.0],
                dtype=torch.float32, device=device,
            )
            t_min = scheduler.timesteps_per_stage[i_s].min()
            t_max = scheduler.timesteps_per_stage[i_s].max()
            timesteps_t = t_min + sigmas_t[:-1] * (t_max - t_min)
            scheduler.sigmas = sigmas_t
            scheduler.timesteps = timesteps_t.to(device=device)
            print(
                f"[custom_stage{i_s}] sigmas={[round(s.item(), 6) for s in sigmas_t]} "
                f"timesteps={[round(t.item(), 4) for t in timesteps_t]}"
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

            pad_h = (p_h - height % p_h) % p_h
            pad_w = (p_w - width % p_w) % p_w
            if pad_h > 0 or pad_w > 0:
                latents = F.pad(latents, (0, pad_w, 0, pad_h))
                height += pad_h
                width += pad_w

            ori_sigma = 1 - scheduler.ori_start_sigmas[i_s]
            alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
            beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

            bs, ch, nf, h, w = latents.shape
            if use_plain_renoise:
                noise = torch.randn(
                    (bs, ch, nf, h, w),
                    generator=generator, device=device, dtype=torch.float32,
                )
            else:
                noise = _sample_block_noise(bs, ch, nf, h, w, patch_size, gamma, device, generator)
            noise = noise.to(device=device, dtype=latents.dtype)
            latents = alpha * latents + beta * noise

            start_point_list.append(latents.clone())

        is_last_stage = (i_s == pyramid_num_stages - 1)
        stage_edit_embeds = (
            edit_instruction_embeds if (adapter_at_all_stages or is_last_stage)
            else None
        )

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
                noise_pred, t, latents,
                generator=generator, return_dict=False,
                cur_sampling_step=idx,
                dmd_noisy_tensor=start_point_list[i_s],
                dmd_sigmas=scheduler.sigmas,
                dmd_timesteps=scheduler.timesteps,
                all_timesteps=timesteps,
            )[0]

    latents = latents[:, :, :, :target_h, :target_w]
    return latents


# ──────────────────────────────────────────────────────────────────────────────
# Adapter checkpoint loading
# ──────────────────────────────────────────────────────────────────────────────

def _find_adapter_path(ckpt_dir, basename="edit_adapter"):
    """Find adapter file, preferring .safetensors over .pth."""
    st_path = os.path.join(ckpt_dir, f"{basename}.safetensors")
    pth_path = os.path.join(ckpt_dir, f"{basename}.pth")
    if os.path.exists(st_path):
        return st_path
    if os.path.exists(pth_path):
        return pth_path
    return None


def _load_state_dict_auto(path, map_location="cpu"):
    """Load state dict from .safetensors or .pth based on file extension."""
    if path.endswith(".safetensors"):
        return safetensors_load_file(path, device=map_location)
    return torch.load(path, map_location=map_location, weights_only=True)


def detect_adapter_features(adapter_state_dict):
    """Inspect a saved EditAdapterCollection state_dict and infer features.

    Returns a dict suitable for splatting into edit_adapter_config.
    """
    has_temporal_sa = any("temporal_self_attn." in k for k in adapter_state_dict.keys())
    has_history_ca = any("history_cross_attn." in k for k in adapter_state_dict.keys())
    return {
        "enable_temporal_self_attn": has_temporal_sa,
        "has_history_cross_attn": has_history_ca,
    }


def load_adapter_into_model(model, ckpt_dir, use_ema, logger=print):
    """Load adapter weights (or EMA) from a ckpt directory."""
    adapter_path = _find_adapter_path(ckpt_dir, "edit_adapter")
    ema_path = _find_adapter_path(ckpt_dir, "edit_adapter_ema")

    if use_ema:
        if ema_path is not None:
            logger(f"[adapter] loading EMA weights: {ema_path}")
            ema_sd = _load_state_dict_auto(ema_path)
            adapter_sd = _extract_adapter_sd_from_ema(ema_sd, model.edit_adapter)
        else:
            logger(f"[adapter] EMA file not found; falling back to live weights")
            adapter_sd = _load_state_dict_auto(adapter_path)
    else:
        logger(f"[adapter] loading live weights: {adapter_path}")
        adapter_sd = _load_state_dict_auto(adapter_path)

    missing, unexpected = model.edit_adapter.load_state_dict(adapter_sd, strict=False)
    non_temporal_missing = [k for k in missing if "temporal_self_attn" not in k]
    if non_temporal_missing:
        raise RuntimeError(
            f"Unexpected missing keys when loading adapter: {non_temporal_missing}"
        )
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading adapter: {unexpected}")

    return adapter_sd


def _extract_adapter_sd_from_ema(ema_sd, adapter_module):
    """EMAAdapter wraps the model weights; produce a flat state_dict to load.

    EMAAdapter.state_dict() format (helios/utils/ema_adapter.py:97-107):
        {
            "decay": ..., "min_decay": ..., "optimization_step": ...,
            "shadow_model": <flat state_dict of the adapter>,
            ...
        }

    Falls back to treating ema_sd itself as the adapter state_dict for
    backwards compatibility with any older save format.
    """
    if isinstance(ema_sd, dict) and "shadow_model" in ema_sd:
        return ema_sd["shadow_model"]
    return ema_sd


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args, cli):
    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=3600))
    accelerator = Accelerator(
        mixed_precision=args.training_config.mixed_precision,
        kwargs_handlers=[init_kwargs],
    )
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ─── OSS cache: download base model to local SSD on cluster ───
    # Mirrors train_edit_adapter.py:1481-1495. Only the base model
    # (VAE + transformer pretrained weights) is fetched from OSS; the
    # adapter checkpoint at cli.ckpt is loaded directly from its given
    # NAS path and NOT routed through OSS, since adapter ckpts live in the
    # training output dirs (not in the OSS-mirrored model store).
    # ─── Inspect ckpt to decide adapter structure ───
    adapter_path = _find_adapter_path(cli.ckpt, "edit_adapter")
    if adapter_path is None:
        raise FileNotFoundError(
            f"adapter checkpoint not found in {cli.ckpt} "
            f"(looked for edit_adapter.safetensors / .pth)")
    if is_main:
        print(f"[ckpt] peeking adapter weights at {adapter_path}")
    peek_sd = _load_state_dict_auto(adapter_path)
    detected = detect_adapter_features(peek_sd)
    if is_main:
        print(f"[ckpt] detected adapter features: {detected}")
    del peek_sd

    # ─── Load VAE & text encoder skipping: ───
    # We rely on the dataset to provide pre-extracted text embeds, so no
    # tokenizer / text encoder is needed. Only VAE is needed to decode latents
    # back into pixels.
    if is_main:
        print("[load] VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        args.model_config.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(
        device, weight_dtype
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(
        device, weight_dtype
    )

    # ─── Load transformer with adapter ───
    if is_main:
        print("[load] transformer + adapter shell...")
    edit_adapter_config = {
        "adapter_dim": args.training_config.edit_adapter_dim,
        "num_heads": args.training_config.edit_adapter_num_heads,
        "text_dim": args.training_config.edit_adapter_text_dim,
        "eps": args.training_config.edit_adapter_eps,
        "enable_temporal_self_attn": detected["enable_temporal_self_attn"],
        "temporal_max_len": getattr(
            args.training_config, "edit_adapter_temporal_max_len", 64
        ),
    }

    transformer_additional_kwargs = {
        "has_multi_term_memory_patch": args.training_config.has_multi_term_memory_patch,
        "zero_history_timestep": args.training_config.zero_history_timestep,
        "restrict_self_attn": args.training_config.restrict_self_attn,
        "guidance_cross_attn": args.training_config.guidance_cross_attn,
    }

    transformer = HeliosTransformer3DModelWithEditAdapter.from_pretrained_with_edit_adapter(
        args.model_config.transformer_model_name_or_path,
        subfolder="transformer",
        edit_adapter_config=edit_adapter_config,
        transformer_additional_kwargs=transformer_additional_kwargs,
    )
    transformer.requires_grad_(False)

    # Cast to weight_dtype before loading adapter, then load adapter in fp32 then re-cast.
    for p in transformer.parameters():
        p.data = p.data.to(weight_dtype)
    transformer.to(device)
    transformer.eval()

    # ─── Load adapter (or EMA) ───
    load_adapter_into_model(transformer, cli.ckpt, use_ema=cli.use_ema)
    # Match dtype after load
    for p in transformer.edit_adapter.parameters():
        p.data = p.data.to(weight_dtype)

    # ─── Kernel optimizations (best-effort) ───
    try:
        from helios.modules.helios_kernels import (
            replace_all_norms_with_flash_norms,
            replace_rmsnorm_with_fp32,
            replace_rope_with_flash_rope,
        )
        transformer = replace_rmsnorm_with_fp32(transformer)
        transformer = replace_all_norms_with_flash_norms(transformer)
        replace_rope_with_flash_rope()
    except Exception as e:
        print(f"[warn] kernel optimizations not applied: {e}")

    # ─── Dataset ───
    if is_main:
        print("[data] building dataset...")
    train_dataset = EditAdapterDataset(
        feature_folder=args.data_config.instance_data_root,
        history_sizes=args.training_config.history_sizes,
        latent_window_size=args.training_config.latent_window_size[0],
        seed=args.seed,
        data_mode=args.training_config.edit_adapter_data_mode,
    )
    if is_main:
        print(f"[data] dataset size = {len(train_dataset)}")

    # Group sample indices by dataset_name so output is organized by style
    from collections import defaultdict
    by_dataset = defaultdict(list)
    for i, s in enumerate(train_dataset.samples):
        by_dataset[s["dataset_name"]].append(i)

    # Subsampling
    if cli.num_samples_per_style is not None:
        sampled = []
        for name, indices in by_dataset.items():
            sampled.extend(indices[:cli.num_samples_per_style])
        sample_indices = sampled
    elif cli.num_samples is not None:
        per_dataset_n = max(1, cli.num_samples // max(1, len(by_dataset)))
        sampled = []
        for name, indices in by_dataset.items():
            sampled.extend(indices[:per_dataset_n])
        sample_indices = sampled[:cli.num_samples] if cli.num_samples > 0 else sampled
    else:
        sample_indices = sum(by_dataset.values(), [])

    if is_main:
        print(f"[data] selected {len(sample_indices)} samples across {len(by_dataset)} datasets")

    # Resolve inference mode
    inference_mode = cli.inference_mode
    if inference_mode == "pyramid":
        pyramid_scheduler_kwargs = {}
        stage_range = cli.pyramid_stage_range or getattr(
            args.validation_config, "pyramid_stage_range", None
        )
        if stage_range is not None:
            pyramid_scheduler_kwargs["stage_range"] = list(stage_range)
        pyramid_scheduler = HeliosScheduler.from_pretrained(
            args.model_config.pretrained_model_name_or_path,
            subfolder="scheduler",
            **pyramid_scheduler_kwargs,
        )
        patch_size = transformer.config.patch_size
        pyramid_steps_list = list(cli.pyramid_steps) if cli.pyramid_steps else list(
            args.validation_config.stage2_simulated_inference_steps
        )
        pyramid_num_stages = len(pyramid_steps_list)
        if is_main:
            print(
                f"[mode] pyramid DMD, stages={pyramid_num_stages}, "
                f"steps_per_stage={pyramid_steps_list}, "
                f"adapter_at_all_stages={cli.adapter_at_all_stages}"
            )
    else:
        if is_main:
            print(
                f"[mode] single-stage Euler, num_inference_steps={cli.num_inference_steps}"
            )

    latent_window_size = args.training_config.latent_window_size[0]
    history_sizes = args.training_config.history_sizes

    out_root = Path(cli.output_dir)
    if is_main:
        out_root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if not is_main:
        out_root.mkdir(parents=True, exist_ok=True)
    if is_main:
        print(f"[out] writing under {out_root}, world_size={world_size}")

    # Multichunk setup (if enabled)
    mc_noise_scheduler = None
    mc_pyramid_scheduler = None
    mc_pyramid_steps = None
    mc_pyramid_stages = None
    mc_run_adapter = False
    mc_run_pyramid = False
    if cli.enable_multichunk:
        mc_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.model_config.pretrained_model_name_or_path, subfolder="scheduler"
        )
        mc_run_adapter = cli.multichunk_mode in ("adapter_only", "both")
        mc_run_pyramid = cli.multichunk_mode in ("adapter_then_pyramid", "both")
        if mc_run_pyramid:
            from functools import partial
            mc_pyramid_scheduler = HeliosScheduler.from_pretrained(
                args.model_config.pretrained_model_name_or_path, subfolder="scheduler"
            )
            mc_pyramid_steps = list(args.validation_config.stage2_simulated_inference_steps)
            mc_pyramid_stages = len(mc_pyramid_steps)
            mc_pyramid_fn = partial(
                run_pyramid_denoise,
                adapter_at_all_stages=False,
                custom_last_stage_sigmas=None,
                use_plain_renoise=False,
            )
        if is_main:
            print(f"[multichunk] mode={cli.multichunk_mode}, chunks={cli.num_chunks}")

    # Build flat task list for round-robin distribution across ranks
    all_tasks = []
    for dataset_name, indices in by_dataset.items():
        indices_to_run = [i for i in indices if i in set(sample_indices)]
        for local_idx, idx in enumerate(indices_to_run):
            all_tasks.append((dataset_name, local_idx, idx))
    if is_main:
        print(f"[run] total tasks={len(all_tasks)}, distributing across {world_size} GPUs")

    for global_i, (dataset_name, local_idx, idx) in enumerate(all_tasks):
        if global_i % world_size != rank:
            continue
        style = Path(dataset_name).name
        out_dir = out_root / style
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{local_idx:04d}_idx{idx}"
        sample = train_dataset[idx]

        history_latents = sample["history_latents"].unsqueeze(0).to(device, dtype=weight_dtype)
        target_latents = sample["target_latents"].unsqueeze(0).to(device, dtype=weight_dtype)
        x0_latents = sample["x0_latents"].unsqueeze(0).to(device, dtype=weight_dtype)
        prompt_embeds = sample["prompt_embeds"].unsqueeze(0).to(device, dtype=weight_dtype)
        edit_prompt_embeds = sample["edit_prompt_embeds"].unsqueeze(0).to(device, dtype=weight_dtype)

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
            history_latents=history_latents,
            target_latents=target_latents,
            x0_latents=x0_latents,
            latent_window_size=latent_window_size,
            history_sizes=history_sizes,
            is_random_drop=False,
            is_keep_x0=True,
            dtype=weight_dtype,
            device=device,
        )

        # ── source video (decode once if missing) ──
        src_path = out_dir / f"source_{tag}.mp4"
        if not src_path.exists():
            if args.training_config.edit_adapter_data_mode == "separated":
                src_latents = history_latents
            else:
                src_latents = torch.cat(
                    [x0_latents, history_latents[:, :, :10, :, :]], dim=2
                )
            src_video = latents_to_uint8_video(src_latents, vae, latents_mean, latents_std)
            save_video(src_video, str(src_path), fps=16)

        # ── GT (decode once if missing) ──
        gt_path = out_dir / f"gt_{tag}.mp4"
        if not gt_path.exists():
            gt_video = latents_to_uint8_video(target_latents, vae, latents_mean, latents_std)
            save_video(gt_video, str(gt_path), fps=16)

        # ── edited video ──
        gen = torch.Generator(device=device).manual_seed(42 + idx)
        init_latents = torch.randn(
            target_latents.shape, generator=gen,
            device=device,
            dtype=(torch.float32 if inference_mode == "pyramid" else target_latents.dtype),
        )

        if inference_mode == "pyramid":
            ed_latents = run_pyramid_denoise(
                transformer=transformer,
                scheduler=pyramid_scheduler,
                latents=init_latents,
                prompt_embeds=prompt_embeds,
                edit_instruction_embeds=edit_prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                pyramid_num_stages=pyramid_num_stages,
                pyramid_num_inference_steps_list=pyramid_steps_list,
                patch_size=patch_size,
                device=device,
                weight_dtype=weight_dtype,
                generator=gen,
                adapter_at_all_stages=cli.adapter_at_all_stages,
                custom_last_stage_sigmas=cli.custom_stage2_sigmas,
                use_plain_renoise=cli.use_plain_renoise,
            )
        else:
            ed_latents = run_single_stage_denoise(
                transformer=transformer,
                latents=init_latents,
                prompt_embeds=prompt_embeds,
                edit_prompt_embeds=edit_prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                args=args,
                num_inference_steps=cli.num_inference_steps,
                device=device,
                weight_dtype=weight_dtype,
                generator=gen,
                final_sigma_floor=cli.final_sigma_floor,
                final_sigma_extra_step=cli.final_sigma_extra_step,
            )

        edited_video = latents_to_uint8_video(ed_latents, vae, latents_mean, latents_std)
        ed_path = out_dir / f"edited_{tag}.mp4"
        save_video(edited_video, str(ed_path), fps=16)
        print(f"  [rank {rank}][{style}] {global_i+1}/{len(all_tasks)} -> {ed_path.name}")

        # ── multichunk generation ──
        if cli.enable_multichunk and mc_run_adapter:
            mc_video = generate_multichunk_with_adapter(
                transformer=transformer, vae=vae,
                noise_scheduler=mc_noise_scheduler,
                history_latents=history_latents, x0_latents=x0_latents,
                prompt_embeds=prompt_embeds,
                edit_prompt_embeds=edit_prompt_embeds,
                latents_mean=latents_mean, latents_std=latents_std,
                args=args, device=device, weight_dtype=weight_dtype,
                num_chunks=cli.num_chunks,
                num_inference_steps=cli.num_inference_steps,
                x0_mode="fixed", seed=42, sample_idx=idx,
            )
            mc_path = out_dir / f"multichunk_adapter_{tag}.mp4"
            save_video(mc_video, str(mc_path), fps=16)
            print(f"  [rank {rank}][{style}] multichunk adapter -> {mc_path.name}")
            del mc_video
            torch.cuda.empty_cache()

        if cli.enable_multichunk and mc_run_pyramid:
            mc_video = generate_multichunk_adapter_then_pyramid(
                transformer=transformer, vae=vae,
                noise_scheduler=mc_noise_scheduler,
                history_latents=history_latents, x0_latents=x0_latents,
                prompt_embeds=prompt_embeds,
                edit_prompt_embeds=edit_prompt_embeds,
                latents_mean=latents_mean, latents_std=latents_std,
                args=args, device=device, weight_dtype=weight_dtype,
                pyramid_denoise_fn=mc_pyramid_fn,
                pyramid_scheduler=mc_pyramid_scheduler,
                pyramid_num_stages=mc_pyramid_stages,
                pyramid_num_inference_steps_list=mc_pyramid_steps,
                patch_size=transformer.config.patch_size,
                num_chunks=cli.num_chunks,
                num_inference_steps=cli.num_inference_steps,
                seed=42, sample_idx=idx,
            )
            mc_path = out_dir / f"multichunk_ap_{tag}.mp4"
            save_video(mc_video, str(mc_path), fps=16)
            print(f"  [rank {rank}][{style}] multichunk ap -> {mc_path.name}")
            del mc_video

        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if is_main:
        print("[done] inference complete.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Checkpoint directory containing edit_adapter.pth (and optionally edit_adapter_ema.pth).")
    parser.add_argument("--config", type=str, required=True,
                        help="Training yaml whose model_config / training_config / validation_config are reused.")
    parser.add_argument("--data_dirs", type=str, default=None,
                        help="Comma-separated style/instruction directories; overrides data_config.instance_data_root.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write per-style subdirectories of mp4 outputs.")
    parser.add_argument("--inference_mode", type=str, default="single_stage",
                        choices=["single_stage", "pyramid"],
                        help="single_stage: Euler DMD; pyramid: 3-stage HeliosScheduler DMD.")
    # single_stage knobs
    parser.add_argument("--num_inference_steps", type=int, default=16,
                        help="Steps for single_stage Euler DMD.")
    # pyramid knobs
    parser.add_argument("--pyramid_steps", type=int, nargs="+", default=None,
                        help="Per-stage step counts for pyramid mode (e.g. 2 2 8). Defaults to yaml's stage2_simulated_inference_steps.")
    parser.add_argument("--pyramid_stage_range", type=float, nargs="+", default=None,
                        help="Optional override of HeliosScheduler stage_range.")
    parser.add_argument("--adapter_at_all_stages", action="store_true",
                        help="pyramid only: enable the adapter at every stage. Off (default) → adapter only on the last, full-resolution stage.")
    parser.add_argument("--custom_stage2_sigmas", type=float, nargs="+", default=None,
                        help="pyramid only: override the last (full-res) stage's sigma schedule. Pass N sigmas "
                             "(no trailing 0), e.g. --custom_stage2_sigmas 0.9997 0.5 0.003532 for a 3-step schedule.")
    parser.add_argument("--use_plain_renoise", action="store_true", default=False,
                        help="pyramid only: replace the patch-correlated _sample_block_noise used at "
                             "stage transitions with plain torch.randn.")
    # Terminal-σ knobs for single_stage mode (mirrors train_edit_adapter.py).
    parser.add_argument("--final_sigma_floor", type=float, default=None,
                        help="single_stage only: replace the schedule's last σ with this smaller value "
                             "(no extra step). Example: 0.01 → end the schedule near σ=0 instead of σ≈0.19.")
    parser.add_argument("--final_sigma_extra_step", type=float, default=None,
                        help="single_stage only: append one extra forward at this σ AFTER the standard "
                             "schedule (step count + 1). Acts as a final refine pass.")
    # sampling
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Max samples to run (split evenly across styles). None = the entire dataset.")
    parser.add_argument("--num_samples_per_style", type=int, default=None,
                        help="Number of samples per style. Overrides --num_samples when set.")
    # Multichunk
    parser.add_argument("--enable_multichunk", action="store_true",
                        help="Also generate multichunk videos for each sample.")
    parser.add_argument("--multichunk_mode", type=str, default="both",
                        choices=["adapter_only", "adapter_then_pyramid", "both"],
                        help="adapter_only: all chunks use adapter + Euler. "
                             "adapter_then_pyramid: chunk 0 uses adapter, rest use pyramid DMD. "
                             "both: run both modes for each sample.")
    parser.add_argument("--num_chunks", type=int, default=3,
                        help="Number of chunks for multichunk generation.")
    parser.add_argument("--use_ema", action="store_true",
                        help="Prefer edit_adapter_ema.pth if present.")
    return parser.parse_args()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cli = parse_cli()
    config = OmegaConf.load(cli.config)
    schema = OmegaConf.structured(Args)
    conf = OmegaConf.merge(schema, config)

    if cli.data_dirs is not None:
        conf.data_config.instance_data_root = cli.data_dirs.split(",")

    main(conf, cli)
