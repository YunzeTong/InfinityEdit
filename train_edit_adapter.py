"""
Training script for the Edit Adapter on Helios-Distilled.

The base transformer stays frozen; only the Edit Adapter (edit cross-attention
plus optional temporal self-attention) is trained, with a flow-matching loss on
a 9-latent-frame target window conditioned on a 19-latent-frame history.

Validation runs single-stage Euler-DMD denoising with two optional knobs that
control where the sigma schedule terminates:

    --final_sigma_floor       replace the schedule's last sigma with a smaller
                              value, keeping the step count unchanged.
    --final_sigma_extra_step  append one extra forward pass at a small sigma
                              after the schedule, as a terminal refine step.

Both default to off. See ``scripts/train_phase1.sh`` and
``scripts/train_phase2.sh`` for the two-stage recipe used in the paper.

Launch:
    accelerate launch --num_processes 8 train_edit_adapter.py \
        --config configs/edit_adapter_phase1.yaml \
        --enable_temporal_self_attn \
        --final_sigma_extra_step 0.01
"""

import argparse
import logging
import math
import os
import shutil
from datetime import timedelta

import torch
import torch.distributed as dist
import transformers
import diffusers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from tqdm.auto import tqdm
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_loss_weighting_for_sd3
import imageio
import numpy as np

from helios.dataset.dataloader_edit_adapter import (
    EditAdapterDataset,
    EditAdapterSampler,
    edit_adapter_collate_fn,
)
from helios.modules.transformer_helios_edit import (
    HeliosTransformer3DModelWithEditAdapter,
)
from helios.utils.train_config import Args
from helios.utils.utils_base import apply_schedule_shift, get_optimizer
from helios.utils.utils_edit_adapter import (
    flow_loss_edit_adapter,
    prepare_noise_input_edit_adapter,
)
from helios.utils.utils_helios_base import (
    prepare_stage1_clean_input_from_latents,
    corrupt_history_latents,
)
from helios.utils.utils_helios_post import add_noise, convert_flow_pred_to_x0
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
from safetensors.torch import save_file as safetensors_save_file
from safetensors.torch import load_file as safetensors_load_file
from helios.utils.ema_adapter import EMAAdapter
from helios.utils.validation_multichunk import (
    generate_multichunk_with_adapter,
    generate_multichunk_adapter_then_pyramid,
)
from helios.evaluation.psnr_ssim_metrics import compute_psnr, compute_ssim

from helios.utils.train_helpers import (
    unwrap_model,
    save_video,
    _select_validation_indices,
    _run_stage2_denoise,
)


logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Single-stage Euler-DMD denoise
# ──────────────────────────────────────────────────────────────────────────────

def _build_denoise_schedule(
    num_inference_steps,
    args,
    init_latents,
    device,
    final_sigma_floor=None,
    final_sigma_extra_step=None,
):
    """Build (sigmas, timesteps) for single-stage DMD denoising.

    Returns:
        sigmas: tensor (num_steps + 1,) — index i is the σ before step i;
                index num_steps is always 0 (terminal).
        timesteps: tensor (num_steps,) — model conditioning per step.
        num_steps: int — number of forward passes the loop will perform.

    final_sigma_floor:
      Replace the last entry of the schedule (the smallest σ before the
      terminal 0) with this value; the step count is unchanged. A linear
      16-step schedule ends at σ≈0.19, so floor=0.01 makes the model run its
      final pred_x0 from a much cleaner input.

    final_sigma_extra_step:
      Append one extra forward at this σ after the schedule, growing the step
      count by 1. The extra step re-noises the previous pred_x0 to the given σ
      and runs one more forward as a refine pass.

    Both knobs may be combined.
    """
    base_sigmas = torch.linspace(
        0.999, 0.0, steps=num_inference_steps + 1,
        dtype=torch.float32, device=device,
    )[:-1]
    if args.training_config.use_dynamic_shifting:
        base_sigmas = apply_schedule_shift(
            sigmas=base_sigmas,
            noise=init_latents,
            base_seq_len=args.training_config.base_seq_len,
            max_seq_len=args.training_config.max_seq_len,
            base_shift=args.training_config.base_shift,
            max_shift=args.training_config.max_shift,
        )

    # Floor the last σ before the terminal 0.
    if final_sigma_floor is not None:
        floor_val = float(final_sigma_floor)
        if base_sigmas[-1].item() > floor_val:
            base_sigmas = torch.cat([
                base_sigmas[:-1],
                torch.tensor([floor_val], dtype=base_sigmas.dtype, device=device),
            ])

    # Append an extra refine step at a small σ.
    if final_sigma_extra_step is not None:
        extra_val = float(final_sigma_extra_step)
        base_sigmas = torch.cat([
            base_sigmas,
            torch.tensor([extra_val], dtype=base_sigmas.dtype, device=device),
        ])

    timesteps = base_sigmas * 1000.0
    sigmas_with_zero = torch.cat([base_sigmas, torch.zeros(1, device=device)])
    return sigmas_with_zero, timesteps, len(timesteps)


def _denoise_one_sample(
    transformer, init_latents,
    prompt_embeds, edit_prompt_embeds,
    indices_hidden_states,
    indices_latents_history_short, indices_latents_history_mid, indices_latents_history_long,
    latents_history_short, latents_history_mid, latents_history_long,
    args, num_inference_steps, device, weight_dtype, generator,
    final_sigma_floor=None, final_sigma_extra_step=None,
):
    """Single-stage Euler-DMD denoise with optional terminal-σ control.

    Returns the final denoised latents.
    """
    sigmas, timesteps, num_steps = _build_denoise_schedule(
        num_inference_steps=num_inference_steps,
        args=args,
        init_latents=init_latents,
        device=device,
        final_sigma_floor=final_sigma_floor,
        final_sigma_extra_step=final_sigma_extra_step,
    )

    latents = init_latents
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
            sigmas=sigmas, timesteps=timesteps,
        )
        if step_idx < num_steps - 1:
            latents = add_noise(
                pred_x0,
                torch.randn(pred_x0.shape, generator=generator,
                            device=pred_x0.device, dtype=pred_x0.dtype),
                timesteps[step_idx + 1] * torch.ones(B, dtype=torch.long, device=pred_x0.device),
                sigmas=sigmas, timesteps=timesteps,
            )
        else:
            latents = pred_x0

    return latents


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def log_validation_edit_adapter(
    transformer, vae, text_encoder, tokenizer, noise_scheduler,
    train_dataset, args, accelerator, global_step, weight_dtype,
    latents_mean, latents_std,
):
    """Generate validation videos with the single-stage path (distributed).

    Work is sharded across all ranks via round-robin index assignment.
    Metrics are gathered / reduced so tensorboard sees the full picture.
    """
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    logger.info(
        f"Starting distributed validation on rank {rank}/{world_size}...")

    val_dir = os.path.join(args.output_dir, "validation", f"step_{global_step}")
    os.makedirs(val_dir, exist_ok=True)

    num_inference_steps = args.validation_config.num_inference_steps
    latent_window_size = args.training_config.latent_window_size[0]
    history_sizes = args.training_config.history_sizes

    final_sigma_floor = getattr(args.validation_config, "final_sigma_floor", None)
    final_sigma_extra_step = getattr(args.validation_config, "final_sigma_extra_step", None)
    logger.info(
        f"  Terminal σ: final_sigma_floor={final_sigma_floor}, "
        f"final_sigma_extra_step={final_sigma_extra_step}"
    )

    # All ranks compute identical indices (deterministic seed=42)
    sample_indices = _select_validation_indices(train_dataset, args)

    if args.validation_config.first_step_valid:
        is_first_validation = (global_step == 1)
    else:
        is_first_validation = (global_step == args.validation_config.validation_steps)

    def _resolve_switch(mode_str):
        m = str(mode_str).lower()
        return m == "true" or (m == "first_only" and is_first_validation)

    run_multichunk_fixed = _resolve_switch(
        args.validation_config.enable_multichunk_fixed_validation
    )
    run_multichunk_pyramid = _resolve_switch(
        args.validation_config.enable_multichunk_pyramid_validation
    )

    _patch_size = transformer.config.patch_size
    _pyramid_steps_list = list(args.validation_config.stage2_simulated_inference_steps)
    _pyramid_scheduler = None
    if run_multichunk_pyramid:
        _pyramid_scheduler_kwargs = {}
        _pyramid_stage_range = getattr(args.validation_config, "pyramid_stage_range", None)
        if _pyramid_stage_range is not None:
            _pyramid_scheduler_kwargs["stage_range"] = list(_pyramid_stage_range)
        _pyramid_scheduler = HeliosScheduler.from_pretrained(
            args.model_config.pretrained_model_name_or_path,
            subfolder="scheduler",
            **_pyramid_scheduler_kwargs,
        )

    # ── Phase 1-4: Video generation (sharded across ranks) ──────────────────
    local_psnr = []
    local_ssim = []

    for global_i, idx in enumerate(sample_indices):
        if global_i % world_size != rank:
            continue
        i = global_i  # file names use global index to avoid collisions

        logger.info(f"  [rank {rank}] Validation sample {i} (dataset idx={idx})...")
        sample = train_dataset[idx]

        history_latents = sample["history_latents"].unsqueeze(0).to(
            accelerator.device, dtype=weight_dtype)
        target_latents = sample["target_latents"].unsqueeze(0).to(
            accelerator.device, dtype=weight_dtype)
        x0_latents = sample["x0_latents"].unsqueeze(0).to(
            accelerator.device, dtype=weight_dtype)
        prompt_embeds = sample["prompt_embeds"].unsqueeze(0).to(
            accelerator.device, dtype=weight_dtype)
        edit_prompt_embeds = sample["edit_prompt_embeds"].unsqueeze(0).to(
            accelerator.device, dtype=weight_dtype)

        # Phase 1: source video (decode once)
        ref_dir = os.path.join(args.output_dir, "validation", "reference")
        src_path = os.path.join(ref_dir, f"source_sample{i}.mp4")
        if not os.path.exists(src_path):
            os.makedirs(ref_dir, exist_ok=True)
            if args.training_config.edit_adapter_data_mode == "separated":
                src_latents = history_latents
            else:
                src_latents = torch.cat(
                    [x0_latents, history_latents[:, :, :10, :, :]], dim=2)
            src_pixels = vae.decode(
                (src_latents / latents_std + latents_mean).to(vae.dtype))[0]
            src_video = (src_pixels * 0.5 + 0.5).clamp(0, 1)
            src_video = (
                src_video[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255
            ).astype("uint8")
            save_video(src_video, src_path, fps=16)

        # Phase 2: edited video
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
            device=accelerator.device,
        )

        noise_gen = torch.Generator(device=accelerator.device).manual_seed(42 + idx)
        init_latents = torch.randn(
            target_latents.shape, generator=noise_gen,
            device=accelerator.device, dtype=target_latents.dtype,
        )

        latents = _denoise_one_sample(
            transformer=transformer,
            init_latents=init_latents,
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
            num_inference_steps=num_inference_steps,
            device=accelerator.device,
            weight_dtype=weight_dtype,
            generator=noise_gen,
            final_sigma_floor=final_sigma_floor,
            final_sigma_extra_step=final_sigma_extra_step,
        )

        torch.cuda.empty_cache()
        edited_pixels = vae.decode(
            (latents / latents_std + latents_mean).to(vae.dtype))[0]
        edited_video = (edited_pixels * 0.5 + 0.5).clamp(0, 1)
        edited_video = (
            edited_video[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255
        ).astype("uint8")
        edited_path = os.path.join(val_dir, f"edited_sample{i}.mp4")
        save_video(edited_video, edited_path, fps=16)

        if run_multichunk_fixed:
            multichunk_fixed_path = os.path.join(val_dir, f"multichunk_fixed_x0_sample{i}.mp4")
            multichunk_fixed_video = generate_multichunk_with_adapter(
                transformer=transformer, vae=vae, noise_scheduler=noise_scheduler,
                history_latents=history_latents, x0_latents=x0_latents,
                prompt_embeds=prompt_embeds, edit_prompt_embeds=edit_prompt_embeds,
                latents_mean=latents_mean, latents_std=latents_std,
                args=args, device=accelerator.device, weight_dtype=weight_dtype,
                num_chunks=3, num_inference_steps=num_inference_steps,
                x0_mode="fixed", seed=42, sample_idx=i,
            )
            save_video(multichunk_fixed_video, multichunk_fixed_path, fps=16)

        if run_multichunk_pyramid:
            multichunk_ap_path = os.path.join(val_dir, f"multichunk_adapter_pyramid_sample{i}.mp4")
            multichunk_ap_video = generate_multichunk_adapter_then_pyramid(
                transformer=transformer, vae=vae, noise_scheduler=noise_scheduler,
                history_latents=history_latents, x0_latents=x0_latents,
                prompt_embeds=prompt_embeds, edit_prompt_embeds=edit_prompt_embeds,
                latents_mean=latents_mean, latents_std=latents_std,
                args=args, device=accelerator.device, weight_dtype=weight_dtype,
                pyramid_denoise_fn=_run_stage2_denoise,
                pyramid_scheduler=_pyramid_scheduler,
                pyramid_num_stages=3,
                pyramid_num_inference_steps_list=_pyramid_steps_list,
                patch_size=_patch_size,
                num_chunks=3, num_inference_steps=num_inference_steps,
                seed=42, sample_idx=i,
            )
            save_video(multichunk_ap_video, multichunk_ap_path, fps=16)

        # Phase 3: GT (decode once)
        gt_path = os.path.join(ref_dir, f"gt_sample{i}.mp4")
        if not os.path.exists(gt_path):
            os.makedirs(ref_dir, exist_ok=True)
            gt_pixels = vae.decode(
                (target_latents / latents_std + latents_mean).to(vae.dtype))[0]
            gt_video = (gt_pixels * 0.5 + 0.5).clamp(0, 1)
            gt_video = (
                gt_video[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255
            ).astype("uint8")
            save_video(gt_video, gt_path, fps=16)
        else:
            gt_video = np.array(list(imageio.mimread(gt_path, memtest=False)))

        # Phase 4: metrics (CPU)
        torch.cuda.empty_cache()
        _metric_dev = torch.device("cpu")
        local_psnr.append(compute_psnr(edited_video, gt_video))
        local_ssim.append(compute_ssim(edited_video, gt_video, _metric_dev))
        logger.info(
            f"    [rank {rank}] Sample {i} metrics: "
            f"PSNR={local_psnr[-1]:.2f}, SSIM={local_ssim[-1]:.4f}"
        )

    # ── Gather video metrics across ranks ───────────────────────────────────
    accelerator.wait_for_everyone()

    local_metrics = {"edited_psnr": local_psnr, "edited_ssim": local_ssim}
    gathered_metrics_list = [None] * world_size
    dist.all_gather_object(gathered_metrics_list, local_metrics)

    # Rank 0 reconstructs full metric lists in original global order
    all_metrics = {"edited_psnr": [], "edited_ssim": []}
    if accelerator.is_main_process:
        for global_i in range(len(sample_indices)):
            src_rank = global_i % world_size
            local_idx = global_i // world_size
            all_metrics["edited_psnr"].append(
                gathered_metrics_list[src_rank]["edited_psnr"][local_idx])
            all_metrics["edited_ssim"].append(
                gathered_metrics_list[src_rank]["edited_ssim"][local_idx])

    # ── Phase 5: val_loss (sharded, then all-reduced) ───────────────────────
    logger.info(f"[rank {rank}] Computing val_loss...")
    val_loss_sigma_levels = list(getattr(
        args.validation_config, "val_loss_sigma_levels",
        [0.1, 0.3, 0.5, 0.7, 0.9],
    ))
    local_sigma_sums = {s: 0.0 for s in val_loss_sigma_levels}
    local_sigma_counts = {s: 0 for s in val_loss_sigma_levels}

    val_loss_sample_indices = _select_validation_indices(train_dataset, args, mode="val_loss")

    for global_i, idx in enumerate(val_loss_sample_indices):
        if global_i % world_size != rank:
            continue

        sample = train_dataset[idx]
        _hist = sample["history_latents"].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        _tgt = sample["target_latents"].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        _x0 = sample["x0_latents"].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        _pe = sample["prompt_embeds"].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        _epe = sample["edit_prompt_embeds"].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)

        (
            _model_input, _idx_hs,
            _idx_hist_s, _idx_hist_m, _idx_hist_l,
            _hist_s, _hist_m, _hist_l,
        ) = prepare_stage1_clean_input_from_latents(
            history_latents=_hist, target_latents=_tgt, x0_latents=_x0,
            latent_window_size=latent_window_size, history_sizes=history_sizes,
            is_random_drop=False, is_keep_x0=True,
            dtype=weight_dtype, device=accelerator.device,
        )

        for s_idx, sigma_val in enumerate(val_loss_sigma_levels):
            _ng = torch.Generator(device=accelerator.device).manual_seed(
                12345 + global_i * 100 + s_idx)
            _noise = torch.randn(_model_input.shape, generator=_ng,
                                 device=accelerator.device, dtype=_model_input.dtype)
            sigma = torch.tensor([sigma_val], device=accelerator.device, dtype=_model_input.dtype)

            if args.training_config.use_dynamic_shifting:
                base_sigmas = noise_scheduler.sigmas
                _closest_idx = (base_sigmas - sigma_val).abs().argmin().item()
                shifted_sigmas = apply_schedule_shift(
                    base_sigmas, _noise,
                    base_seq_len=args.training_config.base_seq_len,
                    max_seq_len=args.training_config.max_seq_len,
                    base_shift=args.training_config.base_shift,
                    max_shift=args.training_config.max_shift,
                )
                sigma = shifted_sigmas[_closest_idx].flatten()
                _ts = sigma * 1000.0
            else:
                _ts = sigma * 1000.0

            _ts = _ts.to(_model_input.device, dtype=_model_input.dtype)
            while len(sigma.shape) < _model_input.ndim:
                sigma = sigma.unsqueeze(-1)
            sigma = sigma.to(_model_input.device, dtype=_model_input.dtype)

            noisy_input = (1.0 - sigma) * _model_input + sigma * _noise
            _target = _noise - _model_input

            model_pred = transformer(
                hidden_states=noisy_input,
                timestep=_ts.expand(_model_input.shape[0]),
                encoder_hidden_states=_pe,
                indices_hidden_states=_idx_hs,
                indices_latents_history_short=_idx_hist_s,
                indices_latents_history_mid=_idx_hist_m,
                indices_latents_history_long=_idx_hist_l,
                latents_history_short=_hist_s,
                latents_history_mid=_hist_m,
                latents_history_long=_hist_l,
                edit_instruction_embeds=_epe,
                return_dict=False,
            )[0]

            sigma_for_w = sigma.flatten()[:1]
            w = compute_loss_weighting_for_sd3(
                weighting_scheme=args.training_config.weighting_scheme,
                sigmas=sigma_for_w,
            )
            _loss_val = (w.float() * (model_pred.float() - _target.float()) ** 2).mean().item()
            local_sigma_sums[sigma_val] += _loss_val
            local_sigma_counts[sigma_val] += 1

    # All-reduce val_loss across ranks
    accelerator.wait_for_everyone()
    per_sigma_losses = {}
    for s in val_loss_sigma_levels:
        sum_t = torch.tensor([local_sigma_sums[s]], device=accelerator.device, dtype=torch.float64)
        cnt_t = torch.tensor([local_sigma_counts[s]], device=accelerator.device, dtype=torch.long)
        dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(cnt_t, op=dist.ReduceOp.SUM)
        per_sigma_losses[s] = sum_t.item() / max(cnt_t.item(), 1)

    _all_vals = list(per_sigma_losses.values())
    val_loss_mean = sum(_all_vals) / len(_all_vals) if _all_vals else 0.0
    logger.info(f"  val_loss={val_loss_mean:.6f}")

    # ── Tensorboard logging (rank 0 only) ──────────────────────────────────
    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                for key, vals in all_metrics.items():
                    if vals:
                        tracker.writer.add_scalar(
                            f"val/{key}", sum(vals) / len(vals), global_step)
                tracker.writer.add_scalar("val/val_loss", val_loss_mean, global_step)
                for s in val_loss_sigma_levels:
                    tracker.writer.add_scalar(
                        f"val/val_loss_sigma_{s}", per_sigma_losses[s], global_step)

    torch.cuda.empty_cache()
    logger.info("Distributed validation complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Main training body
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(
        backend="nccl", timeout=timedelta(seconds=5400)
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.training_config.gradient_accumulation_steps,
        mixed_precision=args.training_config.mixed_precision,
        log_with=args.report_to.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs, init_kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(args.output_dir, "train.log"), mode="a"
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
            )
        )
        logging.getLogger().addHandler(file_handler)

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_config.pretrained_model_name_or_path, subfolder="tokenizer",
    )
    logger.info("Loading noise scheduler...")
    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    logger.info("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        args.model_config.pretrained_model_name_or_path,
        subfolder="vae", torch_dtype=torch.float32,
    )
    vae.requires_grad_(False)
    vae.eval()
    logger.info("Loading text encoder...")
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.model_config.pretrained_model_name_or_path,
        subfolder="text_encoder", torch_dtype=weight_dtype,
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(
        accelerator.device, weight_dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(
        accelerator.device, weight_dtype)

    logger.info("Loading transformer with edit adapter...")
    enable_temporal_sa = args.training_config.edit_adapter_enable_temporal_self_attn
    edit_adapter_config = {
        "adapter_dim": args.training_config.edit_adapter_dim,
        "num_heads": args.training_config.edit_adapter_num_heads,
        "text_dim": args.training_config.edit_adapter_text_dim,
        "eps": args.training_config.edit_adapter_eps,
        "enable_temporal_self_attn": enable_temporal_sa,
        "temporal_max_len": args.training_config.edit_adapter_temporal_max_len,
        "history_adapter_dim": args.training_config.edit_adapter_history_dim,
        "history_num_heads": args.training_config.edit_adapter_history_num_heads,
        "history_num_frames": args.training_config.edit_adapter_history_num_frames,
    }
    logger.info(
        f"Edit adapter: enable_temporal_self_attn={enable_temporal_sa}, "
        f"temporal_max_len={args.training_config.edit_adapter_temporal_max_len}"
    )

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
    transformer.edit_adapter.requires_grad_(True)

    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in transformer.parameters())
    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    for name, param in transformer.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(weight_dtype)
    transformer.to(accelerator.device)

    if args.training_config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

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
        logger.warning(f"Could not apply kernel optimizations: {e}")

    logger.info("Building dataset...")
    feature_folders = args.data_config.instance_data_root
    train_dataset = EditAdapterDataset(
        feature_folder=feature_folders,
        history_sizes=args.training_config.history_sizes,
        latent_window_size=args.training_config.latent_window_size[0],
        seed=args.seed,
        data_mode=args.training_config.edit_adapter_data_mode,
        filter_filelist=args.data_config.filter_filelist,
    )
    num_train_samples = args.data_config.num_train_samples
    if num_train_samples is not None and num_train_samples < len(train_dataset):
        train_dataset.samples = train_dataset.samples[:num_train_samples]
        from collections import defaultdict
        new_buckets = defaultdict(list)
        for i, s in enumerate(train_dataset.samples):
            new_buckets[s["bucket_key"]].append(i)
        train_dataset.buckets = new_buckets
        logger.info(f"Truncated dataset to {num_train_samples} samples")

    logger.info(f"Dataset size: {len(train_dataset)} samples")

    sampler = EditAdapterSampler(
        dataset=train_dataset,
        batch_size=args.training_config.train_batch_size,
        drop_last=True, shuffle=True, seed=args.seed,
        num_sp_groups=accelerator.num_processes, sp_world_size=1,
        global_rank=accelerator.process_index,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=edit_adapter_collate_fn,
        num_workers=args.data_config.dataloader_num_workers,
        pin_memory=args.data_config.pin_memory,
        persistent_workers=args.data_config.persistent_workers
        if args.data_config.dataloader_num_workers > 0 else False,
        prefetch_factor=args.data_config.prefetch_factor
        if args.data_config.dataloader_num_workers > 0 else None,
    )

    trainable_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters()))
    params_to_optimize = [
        {"params": trainable_parameters, "lr": args.training_config.learning_rate}
    ]
    optimizer = get_optimizer(args, accelerator, params_to_optimize)

    lr_scheduler = get_scheduler(
        args.training_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.training_config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.training_config.max_train_steps * accelerator.num_processes,
        num_cycles=args.training_config.lr_num_cycles,
        power=args.training_config.lr_power,
    )

    ema_model = None

    def _save_adapter_state_dict(state_dict, path, is_ema=False):
        if args.training_config.save_safetensors:
            st_path = path.replace(".pth", ".safetensors")
            if is_ema:
                safetensors_save_file(state_dict["shadow_model"], st_path)
                meta_path = st_path + ".meta.pt"
                meta = {k: v for k, v in state_dict.items()
                        if k != "shadow_model"}
                torch.save(meta, meta_path)
            else:
                safetensors_save_file(state_dict, st_path)
        else:
            torch.save(state_dict, path)

    def _load_adapter_state_dict(path, is_ema=False):
        st_path = path.replace(".pth", ".safetensors")
        if os.path.exists(st_path):
            sd = safetensors_load_file(st_path, device="cpu")
            if is_ema:
                result = {"shadow_model": sd}
                meta_path = st_path + ".meta.pt"
                if os.path.exists(meta_path):
                    meta = torch.load(meta_path, map_location="cpu")
                    result.update(meta)
                return result
            return sd
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=True)
        return None

    def save_model_hook(models, weights, output_dir):
        nonlocal ema_model
        if accelerator.is_main_process:
            for model in models:
                model = unwrap_model(model, accelerator)
                if hasattr(model, "edit_adapter"):
                    adapter_path = os.path.join(output_dir, "edit_adapter.pth")
                    _save_adapter_state_dict(
                        model.edit_adapter.state_dict(), adapter_path)
                    if ema_model is not None:
                        ema_path = os.path.join(output_dir, "edit_adapter_ema.pth")
                        _save_adapter_state_dict(
                            ema_model.state_dict(), ema_path, is_ema=True)
                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        nonlocal ema_model
        while len(models) > 0:
            model = models.pop()
            model = unwrap_model(model, accelerator)
            if hasattr(model, "edit_adapter"):
                adapter_path = os.path.join(input_dir, "edit_adapter.pth")
                state_dict = _load_adapter_state_dict(adapter_path)
                if state_dict is not None:
                    missing, unexpected = model.edit_adapter.load_state_dict(
                        state_dict, strict=False)
                    non_temporal_missing = [
                        k for k in missing if "temporal_self_attn" not in k]
                    if non_temporal_missing:
                        raise RuntimeError(
                            f"Unexpected missing keys when loading adapter: "
                            f"{non_temporal_missing}")
                    if unexpected:
                        raise RuntimeError(
                            f"Unexpected keys when loading adapter: {unexpected}")
                    logger.info(
                        f"Loaded edit adapter from {input_dir} "
                        f"(missing temporal_self_attn keys: {len(missing)})")
                ema_path = os.path.join(input_dir, "edit_adapter_ema.pth")
                ema_sd = _load_adapter_state_dict(ema_path, is_ema=True)
                if ema_model is not None and ema_sd is not None:
                    try:
                        ema_model.load_state_dict(ema_sd)
                    except Exception as e:
                        logger.warning(
                            f"EMA load failed ({e}); reinitializing EMA from current weights")
                    else:
                        logger.info(f"Loaded EMA adapter from {input_dir}")

        if args.training_config.mixed_precision != "no":
            cast_training_params([transformer])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    transformer, optimizer, lr_scheduler = accelerator.prepare(
        transformer, optimizer, lr_scheduler)

    if args.training_config.use_ema:
        ema_model = EMAAdapter(
            model=unwrap_model(transformer, accelerator).edit_adapter,
            decay=args.training_config.ema_decay,
            update_after_step=args.training_config.ema_start_step,
        )
        logger.info(
            f"EMA enabled: decay={args.training_config.ema_decay}, "
            f"start_step={args.training_config.ema_start_step}")

    num_update_steps_per_epoch = math.ceil(
        len(sampler) / args.training_config.gradient_accumulation_steps)
    if args.training_config.max_train_steps is None:
        args.training_config.max_train_steps = (
            args.training_config.num_train_epochs * num_update_steps_per_epoch)

    global_step = 0
    first_epoch = 0

    if args.training_config.resume_from_checkpoint:
        if args.training_config.resume_from_checkpoint == "latest":
            if os.path.exists(args.output_dir):
                dirs = [d for d in os.listdir(args.output_dir)
                        if d.startswith("checkpoint")]
                if dirs:
                    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                    path = os.path.join(args.output_dir, dirs[-1])
                else:
                    path = None
            else:
                path = None
        else:
            path = args.training_config.resume_from_checkpoint

        if path is not None and os.path.exists(path):
            # Detect external checkpoint (not inside current output_dir):
            # load weights only, reset step counter to 0 for a fresh run.
            is_external = not os.path.abspath(path).startswith(
                os.path.abspath(args.output_dir))
            if is_external:
                logger.info(
                    f"Loading external checkpoint (Phase 2 fine-tune): {path}")
                logger.info(
                    "  Loading model weights only; reinitializing "
                    "optimizer + lr_scheduler from scratch.")
                accelerator.load_state(path)
                # Reset the already-prepared optimizer for Phase 2:
                # clear Phase 1 Adam momentum/variance, set Phase 2 lr.
                # Must also remove 'initial_lr' — load_state_dict copies it
                # from Phase 1, and LambdaLR uses setdefault which won't
                # overwrite an existing value.
                optimizer.state.clear()
                for pg in optimizer.param_groups:
                    pg['lr'] = args.training_config.learning_rate
                    pg.pop('initial_lr', None)
                # Recreate scheduler on the same (already-prepared) optimizer.
                # This is a raw LambdaLR (not AcceleratedScheduler), so it
                # steps once per micro-batch call. Multiply by
                # gradient_accumulation_steps to match global step count.
                # (Phase 1's AcceleratedScheduler uses * num_processes because
                # it internally loops num_processes times per .step() call.)
                ga_steps = args.training_config.gradient_accumulation_steps
                lr_scheduler = get_scheduler(
                    args.training_config.lr_scheduler,
                    optimizer=optimizer,
                    num_warmup_steps=(args.training_config.lr_warmup_steps
                                     * ga_steps),
                    num_training_steps=(args.training_config.max_train_steps
                                       * ga_steps),
                    num_cycles=args.training_config.lr_num_cycles,
                    power=args.training_config.lr_power,
                )
                if accelerator._schedulers:
                    accelerator._schedulers[0] = lr_scheduler
                # Ensure EMA starts with full decay so Phase 1's shadow
                # weights are preserved instead of being overwritten.
                if ema_model is not None:
                    if ema_model.optimization_step == 0:
                        ema_model.optimization_step = int(
                            os.path.basename(path).split("-")[1])
                    logger.info(
                        f"  EMA optimization_step set to "
                        f"{ema_model.optimization_step}")
                logger.info(
                    f"  Phase 2 LR: initial_lr="
                    f"{optimizer.param_groups[0].get('initial_lr', 'N/A')}, "
                    f"current_lr={optimizer.param_groups[0]['lr']}, "
                    f"scheduler_base_lrs={lr_scheduler.base_lrs}")
                logger.info(
                    "  Optimizer and lr_scheduler reinitialized for Phase 2.")
            else:
                logger.info(f"Resuming from checkpoint: {path}")
                accelerator.load_state(path)
                global_step = int(os.path.basename(path).split("-")[1])
                first_epoch = global_step // num_update_steps_per_epoch
        else:
            logger.info("No checkpoint found, starting from scratch.")

    if accelerator.is_main_process:
        tracker_name = args.report_to.tracker_name or "edit-adapter-train"
        from omegaconf import OmegaConf
        flat_config = OmegaConf.to_container(args, resolve=True)

        def _flatten_for_tb(d, prefix=""):
            out = {}
            for k, v in d.items():
                key = f"{prefix}{k}" if prefix else k
                if isinstance(v, dict):
                    out.update(_flatten_for_tb(v, prefix=f"{key}/"))
                elif isinstance(v, (list, tuple)):
                    out[key] = str(v)
                elif v is None:
                    out[key] = "None"
                else:
                    out[key] = v
            return out

        accelerator.init_trackers(tracker_name)
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                flat = _flatten_for_tb(flat_config)
                config_text = "\n".join(f"**{k}**: {v}" for k, v in sorted(flat.items()))
                tracker.writer.add_text("config", config_text, 0)
                tracker.writer.flush()

        # Save merged config (YAML + CLI overrides) for reproducibility
        import yaml as _yaml
        config_save_path = os.path.join(args.output_dir, "run_config.yaml")
        with open(config_save_path, "w") as f:
            _yaml.dump(
                OmegaConf.to_container(args, resolve=True),
                f, default_flow_style=False, allow_unicode=True,
            )
        logger.info(f"Saved merged run config to {config_save_path}")

    total_batch_size = (
        args.training_config.train_batch_size
        * accelerator.num_processes
        * args.training_config.gradient_accumulation_steps)
    logger.info("***** Running Edit Adapter Training *****")
    logger.info(f"  Num samples = {len(train_dataset)}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Max train steps = {args.training_config.max_train_steps}")
    logger.info(f"  Trainable params = {trainable_params:,}")
    logger.info(
        f"  Terminal σ: final_sigma_floor={args.validation_config.final_sigma_floor}, "
        f"final_sigma_extra_step={args.validation_config.final_sigma_extra_step}")

    progress_bar = tqdm(
        range(global_step, args.training_config.max_train_steps),
        desc="Steps", disable=not accelerator.is_local_main_process,
    )

    latent_window_size = args.training_config.latent_window_size[0]
    history_sizes = args.training_config.history_sizes

    for epoch in range(first_epoch, args.training_config.num_train_epochs):
        transformer.train()
        sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                history_latents = batch["history_latents"].to(accelerator.device, dtype=weight_dtype)
                target_latents = batch["target_latents"].to(accelerator.device, dtype=weight_dtype)
                x0_latents = batch["x0_latents"].to(accelerator.device, dtype=weight_dtype)
                prompt_embeds = batch["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                edit_prompt_embeds = batch["edit_prompt_embeds"].to(accelerator.device, dtype=weight_dtype)

                (
                    model_input,
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
                    device=accelerator.device,
                )

                if args.training_config.corrupt_history and latents_history_short is not None:
                    latents_history_short, latents_history_mid, latents_history_long = corrupt_history_latents(
                        latents_history_short,
                        latents_history_mid,
                        latents_history_long,
                        latent_window_size,
                        is_keep_x0=True,
                        corrupt_mode=args.training_config.corrupt_mode_history,
                        noise_mode_prob=args.training_config.corrupt_mode_prob_history,
                        is_frame_independent=args.training_config.is_frame_independent_corrupt_history,
                        is_chunk_independent=args.training_config.is_chunk_independent_corrupt_history,
                        corrupt_ratio_1x=args.training_config.noise_corrupt_ratio_history_short,
                        corrupt_ratio_2x=args.training_config.noise_corrupt_ratio_history_mid,
                        corrupt_ratio_4x=args.training_config.noise_corrupt_ratio_history_long,
                        noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_history,
                        downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_history,
                        downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_history,
                    )

                (
                    noisy_model_input, sigmas, timesteps, target,
                ) = prepare_noise_input_edit_adapter(
                    args=args, model_input=model_input,
                    noise_scheduler=noise_scheduler,
                )

            with accelerator.accumulate(transformer):
                loss = flow_loss_edit_adapter(
                    args=args, accelerator=accelerator,
                    transformer=transformer,
                    prompt_embeds=prompt_embeds,
                    edit_prompt_embeds=edit_prompt_embeds,
                    noisy_model_input=noisy_model_input,
                    sigmas=sigmas, timesteps=timesteps, target=target,
                    indices_hidden_states=indices_hidden_states,
                    latents_history_short=latents_history_short,
                    indices_latents_history_short=indices_latents_history_short,
                    latents_history_mid=latents_history_mid,
                    indices_latents_history_mid=indices_latents_history_mid,
                    latents_history_long=latents_history_long,
                    indices_latents_history_long=indices_latents_history_long,
                )

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        transformer.parameters(),
                        args.training_config.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if ema_model is not None:
                    ema_model.step(
                        unwrap_model(transformer, accelerator).edit_adapter.parameters())

                adapter = unwrap_model(transformer, accelerator).edit_adapter
                to_out_norms = adapter.get_to_out_norms()
                to_out_norm_mean = sum(to_out_norms) / len(to_out_norms)

                logs = {
                    "train/loss": loss.detach().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item()
                    if torch.is_tensor(grad_norm) else grad_norm,
                    "train/to_out_norm_mean": to_out_norm_mean,
                    "train/epoch": epoch,
                }

                if enable_temporal_sa:
                    sa_norms = adapter.get_temporal_to_out_norms()
                    if sa_norms:
                        logs["train/temporal_to_out_norm_mean"] = sum(sa_norms) / len(sa_norms)

                if ema_model is not None and ema_model.cur_decay_value is not None:
                    logs["train/ema_decay"] = ema_model.cur_decay_value
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step % args.training_config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        if args.training_config.checkpoints_total_limit is not None:
                            checkpoints = sorted(
                                [d for d in os.listdir(args.output_dir)
                                 if d.startswith("checkpoint")],
                                key=lambda x: int(x.split("-")[1]),
                            )
                            if len(checkpoints) > args.training_config.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints)
                                    - args.training_config.checkpoints_total_limit)
                                for ckpt in checkpoints[:num_to_remove]:
                                    ckpt_path = os.path.join(args.output_dir, ckpt)
                                    shutil.rmtree(ckpt_path)
                        logger.info(f"Saved checkpoint at step {global_step}")

                should_validate = (
                    args.validation_config.validation_steps > 0
                    and global_step % args.validation_config.validation_steps == 0)
                if args.validation_config.first_step_valid and global_step == 1:
                    should_validate = True

                if should_validate:
                    logger.info(f"Running validation at step {global_step}...")
                    transformer.eval()

                    # Offload optimizer states to CPU to free ~16-20GB VRAM
                    _opt_device_backup = {}
                    for param_id, state in optimizer.state.items():
                        _opt_device_backup[param_id] = {}
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor) and v.is_cuda:
                                _opt_device_backup[param_id][k] = v
                                state[k] = v.to("cpu", non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    logger.info("  Offloaded optimizer states to CPU for validation")

                    if ema_model is not None and args.training_config.use_ema_validation:
                        _adapter_params = list(
                            unwrap_model(transformer, accelerator).edit_adapter.parameters())
                        ema_model.store(_adapter_params)
                        ema_model.copy_to(_adapter_params)
                        if accelerator.is_main_process:
                            logger.info(
                                f"  Using EMA weights for validation "
                                f"(decay={ema_model.cur_decay_value:.6f})")

                    log_validation_edit_adapter(
                        transformer=unwrap_model(transformer, accelerator),
                        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                        noise_scheduler=noise_scheduler,
                        train_dataset=train_dataset,
                        args=args, accelerator=accelerator,
                        global_step=global_step,
                        weight_dtype=weight_dtype,
                        latents_mean=latents_mean,
                        latents_std=latents_std,
                    )

                    if ema_model is not None and args.training_config.use_ema_validation:
                        _adapter_params = list(
                            unwrap_model(transformer, accelerator).edit_adapter.parameters())
                        ema_model.restore(_adapter_params)

                    transformer.train()

                    # Reload optimizer states back to GPU
                    for param_id, saved in _opt_device_backup.items():
                        for k, v in saved.items():
                            optimizer.state[param_id][k] = v.to(
                                accelerator.device, non_blocking=True)
                    torch.cuda.synchronize()
                    del _opt_device_backup
                    torch.cuda.empty_cache()
                    logger.info("  Reloaded optimizer states to GPU")

                    accelerator.wait_for_everyone()

            if global_step >= args.training_config.max_train_steps:
                break

        if global_step >= args.training_config.max_train_steps:
            break

    if accelerator.is_main_process:
        model = unwrap_model(transformer, accelerator)
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-final")
        os.makedirs(save_path, exist_ok=True)
        adapter_path = os.path.join(save_path, "edit_adapter.pth")
        _save_adapter_state_dict(model.edit_adapter.state_dict(), adapter_path)
        if ema_model is not None:
            ema_path = os.path.join(save_path, "edit_adapter_ema.pth")
            _save_adapter_state_dict(ema_model.state_dict(), ema_path, is_ema=True)

    accelerator.end_training()
    logger.info("Training complete!")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_dirs", type=str, default=None)
    parser.add_argument("--reweighting_along_time", action="store_true", default=False)
    parser.add_argument("--temporal_weight_max_ratio", type=float, default=None)
    parser.add_argument("--enable_temporal_self_attn", action="store_true", default=False)
    parser.add_argument("--temporal_max_len", type=int, default=None)
    # Terminal-σ knobs for single-stage validation
    parser.add_argument("--final_sigma_floor", type=float, default=None,
                        help="Replace the schedule's last σ with this smaller value (no extra step).")
    parser.add_argument("--final_sigma_extra_step", type=float, default=None,
                        help="Append one extra forward at this σ after the standard schedule.")
    parser.add_argument("--weighting_scheme", type=str, default=None,
                        help="Override yaml's training_config.weighting_scheme. "
                             "'logit_normal' (default in yaml) → uses logit_mean/logit_std. "
                             "'uniform' → u ~ U(0,1), giving sigma density that exactly matches the "
                             "16-step inference schedule's coverage of [0.06, 0.999]. "
                             "Note: logit_mean/logit_std are silently ignored in modes other than logit_normal.")
    parser.add_argument("--use_mixture_sigma", action="store_true", default=False,
                        help="Override sigma_sampling_strategy to 'mixture_gaussian'. "
                             "Samples sigma from a mixture of Gaussians centered at the 16 "
                             "inference sigma values instead of the default logit_normal.")
    cli_args = parser.parse_args()

    config = OmegaConf.load(cli_args.config)
    schema = OmegaConf.structured(Args)
    conf = OmegaConf.merge(schema, config)

    if cli_args.output_dir is not None:
        conf.output_dir = cli_args.output_dir
    if cli_args.data_dirs is not None:
        conf.data_config.instance_data_root = cli_args.data_dirs.split(",")
    if cli_args.reweighting_along_time:
        conf.training_config.reweighting_along_time = cli_args.reweighting_along_time
    if cli_args.temporal_weight_max_ratio is not None:
        conf.training_config.temporal_weight_max_ratio = cli_args.temporal_weight_max_ratio
    if cli_args.enable_temporal_self_attn:
        conf.training_config.edit_adapter_enable_temporal_self_attn = True
    if cli_args.temporal_max_len is not None:
        conf.training_config.edit_adapter_temporal_max_len = cli_args.temporal_max_len
    if cli_args.final_sigma_floor is not None:
        conf.validation_config.final_sigma_floor = cli_args.final_sigma_floor
    if cli_args.final_sigma_extra_step is not None:
        conf.validation_config.final_sigma_extra_step = cli_args.final_sigma_extra_step
    if cli_args.weighting_scheme is not None:
        conf.training_config.weighting_scheme = cli_args.weighting_scheme
    if cli_args.use_mixture_sigma:
        conf.training_config.sigma_sampling_strategy = "mixture_gaussian"

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != conf.training_config.local_rank:
        conf.training_config.local_rank = env_local_rank

    main(conf)
