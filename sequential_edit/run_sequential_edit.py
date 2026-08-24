"""
Multi-round sequential editing on real source videos.

The source videos can be supplied either as pre-encoded .pt files (see
``scripts/preprocess/``) or as raw .mp4 files. When .mp4 files are used, the
script encodes them on-the-fly with the VAE and encodes the scene prompt from
the CSV with T5. The edit instructions for each round come from a test CSV and
are always encoded with T5 at runtime.

A round produces one Edit Adapter chunk followed by ``chunks_per_edit - 1``
pyramid continuation chunks; the tail of each round becomes the history for the
next round, so edits accumulate over the whole video.

Launch:
    accelerate launch --config_file <accel.yaml> \
        sequential_edit/run_sequential_edit.py \
        --adapter_ckpt <path> \
        --config <yaml> \
        --data_dir <dir_with_pt_or_mp4_files> \
        --test_csv sequential_edit/benchmark.csv \
        --output_dir <dir>
"""

import argparse
import csv
import glob
import os
import sys
from datetime import timedelta
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torchvision.io
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from diffusers import AutoencoderKLWan
from transformers import AutoTokenizer, UMT5EncoderModel

from helios.modules.transformer_helios_edit import (
    HeliosTransformer3DModelWithEditAdapter,
)
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
from omegaconf import OmegaConf
from helios.utils.train_config import Args
from helios.utils.utils_base import encode_prompt
from helios.utils.utils_helios_base import prepare_stage1_clean_input_from_latents

from run_edit_adapter_inference import (
    save_video,
    detect_adapter_features,
    load_adapter_into_model,
    _find_adapter_path,
    _load_state_dict_auto,
    run_single_stage_denoise,
    run_pyramid_denoise,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-round sequential editing on pre-encoded source videos"
    )
    parser.add_argument("--adapter_ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .pt (pre-encoded) or .mp4 (raw video) files")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="CSV with edit_1/2/3 columns for edit instructions")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--use_ema", action="store_true", default=True)

    # Adapter edit params
    parser.add_argument("--edit_num_inference_steps", type=int, default=16)
    parser.add_argument("--final_sigma_floor", type=float, default=None)
    parser.add_argument("--final_sigma_extra_step", type=float, default=0.01)

    # Pyramid continuation params
    parser.add_argument("--pyramid_num_stages", type=int, default=3)
    parser.add_argument("--pyramid_steps", type=int, nargs="+", default=[2, 2, 2])
    parser.add_argument("--chunks_per_edit", type=int, default=3)

    # Base model path
    parser.add_argument("--base_model_path", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_history_adapter", action="store_true", default=False,
                        help="Load new adapter with history cross-attention module")
    return parser.parse_args()


def load_test_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def find_source_files(data_dir, num_samples=None):
    """Find .pt or .mp4 source files in data_dir (or its immediate subdirs)."""
    pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    if not pt_files:
        for sub in sorted(Path(data_dir).iterdir()):
            if sub.is_dir():
                pt_files.extend(sorted(glob.glob(str(sub / "*.pt"))))
    if pt_files:
        return pt_files if num_samples is None else pt_files[:num_samples]

    mp4_files = sorted(glob.glob(os.path.join(data_dir, "*.mp4")))
    if not mp4_files:
        for sub in sorted(Path(data_dir).iterdir()):
            if sub.is_dir():
                mp4_files.extend(sorted(glob.glob(str(sub / "*.mp4"))))
    if not mp4_files:
        raise ValueError(f"No .pt or .mp4 files found in {data_dir}")
    return mp4_files if num_samples is None else mp4_files[:num_samples]


def load_and_resize_video(video_path, target_height=384, target_width=640,
                          target_num_frames=73):
    """Load a video, temporally subsample, resize, and normalise to [-1, 1]."""
    video, _audio, _info = torchvision.io.read_video(video_path, pts_unit="sec")
    orig_num_frames = video.shape[0]

    step = max(1, orig_num_frames // target_num_frames)
    if step > 1:
        offset = (orig_num_frames - 1) % step
        video = video[offset::step]

    T = video.shape[0]
    valid_T = ((T - 1) // 4) * 4 + 1
    if valid_T < 1:
        valid_T = 1
    video = video[-valid_T:]

    video = video.permute(0, 3, 1, 2).float()
    video = TF.resize(video, [target_height, target_width], antialias=True)
    video = video / 127.5 - 1.0
    video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
    return video


def vae_encode_video(vae, pixel_values, latents_mean, latents_std, device):
    """Encode pixel video (C, T, H, W) → latent (C, T_lat, H_lat, W_lat)."""
    pixel_values = pixel_values.unsqueeze(0).to(dtype=vae.dtype, device=device)
    T = pixel_values.shape[2]
    frame_window_size = 33  # (9 - 1) * 4 + 1

    if T <= frame_window_size:
        with torch.no_grad():
            latent = vae.encode(pixel_values).latent_dist.sample()
            latent = (latent - latents_mean) * latents_std
        return latent.squeeze(0)

    num_chunks = T // frame_window_size
    chunk_latents = []
    for i in range(num_chunks):
        start = i * frame_window_size
        chunk = pixel_values[:, :, start:start + frame_window_size, :, :]
        with torch.no_grad():
            lat = vae.encode(chunk).latent_dist.sample()
            lat = (lat - latents_mean) * latents_std
        chunk_latents.append(lat.squeeze(0))

    remaining_start = num_chunks * frame_window_size
    remaining_frames = T - remaining_start
    if remaining_frames >= 1:
        valid_remaining = ((remaining_frames - 1) // 4) * 4 + 1
        if valid_remaining >= 1:
            chunk = pixel_values[:, :, remaining_start:remaining_start + valid_remaining, :, :]
            with torch.no_grad():
                lat = vae.encode(chunk).latent_dist.sample()
                lat = (lat - latents_mean) * latents_std
            chunk_latents.append(lat.squeeze(0))

    return torch.cat(chunk_latents, dim=1)


@torch.no_grad()
def adapter_edit_one_chunk(
    transformer, latents_noise, prompt_embeds, edit_prompt_embeds,
    history_latents, x0_latents,
    args, num_inference_steps, device, weight_dtype, generator,
    final_sigma_floor=None, final_sigma_extra_step=None,
):
    latent_window_size = args.training_config.latent_window_size[0]
    history_sizes = sorted(args.training_config.history_sizes, reverse=True)
    B, C, _, H, W = history_latents.shape

    dummy_target = torch.zeros(
        B, C, latent_window_size, H, W, device=device, dtype=weight_dtype
    )

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
        target_latents=dummy_target,
        x0_latents=x0_latents,
        latent_window_size=latent_window_size,
        history_sizes=history_sizes,
        is_random_drop=False,
        is_keep_x0=True,
        dtype=weight_dtype,
        device=device,
    )

    result = run_single_stage_denoise(
        transformer=transformer,
        latents=latents_noise,
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
        device=device,
        weight_dtype=weight_dtype,
        generator=generator,
        final_sigma_floor=final_sigma_floor,
        final_sigma_extra_step=final_sigma_extra_step,
    )
    return result


@torch.no_grad()
def pyramid_continue_one_chunk(
    transformer, scheduler, latents_noise, prompt_embeds,
    history_latents, x0_latents,
    args, pyramid_num_stages, pyramid_steps, patch_size,
    device, weight_dtype, generator,
):
    latent_window_size = args.training_config.latent_window_size[0]
    history_sizes = sorted(args.training_config.history_sizes, reverse=True)
    B, C, _, H, W = history_latents.shape

    dummy_target = torch.zeros(
        B, C, latent_window_size, H, W, device=device, dtype=weight_dtype
    )

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
        target_latents=dummy_target,
        x0_latents=x0_latents,
        latent_window_size=latent_window_size,
        history_sizes=history_sizes,
        is_random_drop=False,
        is_keep_x0=True,
        dtype=weight_dtype,
        device=device,
    )

    result = run_pyramid_denoise(
        transformer=transformer,
        scheduler=scheduler,
        latents=latents_noise,
        prompt_embeds=prompt_embeds,
        edit_instruction_embeds=None,
        indices_hidden_states=indices_hidden_states,
        indices_latents_history_short=indices_latents_history_short,
        indices_latents_history_mid=indices_latents_history_mid,
        indices_latents_history_long=indices_latents_history_long,
        latents_history_short=latents_history_short,
        latents_history_mid=latents_history_mid,
        latents_history_long=latents_history_long,
        pyramid_num_stages=pyramid_num_stages,
        pyramid_num_inference_steps_list=pyramid_steps,
        patch_size=patch_size,
        device=device,
        weight_dtype=weight_dtype,
        generator=generator,
        adapter_at_all_stages=False,
    )
    return result


@torch.no_grad()
def encode_text(text, tokenizer, text_encoder, device, dtype, max_length=512):
    # Match the preprocessing that produced the precomputed embeds
    # (encode_prompt -> _get_t5_prompt_embeds): prompt_clean + zero-padding of
    # pad positions. The transformer attends over the full 512 tokens with no
    # text mask, so raw non-zero T5 outputs at pad positions corrupt the
    # conditioning.
    embeds, _ = encode_prompt(
        tokenizer, text_encoder, text,
        max_sequence_length=max_length, device=device, dtype=dtype,
    )
    return embeds


def main():
    cli = parse_args()
    config = OmegaConf.load(cli.config)
    schema = OmegaConf.structured(Args)
    args = OmegaConf.merge(schema, config)

    # Accelerator init
    init_kwargs = InitProcessGroupKwargs(
        backend="nccl", timeout=timedelta(seconds=3600)
    )
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

    base_model_path = (
        cli.base_model_path
        or getattr(args.model_config, "pretrained_model_name_or_path", None)
        or getattr(args.model_config, "transformer_model_name_or_path", None)
    )
    if not base_model_path:
        raise ValueError("base_model_path is None.")

    # ─── Load transformer with adapter ───
    if is_main:
        print("[load] transformer + adapter...")
    adapter_path = _find_adapter_path(cli.adapter_ckpt, "edit_adapter")
    if adapter_path is None:
        raise FileNotFoundError(
            f"adapter checkpoint not found in {cli.adapter_ckpt}")
    peek_sd = _load_state_dict_auto(adapter_path)
    detected = detect_adapter_features(peek_sd)
    del peek_sd
    if is_main:
        print(f"[ckpt] detected features: {detected}")

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
    if cli.use_history_adapter:
        edit_adapter_config["history_adapter_dim"] = getattr(
            args.training_config, "edit_adapter_history_dim", 512)
        edit_adapter_config["history_num_heads"] = getattr(
            args.training_config, "edit_adapter_history_num_heads", 4)
        edit_adapter_config["history_num_frames"] = getattr(
            args.training_config, "edit_adapter_history_num_frames", 2)
    elif detected.get("has_history_cross_attn", False):
        print("[WARN] Checkpoint contains history_cross_attn weights but "
              "--use_history_adapter was not passed. Model may have missing keys.")

    transformer_additional_kwargs = {
        "has_multi_term_memory_patch": args.training_config.has_multi_term_memory_patch,
        "zero_history_timestep": args.training_config.zero_history_timestep,
        "restrict_self_attn": args.training_config.restrict_self_attn,
        "guidance_cross_attn": args.training_config.guidance_cross_attn,
    }

    transformer = (
        HeliosTransformer3DModelWithEditAdapter.from_pretrained_with_edit_adapter(
            args.model_config.transformer_model_name_or_path,
            subfolder="transformer",
            edit_adapter_config=edit_adapter_config,
            transformer_additional_kwargs=transformer_additional_kwargs,
        )
    )
    transformer.requires_grad_(False)
    for p in transformer.parameters():
        p.data = p.data.to(weight_dtype)
    transformer.to(device)
    transformer.eval()

    load_adapter_into_model(transformer, cli.adapter_ckpt, use_ema=cli.use_ema)
    if is_main:
        print("[load] adapter weights loaded")

    # ─── Load VAE ───
    if is_main:
        print("[load] VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        base_model_path, subfolder="vae", torch_dtype=torch.float32,
    )
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, -1, 1, 1, 1)
        .to(device, weight_dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(vae.config.latents_std)
        .view(1, -1, 1, 1, 1)
        .to(device, weight_dtype)
    )

    # ─── Load HeliosScheduler for pyramid DMD ───
    pyramid_scheduler = HeliosScheduler.from_pretrained(
        base_model_path, subfolder="scheduler"
    )

    # ─── Load tokenizer + text_encoder (for runtime edit instruction encoding) ───
    if is_main:
        print("[load] tokenizer + text_encoder...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, subfolder="tokenizer"
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_model_path, subfolder="text_encoder", torch_dtype=weight_dtype,
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    text_encoder.to(device)

    # ─── Config params ───
    latent_window_size = args.training_config.latent_window_size[0]  # 9
    history_sizes = sorted(
        args.training_config.history_sizes, reverse=True
    )  # [16, 2, 1]
    history_window_size = sum(history_sizes)  # 19
    patch_size = (1, 2, 2)
    num_channels = 16

    # ─── Load test CSV (for edit instructions) ───
    test_rows = load_test_csv(cli.test_csv)
    if is_main:
        print(f"[data] Loaded {len(test_rows)} edit-instruction sets from {cli.test_csv}")

    # Build clip_id -> csv_row lookup for correct matching
    csv_by_clip = {}
    for row in test_rows:
        cid = row.get("clip_id", "").replace(".mp4", "")
        if cid:
            csv_by_clip[cid] = row

    # ─── Find source files (.pt or .mp4) ───
    source_files = find_source_files(cli.data_dir)
    is_video_mode = source_files[0].endswith(".mp4")
    if is_main:
        fmt = "mp4 (raw video)" if is_video_mode else "pt (pre-encoded)"
        print(f"[data] Found {len(source_files)} {fmt} files in {cli.data_dir}")

    matched_files = [p for p in source_files if Path(p).stem in csv_by_clip]
    if is_main:
        print(f"[data] Matched {len(matched_files)}/{len(source_files)} files with CSV rows")
    source_files = matched_files[:cli.num_samples]
    if is_main:
        print(f"[data] Using {len(source_files)} samples (num_samples={cli.num_samples})")

    # ─── Output dir ───
    out_dir = Path(cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── Round-robin distribution ───
    for global_i, src_path in enumerate(source_files):
        if global_i % world_size != rank:
            continue

        csv_row = csv_by_clip[Path(src_path).stem]
        edits = [
            (csv_row["edit_1_type"], csv_row["edit_1"]),
            (csv_row["edit_2_type"], csv_row["edit_2"]),
            (csv_row["edit_3_type"], csv_row["edit_3"]),
        ]
        chunks_per_edit = [
            int(csv_row.get("chunks_edit_1", cli.chunks_per_edit)),
            int(csv_row.get("chunks_edit_2", cli.chunks_per_edit)),
            int(csv_row.get("chunks_edit_3", cli.chunks_per_edit)),
        ]

        sample_out_dir = out_dir / f"sample_{global_i:04d}"
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        output_path = sample_out_dir / "full_video.mp4"
        if output_path.exists():
            print(f"[rank {rank}] Skipping sample {global_i} (exists)")
            continue

        # ─── Load source data ───
        if is_video_mode:
            pixel_video = load_and_resize_video(src_path)
            src_lat = vae_encode_video(
                vae, pixel_video, latents_mean, latents_std, device
            ).to(torch.float16).cpu()
            scene_prompt_raw = csv_row.get("scene_prompt", "")
            prompt_embeds, _ = encode_prompt(
                tokenizer, text_encoder, scene_prompt_raw,
                max_sequence_length=512, device=device, dtype=weight_dtype,
            )
        else:
            data = torch.load(src_path, map_location="cpu", weights_only=False)
            src_lat = data["src_vae_latent"]  # (C, T_src, H, W), float16
            prompt_embeds = data["scene_prompt_embed"].unsqueeze(0).to(
                device, dtype=weight_dtype
            )  # (1, 512, 4096)
            scene_prompt_raw = data.get("scene_prompt_raw", "")

        print(
            f"[rank {rank}] Sample {global_i}: "
            f"src_lat={list(src_lat.shape)}, "
            f"edits: {edits[0][0]} → {edits[1][0]} → {edits[2][0]}"
        )

        # ─── Prepare source latents ───
        # .pt stores (C, T, H, W) — add batch dim → (1, C, T, H, W)
        src_lat_5d = src_lat.unsqueeze(0).to(device, dtype=weight_dtype)
        lat_h = src_lat_5d.shape[3]
        lat_w = src_lat_5d.shape[4]

        # history = last 19 frames of source, x0 = first frame of history
        # (matches training: dataloader_edit_adapter.py line 262)
        history_latents = src_lat_5d[:, :, -history_window_size:, :, :]
        x0_latents = history_latents[:, :, :1, :, :].clone()

        # Decode source video chunk-by-chunk and save
        src_T = src_lat_5d.shape[2]
        src_decoded_chunks = []
        for ci in range(0, src_T, latent_window_size):
            chunk_lat = src_lat_5d[:, :, ci : ci + latent_window_size, :, :]
            chunk_normed = (chunk_lat / latents_std + latents_mean).to(vae.dtype)
            chunk_pixels = vae.decode(chunk_normed)[0]
            src_decoded_chunks.append((chunk_pixels * 0.5 + 0.5).clamp(0, 1))
        src_pixel_video = torch.cat(src_decoded_chunks, dim=2)

        src_video_np = (
            src_pixel_video[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255
        ).astype("uint8")
        save_video(src_video_np, str(sample_out_dir / "source.mp4"))

        all_decoded_chunks = [src_pixel_video]
        generated_chunks = []  # generated only (no source), for generated_full.mp4
        per_edit_frames = []

        # ─── Sequential edits (3 different instructions from test CSV) ───
        sample_seed = cli.seed + global_i
        chunk_counter = 0

        for edit_idx, (edit_type, edit_instruction) in enumerate(edits):
            print(f"[rank {rank}]   Edit {edit_idx+1}/3: {edit_type}")
            edit_chunks = []  # this edit's chunks (adapter chunk + continuations)

            # Runtime-encode edit instruction (same path as run_sequential_edit_t2v)
            edit_prompt_embeds = encode_text(
                edit_instruction, tokenizer, text_encoder, device, weight_dtype
            )

            # Generate noise for adapter chunk
            chunk_counter += 1
            chunk_seed = sample_seed + chunk_counter * 100
            gen = torch.Generator(device=device).manual_seed(chunk_seed)
            noise = torch.randn(
                1, num_channels, latent_window_size, lat_h, lat_w,
                generator=gen, device=device, dtype=weight_dtype,
            )

            # Adapter edit
            edited_latents = adapter_edit_one_chunk(
                transformer=transformer,
                latents_noise=noise,
                prompt_embeds=prompt_embeds,
                edit_prompt_embeds=edit_prompt_embeds,
                history_latents=history_latents,
                x0_latents=x0_latents,
                args=args,
                num_inference_steps=cli.edit_num_inference_steps,
                device=device,
                weight_dtype=weight_dtype,
                generator=gen,
                final_sigma_floor=cli.final_sigma_floor,
                final_sigma_extra_step=cli.final_sigma_extra_step,
            )

            # Decode adapter chunk
            chunk_normed = (edited_latents / latents_std + latents_mean).to(vae.dtype)
            chunk_pixels = vae.decode(chunk_normed)[0]
            chunk_video = (chunk_pixels * 0.5 + 0.5).clamp(0, 1)
            all_decoded_chunks.append(chunk_video)
            generated_chunks.append(chunk_video)
            edit_chunks.append(chunk_video)

            # Update state (matches generate_multichunk_adapter_then_pyramid)
            x0_latents = edited_latents[:, :, :1, :, :].clone()
            history_latents = torch.cat(
                [history_latents, edited_latents], dim=2
            )[:, :, -history_window_size:, :, :]

            # Pyramid continuation chunks
            n_continuation = chunks_per_edit[edit_idx]
            for cont_idx in range(n_continuation):
                chunk_counter += 1
                chunk_seed = sample_seed + chunk_counter * 100
                gen = torch.Generator(device=device).manual_seed(chunk_seed)
                noise = torch.randn(
                    1, num_channels, latent_window_size, lat_h, lat_w,
                    generator=gen, device=device, dtype=weight_dtype,
                )

                cont_latents = pyramid_continue_one_chunk(
                    transformer=transformer,
                    scheduler=pyramid_scheduler,
                    latents_noise=noise,
                    prompt_embeds=prompt_embeds,
                    history_latents=history_latents,
                    x0_latents=x0_latents,
                    args=args,
                    pyramid_num_stages=cli.pyramid_num_stages,
                    pyramid_steps=cli.pyramid_steps,
                    patch_size=patch_size,
                    device=device,
                    weight_dtype=weight_dtype,
                    generator=gen,
                )

                # Decode
                chunk_normed = (cont_latents / latents_std + latents_mean).to(
                    vae.dtype
                )
                chunk_pixels = vae.decode(chunk_normed)[0]
                chunk_video = (chunk_pixels * 0.5 + 0.5).clamp(0, 1)
                all_decoded_chunks.append(chunk_video)
                generated_chunks.append(chunk_video)
                edit_chunks.append(chunk_video)

                # Update history (x0 stays as edited anchor)
                history_latents = torch.cat(
                    [history_latents, cont_latents], dim=2
                )[:, :, -history_window_size:, :, :]

            print(
                f"[rank {rank}]     + {n_continuation} pyramid chunks done"
            )

            # Save this edit's segment (adapter chunk + continuations) so the
            # output matches the backbone structure and the eval loader's
            # `*edit_{n}_*.mp4` naming.
            edit_video = torch.cat(edit_chunks, dim=2)
            edit_np = (
                edit_video[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255
            ).astype("uint8")
            save_video(
                edit_np,
                str(sample_out_dir / f"edit_{edit_idx+1}_{edit_type}.mp4"),
            )
            per_edit_frames.append(int(edit_np.shape[0]))

        # ─── Concatenate and save ───
        full_video = torch.cat(all_decoded_chunks, dim=2)
        video_np = (
            full_video[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255
        ).astype("uint8")
        save_video(video_np, str(output_path))

        # Generated-only video (no source prefix), matching backbone's
        # generated_full.mp4.
        generated_video = torch.cat(generated_chunks, dim=2)
        generated_np = (
            generated_video[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255
        ).astype("uint8")
        save_video(generated_np, str(sample_out_dir / "generated_full.mp4"))

        # Save metadata
        meta_path = sample_out_dir / "metadata.txt"
        with open(meta_path, "w") as f:
            f.write(f"source_file: {src_path}\n")
            f.write(f"scene_prompt: {scene_prompt_raw}\n")
            for i, (etype, einstr) in enumerate(edits):
                f.write(f"edit_{i+1}_type: {etype}\n")
                f.write(f"edit_{i+1}: {einstr}\n")
            f.write(f"chunks_per_edit: {chunks_per_edit}\n")
            f.write(f"per_edit_frames: {per_edit_frames}\n")
            f.write(f"total_chunks: {len(all_decoded_chunks)}\n")
            f.write(f"total_frames: {video_np.shape[0]}\n")
            f.write(f"src_lat_shape: {list(src_lat.shape)}\n")

        print(
            f"[rank {rank}]   Saved: {output_path} ({video_np.shape[0]} frames)"
        )

        del all_decoded_chunks, full_video, video_np, src_lat, src_lat_5d
        del generated_chunks, generated_video, generated_np
        torch.cuda.empty_cache()

    # ─── Barrier ───
    accelerator.wait_for_everyone()
    if is_main:
        print(f"[done] All samples processed. Output: {cli.output_dir}")


if __name__ == "__main__":
    main()
