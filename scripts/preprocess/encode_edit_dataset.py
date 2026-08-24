#!/usr/bin/env python3
"""
Pre-encode a paired (source video, edited video) dataset into VAE latents +
T5 prompt embeddings, with ratio-based temporal subsampling.

Temporal handling:
  - src video is subsampled to ~73 pixel frames (→ ~19 latent frames)
  - tgt video is subsampled to ~33 pixel frames (→ ~9 latent frames)

Subsampling strategy:
  step = max(1, orig_frames // target_frames)
  video = video[::step]
  Then 4n+1 alignment (no truncation beyond that).

Output .pt format:
    {
        "src_vae_latent":             (C, T_src, H_lat, W_lat),
        "tgt_vae_latent":             (C, T_tgt, H_lat, W_lat),
        "scene_prompt_embed":         (512, 4096),
        "scene_prompt_attention_mask": (512,),
        "edit_prompt_embed":          (512, 4096),
        "edit_prompt_attention_mask":  (512,),
        "scene_prompt_raw":           str,
        "edit_prompt_raw":            str,
        "src_latent_T":               int,
        "tgt_latent_T":               int,
        "src_orig_frames":            int,
        "tgt_orig_frames":            int,
    }

Launch (multi-GPU via accelerate):
    accelerate launch --num_processes=8 scripts/preprocess/encode_edit_dataset.py \
        --csv /path/to/style.csv \
        --output_dir /path/to/output \
        --pretrained_model_name_or_path /path/to/Helios-Distilled

Launch (single GPU):
    python scripts/preprocess/encode_edit_dataset.py --csv /path/to/csv
"""

import argparse
import csv
import logging
import os
import sys

import torch
import torchvision.io
import torchvision.transforms.functional as TF
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers import AutoencoderKLWan

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from helios.utils.utils_base import encode_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-encode a paired edit dataset into VAE latents + T5 embeddings")
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Input CSV with columns: src_video_path, tgt_video_path, src_video_caption, edit_video_instruction")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for .pt files.")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str,
        default="./pretrained/Helios-Distilled",
        help="HuggingFace model path for VAE, tokenizer, and text encoder")
    parser.add_argument("--target_height", type=int, default=384)
    parser.add_argument("--target_width", type=int, default=640)
    parser.add_argument("--src_target_frames", type=int, default=73,
                        help="Target number of pixel frames for src video (default: 73 → ~19 latent frames)")
    parser.add_argument("--tgt_target_frames", type=int, default=33,
                        help="Target number of pixel frames for tgt video (default: 33 → ~9 latent frames)")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to encode (None = all). "
                             "Already-encoded samples do NOT count towards this limit.")
    return parser.parse_args()


def load_csv_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_and_resize_video(video_path, target_height, target_width, target_num_frames, keep_tail=False):
    """Load a video file, ratio-based temporal subsample, and resize.

    Subsampling: step = max(1, orig_frames // target_num_frames), then video[::step].
    No truncation — only 4n+1 alignment after subsampling.

    Args:
        video_path: path to video file.
        target_height: spatial target height.
        target_width: spatial target width.
        target_num_frames: target number of pixel frames (e.g. 73 for src, 33 for tgt).
        keep_tail: if True, offset the start so that the last frame is always kept
                   (drops head frames instead of tail). Use for src videos where
                   tail frames are important for history.

    Returns:
        video: tensor of shape (C, T, H, W) in float32, range [-1, 1].
        orig_num_frames: original frame count before any processing.
    """
    video, audio, info = torchvision.io.read_video(video_path, pts_unit="sec")
    # video shape: (T, H, W, C), uint8

    orig_num_frames = video.shape[0]

    # Ratio-based temporal subsampling
    step = max(1, orig_num_frames // target_num_frames)
    if step > 1:
        if keep_tail:
            # Offset so the last frame is always selected, drop head remainder
            offset = (orig_num_frames - 1) % step
            video = video[offset::step]
        else:
            video = video[::step]

    # Ensure frame count is 4n+1 for VAE compatibility (temporal stride = 4)
    T = video.shape[0]
    valid_T = ((T - 1) // 4) * 4 + 1
    if valid_T < 1:
        valid_T = 1
    if keep_tail:
        video = video[-valid_T:]   # trim head, preserve tail
    else:
        video = video[:valid_T]    # trim tail, preserve head

    # (T, H, W, C) -> (T, C, H, W)
    video = video.permute(0, 3, 1, 2).float()

    # Resize spatially
    video = TF.resize(video, [target_height, target_width], antialias=True)

    # Normalize to [-1, 1]
    video = video / 127.5 - 1.0

    # (T, C, H, W) -> (C, T, H, W) for VAE
    video = video.permute(1, 0, 2, 3)
    return video, orig_num_frames


def vae_encode_video(vae, pixel_values, latents_mean, latents_std, device):
    """Encode a single video (C, T, H, W) to latent (C_lat, T_lat, H_lat, W_lat).

    For long videos, encode in 33-frame chunks to avoid OOM.
    """
    pixel_values = pixel_values.unsqueeze(0).to(dtype=vae.dtype, device=device)  # (1, C, T, H, W)

    T = pixel_values.shape[2]
    latent_window_size = 9
    frame_window_size = (latent_window_size - 1) * 4 + 1  # = 33

    if T <= frame_window_size:
        # Single-pass encoding
        with torch.no_grad():
            latent = vae.encode(pixel_values).latent_dist.sample()
            latent = (latent - latents_mean) * latents_std
        return latent.squeeze(0)  # (C_lat, T_lat, H_lat, W_lat)
    else:
        # Chunked encoding
        num_chunks = T // frame_window_size
        chunk_latents = []
        for i in range(num_chunks):
            start = i * frame_window_size
            end = start + frame_window_size
            chunk = pixel_values[:, :, start:end, :, :]
            with torch.no_grad():
                lat = vae.encode(chunk).latent_dist.sample()
                lat = (lat - latents_mean) * latents_std
            chunk_latents.append(lat.squeeze(0))

        # Handle remaining frames (if any meaningful frames left)
        remaining_start = num_chunks * frame_window_size
        remaining_frames = T - remaining_start
        if remaining_frames >= 1:
            # Pad to valid length (4n+1)
            valid_remaining = ((remaining_frames - 1) // 4) * 4 + 1
            if valid_remaining >= 1:
                chunk = pixel_values[:, :, remaining_start:remaining_start + valid_remaining, :, :]
                with torch.no_grad():
                    lat = vae.encode(chunk).latent_dist.sample()
                    lat = (lat - latents_mean) * latents_std
                chunk_latents.append(lat.squeeze(0))

        # Concatenate along temporal dimension
        return torch.cat(chunk_latents, dim=1)  # (C_lat, total_T_lat, H_lat, W_lat)


def main():
    args = parse_args()

    # Distributed env
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    is_main = rank == 0

    if world_size > 1:
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])

    device = torch.device(f"cuda:{local_rank}")
    weight_dtype = torch.bfloat16

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.csv), "encoded_edit_latents")
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    # Load models
    logging.info(f"Loading models from {args.pretrained_model_name_or_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer")

    text_encoder = UMT5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        torch_dtype=weight_dtype)
    text_encoder.eval().requires_grad_(False).to(device)

    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        torch_dtype=torch.float32)
    vae.eval().requires_grad_(False).to(device)

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)

    logging.info("Models loaded.")
    logging.info(f"Ratio-based subsampling: src→{args.src_target_frames} frames, tgt→{args.tgt_target_frames} frames")

    # Load CSV
    rows = load_csv_rows(args.csv)
    logging.info(f"Loaded {len(rows)} rows from CSV")

    # Filter valid rows
    valid_rows = []
    for row in rows:
        src = row.get("src_video_path", "")
        tgt = row.get("tgt_video_path", "")
        if src and tgt and os.path.exists(src) and os.path.exists(tgt):
            valid_rows.append(row)
        elif is_main:
            clip_id = row.get("clip_id", "unknown")
            if not os.path.exists(src):
                logging.warning(f"src not found for {clip_id}: {src}")
            if not os.path.exists(tgt):
                logging.warning(f"tgt not found for {clip_id}: {tgt}")
    logging.info(f"Valid rows (both videos exist): {len(valid_rows)}")

    # Skip already encoded
    pending_rows = []
    for row in valid_rows:
        clip_id = row.get("clip_id", "")
        if not clip_id:
            clip_id = os.path.splitext(os.path.basename(row["src_video_path"]))[0]
        # Remove .mp4 extension if present in clip_id
        clip_id = clip_id.replace(".mp4", "").replace(".MP4", "")
        row["_clip_id"] = clip_id
        out_path = os.path.join(args.output_dir, f"{clip_id}.pt")
        if not os.path.exists(out_path):
            pending_rows.append(row)
    logging.info(f"Pending (not yet encoded): {len(pending_rows)}")

    # Limit number of samples to encode
    if args.max_samples is not None and args.max_samples > 0:
        pending_rows = pending_rows[:args.max_samples]
        logging.info(f"Capped to --max_samples={args.max_samples}, will encode {len(pending_rows)} samples")

    # Shard across GPUs
    if world_size > 1:
        pending_rows = [r for i, r in enumerate(pending_rows) if i % world_size == rank]
        logging.info(f"[Rank {rank}] Assigned {len(pending_rows)} samples")

    # Process
    pbar = tqdm(pending_rows, desc=f"[Rank {rank}] Encoding", disable=not is_main)
    success_count = 0
    fail_count = 0

    for row in pbar:
        clip_id = row["_clip_id"]
        out_path = os.path.join(args.output_dir, f"{clip_id}.pt")

        if os.path.exists(out_path):
            continue

        try:
            # Load and resize videos (ratio-based subsampling)
            # src: keep_tail=True → preserve tail frames (used as history)
            # tgt: keep_tail=False → preserve head frames (used as GT)
            src_video, src_orig_frames = load_and_resize_video(
                row["src_video_path"], args.target_height, args.target_width,
                target_num_frames=args.src_target_frames, keep_tail=True)
            tgt_video, tgt_orig_frames = load_and_resize_video(
                row["tgt_video_path"], args.target_height, args.target_width,
                target_num_frames=args.tgt_target_frames, keep_tail=False)

            src_pixel_frames = src_video.shape[1]  # after subsample + 4n+1
            tgt_pixel_frames = tgt_video.shape[1]

            # VAE encode
            src_latent = vae_encode_video(vae, src_video, latents_mean, latents_std, device)
            tgt_latent = vae_encode_video(vae, tgt_video, latents_mean, latents_std, device)

            # Text encode
            scene_caption = row.get("src_video_caption", "")
            edit_instruction = row.get("edit_video_instruction", "")

            with torch.no_grad():
                scene_embed, scene_mask = encode_prompt(
                    tokenizer, text_encoder, scene_caption,
                    max_sequence_length=args.max_sequence_length,
                    device=device, dtype=weight_dtype)
                edit_embed, edit_mask = encode_prompt(
                    tokenizer, text_encoder, edit_instruction,
                    max_sequence_length=args.max_sequence_length,
                    device=device, dtype=weight_dtype)

            # Save
            data = {
                "src_vae_latent": src_latent.cpu().to(torch.float16),
                "tgt_vae_latent": tgt_latent.cpu().to(torch.float16),
                "scene_prompt_embed": scene_embed.squeeze(0).cpu(),
                "scene_prompt_attention_mask": scene_mask.squeeze(0).cpu(),
                "edit_prompt_embed": edit_embed.squeeze(0).cpu(),
                "edit_prompt_attention_mask": edit_mask.squeeze(0).cpu(),
                "scene_prompt_raw": scene_caption,
                "edit_prompt_raw": edit_instruction,
                "src_latent_T": src_latent.shape[1],
                "tgt_latent_T": tgt_latent.shape[1],
                "src_orig_frames": src_orig_frames,
                "tgt_orig_frames": tgt_orig_frames,
            }
            torch.save(data, out_path)

            success_count += 1
            pbar.set_postfix({
                "clip": clip_id,
                "src": f"{src_orig_frames}→{src_pixel_frames}px→{src_latent.shape[1]}lat",
                "tgt": f"{tgt_orig_frames}→{tgt_pixel_frames}px→{tgt_latent.shape[1]}lat",
            })
            logging.info(
                f"[{clip_id}] src: "
                f"frames={src_orig_frames}→step{max(1, src_orig_frames // args.src_target_frames)}"
                f"→{src_pixel_frames}px→{src_latent.shape[1]}lat | "
                f"tgt: "
                f"frames={tgt_orig_frames}→step{max(1, tgt_orig_frames // args.tgt_target_frames)}"
                f"→{tgt_pixel_frames}px→{tgt_latent.shape[1]}lat"
            )

        except Exception as e:
            fail_count += 1
            logging.warning(f"Failed to encode {clip_id}: {e}")
            continue

    logging.info(f"[Rank {rank}] Done. success={success_count}, failed={fail_count}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Encoding complete!")


if __name__ == "__main__":
    main()
