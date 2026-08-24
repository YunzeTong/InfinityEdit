"""
Dataset for Edit Adapter training.

Loads pre-encoded .pt files (from encode_edit_dataset.py).

Supports two data modes controlled by ``data_mode``:

  "splice" (default) — src and tgt are long videos; history is spliced from both:
    .pt contains:
      src_vae_latent:  (C, T_src, H, W)
      tgt_vae_latent:  (C, T_tgt, H, W)
    history = source[-10:] + target[:9]   →  19 latent frames
    GT      = target[9:18]                →  9 latent frames
    x0      = history first frame

  "separated" — src tail is history, tgt head is ground-truth:
    .pt contains:
      src_vae_latent:  (C, T_src, H, W)  T_src >= 19, uses src[-19:]
      tgt_vae_latent:  (C, T_tgt, H, W)  T_tgt >= 9,  uses tgt[:9]
    history = src[-19:]
    GT      = tgt[:9]
    x0 = src first frame
"""

import os
import pickle
import time
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm


class EditAdapterDataset(Dataset):
    def __init__(
        self,
        feature_folder,
        history_sizes=[16, 2, 1],
        latent_window_size=9,
        seed=42,
        data_mode="splice",
        filter_filelist=None,
    ):
        assert data_mode in ("splice", "separated"), (
            f"Unknown data_mode={data_mode!r}, expected 'splice' or 'separated'"
        )
        self.data_mode = data_mode
        self.history_sizes = sorted(history_sizes, reverse=True)  # [16, 2, 1]
        self.history_window_size = sum(self.history_sizes)         # 19
        self.latent_window_size = latent_window_size               # 9
        self.base_seed = seed
        self._epoch = 0

        if isinstance(feature_folder, str):
            raw_folders = [feature_folder]
        else:
            raw_folders = list(feature_folder)

        # Expand: if a folder has no .pt files but has subdirs containing .pt,
        # recursively collect all subdirs that contain .pt files.
        self.feature_folders = []
        for folder in raw_folders:
            self.feature_folders.extend(self._expand_folder(folder))

        self.samples = []
        self.buckets = defaultdict(list)

        t_start = time.time()
        for i, folder in enumerate(self.feature_folders):
            print(f"[{i+1}/{len(self.feature_folders)}] Processing folder: {os.path.basename(folder)}")
            self._process_folder(folder)

        if filter_filelist is not None:
            self._apply_filelist_filter(filter_filelist)

        print(f"Dataset ready: {len(self.samples)} total samples, "
              f"{len(self.buckets)} buckets, loaded in {time.time()-t_start:.1f}s")

    @staticmethod
    def _expand_folder(folder):
        """If folder directly contains .pt files, return [folder].
        Otherwise, recursively find all subdirs that contain .pt files."""
        if not os.path.isdir(folder):
            print(f"  [WARN] Not a directory, skipping: {folder}")
            return []

        has_pt = any(f.endswith(".pt") for f in os.listdir(folder))
        if has_pt:
            return [folder]

        # Recurse into subdirs
        expanded = []
        for root, dirs, files in os.walk(folder):
            if any(f.endswith(".pt") for f in files):
                expanded.append(root)

        if expanded:
            print(f"  Expanded {folder} -> {len(expanded)} sub-folders with .pt files")
        else:
            print(f"  [WARN] No .pt files found under: {folder}")
        return sorted(expanded)

    def _apply_filelist_filter(self, filelist_path):
        """Filter samples to only keep those whose file_path appears in the filelist."""
        with open(filelist_path, "r") as f:
            allowed = set(line.strip() for line in f if line.strip())

        pre_filter = len(self.samples)
        keep_indices = []
        for i, s in enumerate(self.samples):
            if s["file_path"] in allowed:
                keep_indices.append(i)

        old_to_new = {old: new for new, old in enumerate(keep_indices)}
        self.samples = [self.samples[i] for i in keep_indices]

        new_buckets = defaultdict(list)
        for bucket_key, indices in self.buckets.items():
            for idx in indices:
                if idx in old_to_new:
                    new_buckets[bucket_key].append(old_to_new[idx])
        self.buckets = new_buckets

        print(f"  Filter filelist applied: {pre_filter} -> {len(self.samples)} samples "
              f"(kept {len(self.samples)}/{len(allowed)} from filelist)")

    def _process_folder(self, folder):
        cache_file = os.path.join(folder, "edit_adapter_cache.pkl")

        t0 = time.time()
        if os.path.exists(cache_file):
            print(f"Loading cached metadata from: {folder}")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            folder_samples = cached_data["samples"]
            folder_buckets = cached_data["buckets"]
            print(f"Loaded {len(folder_samples)} samples from cache in {time.time()-t0:.1f}s: {folder}\n")
        else:
            print(f"Building metadata cache for folder: {folder}")
            folder_samples, folder_buckets = self._build_folder_metadata(folder)
            cached_data = {"samples": folder_samples, "buckets": folder_buckets}
            with open(cache_file, "wb") as f:
                pickle.dump(cached_data, f)
            print(f"Cached {len(folder_samples)} samples in {time.time()-t0:.1f}s from {folder}\n")

        sample_idx_offset = len(self.samples)
        self.samples.extend(folder_samples)

        for bucket_key, indices in folder_buckets.items():
            adjusted_indices = [idx + sample_idx_offset for idx in indices]
            self.buckets[bucket_key].extend(adjusted_indices)

    def _build_folder_metadata(self, folder):
        feature_files = [f for f in os.listdir(folder) if f.endswith(".pt")]
        samples = []
        buckets = defaultdict(list)
        sample_idx = 0

        for feature_file in tqdm(
            sorted(feature_files),
            desc=f"Scanning {os.path.basename(folder)}",
            unit="file",
        ):
            feature_path = os.path.join(folder, feature_file)

            # Quick-load to check shapes
            try:
                data = torch.load(feature_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"  Skipping {feature_file}: load error: {e}")
                continue

            src_lat = data.get("src_vae_latent")
            tgt_lat = data.get("tgt_vae_latent")
            if src_lat is None or tgt_lat is None:
                print(f"  Skipping {feature_file}: missing latent keys")
                continue

            src_T = src_lat.shape[1]  # (C, T, H, W)
            tgt_T = tgt_lat.shape[1]
            H = src_lat.shape[2]
            W = src_lat.shape[3]

            if self.data_mode == "splice":
                # source: 10 frames for history → need src_T >= 10
                # target: 9 frames for history + 9 frames for GT → need tgt_T >= 18
                if src_T < 10:
                    print(f"  Skipping {feature_file}: src_T={src_T} < 10")
                    continue
                if tgt_T < 18:
                    print(f"  Skipping {feature_file}: tgt_T={tgt_T} < 18")
                    continue
            else:  # separated
                if src_T < self.history_window_size:
                    print(f"  Skipping {feature_file}: src_T={src_T} < history_window_size={self.history_window_size}")
                    continue
                if tgt_T < self.latent_window_size:
                    print(f"  Skipping {feature_file}: tgt_T={tgt_T} < latent_window_size={self.latent_window_size}")
                    continue

            clip_id = os.path.splitext(feature_file)[0]
            bucket_key = (H, W)

            sample_info = {
                "clip_id": clip_id,
                "dataset_name": folder.rstrip("/"),
                "file_path": feature_path,
                "bucket_key": bucket_key,
                "height": H,
                "width": W,
                "src_T": src_T,
                "tgt_T": tgt_T,
            }

            samples.append(sample_info)
            buckets[bucket_key].append(sample_idx)
            sample_idx += 1

        return samples, buckets

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        feature_data = torch.load(sample_info["file_path"], map_location="cpu", weights_only=False)

        src_lat = feature_data["src_vae_latent"]  # (C, T_src, H, W), float16
        tgt_lat = feature_data["tgt_vae_latent"]  # (C, T_tgt, H, W), float16

        if self.data_mode == "splice":
            history_latents, target_latents, x0_latent = self._split_splice(src_lat, tgt_lat)
        else:
            history_latents, target_latents, x0_latent = self._split_separated(src_lat, tgt_lat)

        # Prompt embeddings
        scene_prompt_embed = feature_data["scene_prompt_embed"]               # (512, 4096)
        scene_prompt_attention_mask = feature_data["scene_prompt_attention_mask"]  # (512,)
        edit_prompt_embed = feature_data["edit_prompt_embed"]                  # (512, 4096)
        edit_prompt_attention_mask = feature_data["edit_prompt_attention_mask"]    # (512,)

        return {
            "clip_id": sample_info["clip_id"],
            "bucket_key": sample_info["bucket_key"],
            "dataset_name": sample_info["dataset_name"],
            "height": sample_info["height"],
            "width": sample_info["width"],
            "x0_latents": x0_latent,
            "history_latents": history_latents,
            "target_latents": target_latents,
            "prompt_embeds": scene_prompt_embed,
            "prompt_attention_masks": scene_prompt_attention_mask,
            "edit_prompt_embeds": edit_prompt_embed,
            "edit_prompt_attention_masks": edit_prompt_attention_mask,
        }

    def _split_splice(self, src_lat, tgt_lat):
        """Splice mode: extract history from tail of src + head of tgt."""
        src_T = src_lat.shape[1]

        src_history_len = self.history_window_size - self.latent_window_size  # 19-9=10
        tgt_history_len = self.latent_window_size                             # 9

        src_history = src_lat[:, src_T - src_history_len :, :, :]  # (C, 10, H, W)
        tgt_history = tgt_lat[:, :tgt_history_len, :, :]           # (C, 9, H, W)

        history_latents = torch.cat([src_history, tgt_history], dim=1)  # (C, 19, H, W)
        target_latents = tgt_lat[:, tgt_history_len : tgt_history_len + self.latent_window_size, :, :]  # (C, 9, H, W)
        x0_latent = src_history[:, :1, :, :]  # (C, 1, H, W)

        return history_latents, target_latents, x0_latent

    def _split_separated(self, src_lat, tgt_lat):
        """Separated mode: src is history, tgt is ground-truth, used as-is."""
        history_latents = src_lat[:, -self.history_window_size:, :, :]   # (C, 19, H, W)
        target_latents = tgt_lat[:, :self.latent_window_size, :, :]     # (C, 9, H, W)
        x0_latent = history_latents[:, :1, :, :]  # (C, 1, H, W) — first frame of history

        return history_latents, target_latents, x0_latent


class EditAdapterSampler(Sampler):
    """Bucketed sampler for edit adapter training.

    Groups samples by spatial resolution (H, W) to enable batching.
    Supports distributed training via SP group sharding.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        drop_last=True,
        shuffle=True,
        seed=42,
        num_sp_groups=1,
        sp_world_size=1,
        global_rank=0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.generator = torch.Generator()
        self.buckets = dataset.buckets
        self._epoch = 0

        self.num_sp_groups = num_sp_groups
        self.sp_world_size = sp_world_size
        self.global_rank = global_rank
        self.ith_sp_group = self.global_rank // self.sp_world_size

    def set_epoch(self, epoch):
        self._epoch = epoch

    def _shard_indices_for_sp_group(self, indices):
        if self.num_sp_groups == 1:
            return indices

        indices_tensor = torch.tensor(indices, dtype=torch.long) if isinstance(indices, list) else indices

        total_size = len(indices_tensor)
        if total_size % self.num_sp_groups != 0:
            if not self.drop_last:
                padding_size = self.num_sp_groups - (total_size % self.num_sp_groups)
                indices_tensor = torch.cat([indices_tensor, indices_tensor[:padding_size]])
            else:
                truncate_size = (total_size // self.num_sp_groups) * self.num_sp_groups
                indices_tensor = indices_tensor[:truncate_size]

        sp_group_indices = indices_tensor[self.ith_sp_group :: self.num_sp_groups]
        return sp_group_indices.tolist()

    def __iter__(self):
        epoch_seed = self.seed + self._epoch
        self.generator.manual_seed(epoch_seed)

        bucket_iterators = {}

        for bucket_key, sample_indices in self.buckets.items():
            indices = list(sample_indices)

            if self.shuffle:
                perm = torch.randperm(len(indices), generator=self.generator).tolist()
                indices = [indices[i] for i in perm]

            sp_indices = self._shard_indices_for_sp_group(indices)

            batches = []
            for i in range(0, len(sp_indices), self.batch_size):
                batch = sp_indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

            if batches:
                bucket_iterators[bucket_key] = iter(batches)

        remaining_buckets = list(bucket_iterators.keys())

        while remaining_buckets:
            idx = torch.randint(len(remaining_buckets), (1,), generator=self.generator).item()
            bucket_key = remaining_buckets[idx]

            try:
                batch = next(bucket_iterators[bucket_key])
                yield batch
            except StopIteration:
                remaining_buckets.remove(bucket_key)

    def __len__(self):
        total_batches = 0
        for bucket_key, sample_indices in self.buckets.items():
            sp_group_size = len(sample_indices) // self.num_sp_groups
            if not self.drop_last and len(sample_indices) % self.num_sp_groups != 0:
                sp_group_size += 1

            num_batches = sp_group_size // self.batch_size
            if not self.drop_last and sp_group_size % self.batch_size != 0:
                num_batches += 1
            total_batches += num_batches
        return total_batches


def edit_adapter_collate_fn(batch):
    return {
        key: torch.stack([d[key] for d in batch])
        if isinstance(batch[0][key], torch.Tensor)
        else [d[key] for d in batch]
        for key in batch[0]
    }
