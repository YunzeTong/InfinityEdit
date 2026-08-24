"""
Edit Adapter module for Helios-Distilled.

Three-module pipeline per adapter block:
  1. History cross-attention: first N frames of the current chunk attend to
     history hidden states, injecting temporal continuity information.
  2. Causal temporal self-attention: tokens at the same (h, w) location attend
     across frames with a causal mask (frame t sees frames 0..t only).
  3. Edit instruction cross-attention: all current tokens attend to edit
     instruction embeddings (unchanged from original design).

All adapter output projections (to_out) are zero-initialized so each module
starts as identity. Sigma-aware AdaLN modulation (shared SigmaEmbedding) is
applied to the outputs of history cross-attn and edit cross-attn.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmaEmbedding(nn.Module):
    """Sinusoidal embedding for sigma/timestep -> scale+shift modulation.

    Zero-initialized output so the modulation starts as identity (scale=1, shift=0).
    """

    def __init__(self, freq_dim: int = 256, out_dim: int = 10240):
        super().__init__()
        half = freq_dim // 2
        freqs = torch.exp(torch.linspace(
            math.log(1.0), math.log(10000.0), half,
        ))
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, freq_dim),
            nn.SiLU(),
            nn.Linear(freq_dim, out_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.float().unsqueeze(-1)  # (B, 1)
        embed = torch.cat([
            (t * self.freqs).sin(),
            (t * self.freqs).cos(),
        ], dim=-1)  # (B, freq_dim)
        embed = embed.to(self.mlp[0].weight.dtype)
        return self.mlp(embed)  # (B, out_dim)


class HistoryCrossAttention(nn.Module):
    """Cross-attention: Q from first N frames of current chunk, K/V from history."""

    def __init__(
        self,
        dim: int = 5120,
        adapter_dim: int = 512,
        num_heads: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = adapter_dim // num_heads
        assert adapter_dim % num_heads == 0

        self.norm_q_input = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm_k_input = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, adapter_dim, bias=True)
        self.to_k = nn.Linear(dim, adapter_dim, bias=True)
        self.to_v = nn.Linear(dim, adapter_dim, bias=True)
        self.to_out = nn.Linear(adapter_dim, dim, bias=True)

        self.norm_q = nn.RMSNorm(adapter_dim, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(adapter_dim, eps=eps, elementwise_affine=True)

        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(
        self,
        first_frames: torch.Tensor,     # (B, first_N_tokens, dim)
        history_hidden: torch.Tensor,    # (B, history_seq_len, dim)
    ) -> torch.Tensor:
        residual_dtype = first_frames.dtype
        normed_q = self.norm_q_input(first_frames.float()).to(residual_dtype)
        normed_k = self.norm_k_input(history_hidden.float()).to(residual_dtype)

        q = self.norm_q(self.to_q(normed_q))
        k = self.norm_k(self.to_k(normed_k))
        v = self.to_v(normed_k)

        B, Sq, _ = q.shape
        _, Sk, _ = k.shape

        q = q.view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, Sq, -1)

        return self.to_out(attn_out)


class EditAdapterCrossAttention(nn.Module):
    """Cross-attention layer: Q from hidden states, K/V from edit instruction."""

    def __init__(
        self,
        dim: int = 5120,
        adapter_dim: int = 1024,
        text_dim: int = 4096,
        num_heads: int = 8,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = adapter_dim // num_heads
        assert adapter_dim % num_heads == 0

        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, adapter_dim, bias=True)
        self.to_k = nn.Linear(text_dim, adapter_dim, bias=True)
        self.to_v = nn.Linear(text_dim, adapter_dim, bias=True)
        self.to_out = nn.Linear(adapter_dim, dim, bias=True)

        self.norm_q = nn.RMSNorm(adapter_dim, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(adapter_dim, eps=eps, elementwise_affine=True)

        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,       # (B, seq_len, dim)
        edit_embeds: torch.Tensor,          # (B, text_seq_len, text_dim)
    ) -> torch.Tensor:
        residual_dtype = hidden_states.dtype
        normed = self.norm(hidden_states.float()).to(residual_dtype)

        q = self.norm_q(self.to_q(normed))
        k = self.norm_k(self.to_k(edit_embeds))
        v = self.to_v(edit_embeds)

        B, Sq, _ = q.shape
        _, Sk, _ = k.shape

        q = q.view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, Sq, -1)

        return self.to_out(attn_out)


class TemporalSelfAttention(nn.Module):
    """Causal temporal-axial self-attention for current chunk tokens.

    Reshapes (B, T*H*W, C) -> (B*H*W, T, C) so each spatial location attends
    across frames within the chunk with a causal mask (frame t sees 0..t).
    Uses internal 1D temporal RoPE; output projection is zero-initialized.
    """

    def __init__(
        self,
        dim: int = 5120,
        adapter_dim: int = 1024,
        num_heads: int = 8,
        eps: float = 1e-6,
        max_temporal_len: int = 64,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = adapter_dim // num_heads
        assert adapter_dim % num_heads == 0, "adapter_dim must be divisible by num_heads"
        assert self.head_dim % 2 == 0, "head_dim must be even for 1D RoPE"

        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, adapter_dim, bias=True)
        self.to_k = nn.Linear(dim, adapter_dim, bias=True)
        self.to_v = nn.Linear(dim, adapter_dim, bias=True)
        self.to_out = nn.Linear(adapter_dim, dim, bias=True)

        self.norm_q = nn.RMSNorm(adapter_dim, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(adapter_dim, eps=eps, elementwise_affine=True)

        half = self.head_dim // 2
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half)
        )
        t = torch.arange(max_temporal_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)
        self.max_temporal_len = max_temporal_len

        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def _apply_rope(self, x: torch.Tensor, T: int) -> torch.Tensor:
        if T > self.max_temporal_len:
            raise ValueError(
                f"Temporal self-attn got T={T}, exceeds max_temporal_len="
                f"{self.max_temporal_len}. Increase edit_adapter_temporal_max_len."
            )
        cos = self.rope_cos[:T].to(x.dtype)
        sin = self.rope_sin[:T].to(x.dtype)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        x1, x2 = x.chunk(2, dim=-1)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,   # (B, T*H*W, dim)
        T: int, H: int, W: int,
    ) -> torch.Tensor:
        B, S, C = hidden_states.shape
        assert S == T * H * W, (
            f"TemporalSelfAttention got seq_len={S} but T*H*W={T * H * W}"
        )

        residual_dtype = hidden_states.dtype
        normed = self.norm(hidden_states.float()).to(residual_dtype)

        q = self.norm_q(self.to_q(normed))
        k = self.norm_k(self.to_k(normed))
        v = self.to_v(normed)

        def to_axial(x):
            x = x.view(B, T, H * W, self.num_heads, self.head_dim)
            x = x.permute(0, 2, 3, 1, 4).contiguous()  # (B, H*W, heads, T, hd)
            return x.view(B * H * W, self.num_heads, T, self.head_dim)

        q = to_axial(q)
        k = to_axial(k)
        v = to_axial(v)

        q = self._apply_rope(q, T)
        k = self._apply_rope(k, T)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.view(B, H * W, self.num_heads, T, self.head_dim)
        attn_out = attn_out.permute(0, 3, 1, 2, 4).contiguous()
        attn_out = attn_out.view(B, T * H * W, -1)

        return self.to_out(attn_out)


class EditAdapterBlock(nn.Module):
    """One adapter block: history cross-attn -> causal temporal SA -> edit cross-attn.

    Splits hidden_states into history/current. History cross-attn injects history
    context into the first N frames. Causal temporal SA propagates information
    forward in time. Edit cross-attn injects edit instruction into all current
    tokens. History tokens are left untouched throughout.
    """

    def __init__(
        self,
        dim: int = 5120,
        adapter_dim: int = 1024,
        text_dim: int = 4096,
        num_heads: int = 8,
        eps: float = 1e-6,
        enable_temporal_self_attn: bool = False,
        temporal_max_len: int = 64,
        history_adapter_dim: int | None = None,
        history_num_heads: int = 4,
        history_num_frames: int = 2,
    ):
        super().__init__()
        self.enable_temporal_self_attn = enable_temporal_self_attn
        self.history_num_frames = history_num_frames

        self.history_cross_attn = None
        if history_adapter_dim is not None:
            self.history_cross_attn = HistoryCrossAttention(
                dim=dim,
                adapter_dim=history_adapter_dim,
                num_heads=history_num_heads,
                eps=eps,
            )

        if enable_temporal_self_attn:
            self.temporal_self_attn = TemporalSelfAttention(
                dim=dim,
                adapter_dim=adapter_dim,
                num_heads=num_heads,
                eps=eps,
                max_temporal_len=temporal_max_len,
            )

        self.cross_attn = EditAdapterCrossAttention(
            dim=dim,
            adapter_dim=adapter_dim,
            text_dim=text_dim,
            num_heads=num_heads,
            eps=eps,
        )

    @staticmethod
    def _apply_sigma_modulation(
        adapter_out: torch.Tensor,
        sigma_modulation: torch.Tensor | None,
    ) -> torch.Tensor:
        if sigma_modulation is None:
            return adapter_out
        scale, shift = sigma_modulation.chunk(2, dim=-1)  # each (B, dim)
        return adapter_out * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _apply_history_cross_attn(
        self,
        current_hidden: torch.Tensor,
        history_hidden: torch.Tensor,
        T: int, H: int, W: int,
        sigma_modulation: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply history cross-attn to the first N frames of current_hidden."""
        if self.history_cross_attn is None:
            return current_hidden
        n_frames = min(self.history_num_frames, T)
        first_n_tokens = n_frames * H * W

        first_frames = current_hidden[:, :first_n_tokens]
        hist_out = self.history_cross_attn(first_frames, history_hidden)
        hist_out = self._apply_sigma_modulation(hist_out, sigma_modulation)

        if first_n_tokens < current_hidden.shape[1]:
            current_hidden = torch.cat([
                first_frames + hist_out,
                current_hidden[:, first_n_tokens:],
            ], dim=1)
        else:
            current_hidden = first_frames + hist_out

        return current_hidden

    def forward(
        self,
        hidden_states: torch.Tensor,            # (B, total_seq, dim)
        edit_embeds: torch.Tensor,               # (B, text_seq, text_dim)
        original_context_length: int,
        original_context_length_list: list,
        post_patch_num_frames_list: list = None,
        post_patch_height_list: list = None,
        post_patch_width_list: list = None,
        sigma_modulation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        history_seq_len = (
            hidden_states.shape[1] - original_context_length
        ) // len(original_context_length_list)

        enable_navit = len(original_context_length_list) > 1

        assert (
            post_patch_num_frames_list is not None
            and post_patch_height_list is not None
            and post_patch_width_list is not None
        ), "post_patch shape lists must be provided to EditAdapterBlock.forward()."

        if enable_navit:
            num_seqs = len(original_context_length_list)
            hidden_states_list = [None] * num_seqs
            history_hidden_states_list = [None] * num_seqs

            seq_start = 0
            for idx, cur_seq_len in enumerate(original_context_length_list[::-1]):
                seq_end = seq_start + cur_seq_len + history_seq_len
                cur = hidden_states[:, seq_start:seq_end, :]
                history_hidden_states_list[idx] = cur[:, :history_seq_len]
                hidden_states_list[idx] = cur[:, history_seq_len:]
                seq_start += cur_seq_len + history_seq_len

            rev_T = list(post_patch_num_frames_list)[::-1]
            rev_H = list(post_patch_height_list)[::-1]
            rev_W = list(post_patch_width_list)[::-1]

            # Stage 1: History cross-attn (per-subseq)
            for idx in range(num_seqs):
                T_i, H_i, W_i = rev_T[idx], rev_H[idx], rev_W[idx]
                hidden_states_list[idx] = self._apply_history_cross_attn(
                    hidden_states_list[idx],
                    history_hidden_states_list[idx],
                    T_i, H_i, W_i,
                    sigma_modulation,
                )

            # Stage 2: Causal temporal SA (per-subseq)
            if self.enable_temporal_self_attn:
                for idx in range(num_seqs):
                    T_i, H_i, W_i = rev_T[idx], rev_H[idx], rev_W[idx]
                    sa_out = self.temporal_self_attn(
                        hidden_states_list[idx], T_i, H_i, W_i,
                    )
                    hidden_states_list[idx] = hidden_states_list[idx] + sa_out

            # Stage 3: Edit cross-attn on concatenated current
            current_hidden = torch.cat(hidden_states_list, dim=1)
            adapter_out = self.cross_attn(current_hidden, edit_embeds)
            adapter_out = self._apply_sigma_modulation(adapter_out, sigma_modulation)
            current_hidden = current_hidden + adapter_out

            # Re-interleave history and current
            result_parts = []
            cur_start = 0
            for idx, cur_seq_len in enumerate(original_context_length_list[::-1]):
                result_parts.append(history_hidden_states_list[idx])
                result_parts.append(current_hidden[:, cur_start:cur_start + cur_seq_len])
                cur_start += cur_seq_len
            hidden_states = torch.cat(result_parts, dim=1)
        else:
            # Standard mode: single resolution
            history_hidden = hidden_states[:, :history_seq_len]
            current_hidden = hidden_states[:, history_seq_len:]

            T = post_patch_num_frames_list[0]
            H = post_patch_height_list[0]
            W = post_patch_width_list[0]

            # Stage 1: History cross-attn on first N frames
            current_hidden = self._apply_history_cross_attn(
                current_hidden, history_hidden, T, H, W, sigma_modulation,
            )

            # Stage 2: Causal temporal SA
            if self.enable_temporal_self_attn:
                sa_out = self.temporal_self_attn(current_hidden, T, H, W)
                current_hidden = current_hidden + sa_out

            # Stage 3: Edit cross-attn
            adapter_out = self.cross_attn(current_hidden, edit_embeds)
            adapter_out = self._apply_sigma_modulation(adapter_out, sigma_modulation)
            current_hidden = current_hidden + adapter_out

            hidden_states = torch.cat([history_hidden, current_hidden], dim=1)

        return hidden_states


class EditAdapterCollection(nn.Module):
    """Collection of adapter blocks, one per transformer layer."""

    def __init__(
        self,
        num_layers: int = 40,
        dim: int = 5120,
        adapter_dim: int = 1024,
        text_dim: int = 4096,
        num_heads: int = 8,
        eps: float = 1e-6,
        enable_temporal_self_attn: bool = False,
        temporal_max_len: int = 64,
        sigma_freq_dim: int = 256,
        history_adapter_dim: int | None = None,
        history_num_heads: int = 4,
        history_num_frames: int = 2,
    ):
        super().__init__()
        self.enable_temporal_self_attn = enable_temporal_self_attn
        self.sigma_embed = None
        if history_adapter_dim is not None:
            self.sigma_embed = SigmaEmbedding(
                freq_dim=sigma_freq_dim,
                out_dim=dim * 2,
            )
        self.adapter_blocks = nn.ModuleList([
            EditAdapterBlock(
                dim=dim,
                adapter_dim=adapter_dim,
                text_dim=text_dim,
                num_heads=num_heads,
                eps=eps,
                enable_temporal_self_attn=enable_temporal_self_attn,
                temporal_max_len=temporal_max_len,
                history_adapter_dim=history_adapter_dim,
                history_num_heads=history_num_heads,
                history_num_frames=history_num_frames,
            )
            for _ in range(num_layers)
        ])

    def compute_sigma_modulation(
        self, timestep: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Compute sigma modulation once, shared across all layers."""
        if timestep is None or self.sigma_embed is None:
            return None
        return self.sigma_embed(timestep)  # (B, dim*2)

    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        edit_embeds: torch.Tensor,
        original_context_length: int,
        original_context_length_list: list,
        post_patch_num_frames_list: list = None,
        post_patch_height_list: list = None,
        post_patch_width_list: list = None,
        sigma_modulation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.adapter_blocks[layer_idx](
            hidden_states, edit_embeds,
            original_context_length, original_context_length_list,
            post_patch_num_frames_list,
            post_patch_height_list,
            post_patch_width_list,
            sigma_modulation=sigma_modulation,
        )

    def get_to_out_norms(self) -> list[float]:
        """Return cross_attn to_out weight norms for monitoring adapter growth."""
        return [
            block.cross_attn.to_out.weight.norm().item()
            for block in self.adapter_blocks
        ]

    def get_temporal_to_out_norms(self) -> list[float]:
        """Return temporal_self_attn to_out weight norms (if enabled, else [])."""
        if not self.enable_temporal_self_attn:
            return []
        return [
            block.temporal_self_attn.to_out.weight.norm().item()
            for block in self.adapter_blocks
        ]

    def get_history_to_out_norms(self) -> list[float]:
        """Return history_cross_attn to_out weight norms for monitoring."""
        if self.adapter_blocks[0].history_cross_attn is None:
            return []
        return [
            block.history_cross_attn.to_out.weight.norm().item()
            for block in self.adapter_blocks
        ]
