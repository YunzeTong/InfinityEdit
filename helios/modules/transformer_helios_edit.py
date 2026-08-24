"""
HeliosTransformer3DModel with Edit Adapter.

Subclasses the base Helios transformer to inject cross-attention adapter layers
after each transformer block. Only the adapter parameters are trainable; the
base transformer remains frozen.

Usage:
    model = HeliosTransformer3DModelWithEditAdapter.from_pretrained(
        "BestWishYsh/Helios-Distilled",
        edit_adapter_config={...},
    )
    model.requires_grad_(False)
    model.edit_adapter.requires_grad_(True)
"""

from typing import Any, Dict, List, Optional, Union

import torch

from .edit_adapter import EditAdapterCollection
from .transformer_helios import HeliosTransformer3DModel


class HeliosTransformer3DModelWithEditAdapter(HeliosTransformer3DModel):
    """Helios transformer with injected Edit Adapter cross-attention layers."""

    def __init__(self, *args, **kwargs):
        # Pop edit_adapter_config before passing to parent
        edit_adapter_config = kwargs.pop("edit_adapter_config", None)
        super().__init__(*args, **kwargs)

        # Build adapter with provided config or defaults.
        cfg = edit_adapter_config or {}
        self.edit_adapter = EditAdapterCollection(
            num_layers=cfg.get("num_layers", len(self.blocks)),
            dim=cfg.get("dim", self.inner_dim),
            adapter_dim=cfg.get("adapter_dim", 1024),
            text_dim=cfg.get("text_dim", 4096),
            num_heads=cfg.get("num_heads", 8),
            eps=cfg.get("eps", 1e-6),
            enable_temporal_self_attn=cfg.get("enable_temporal_self_attn", False),
            temporal_max_len=cfg.get("temporal_max_len", 64),
            sigma_freq_dim=cfg.get("sigma_freq_dim", 256),
            history_adapter_dim=cfg.get("history_adapter_dim", None),
            history_num_heads=cfg.get("history_num_heads", 4),
            history_num_frames=cfg.get("history_num_frames", 2),
        )

    @classmethod
    def from_pretrained_with_edit_adapter(cls, pretrained_model_name_or_path, edit_adapter_config=None, **kwargs):
        """Load pretrained Helios weights and attach a fresh Edit Adapter.

        This works around the ConfigMixin/register_to_config issue by:
        1. Loading the base model as HeliosTransformer3DModel
        2. Creating our subclass instance with the same config
        3. Copying the base weights over
        """
        # Load base model
        base_model = HeliosTransformer3DModel.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        # Create subclass instance with the same config + edit adapter
        config = dict(base_model.config)
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)

        model = cls(**config, edit_adapter_config=edit_adapter_config)

        # Copy base weights (strict=False to skip edit_adapter keys)
        base_state_dict = base_model.state_dict()
        missing, unexpected = model.load_state_dict(base_state_dict, strict=False)

        # Only edit_adapter keys should be missing
        non_adapter_missing = [k for k in missing if not k.startswith("edit_adapter.")]
        if non_adapter_missing:
            raise RuntimeError(
                f"Missing non-adapter keys when loading base weights: {non_adapter_missing}"
            )

        del base_model
        return model

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        # ------------ Stage 1 ------------
        indices_hidden_states=None,
        indices_latents_history_short=None,
        indices_latents_history_mid=None,
        indices_latents_history_long=None,
        latents_history_short=None,
        latents_history_mid=None,
        latents_history_long=None,
        is_first_denoising_step: bool = False,
        # ------------ GAN ------------
        gan_mode: bool = False,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        # ------------ Edit Adapter ------------
        edit_instruction_embeds: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with edit adapter injection after each transformer block.

        All parameters are identical to HeliosTransformer3DModel.forward() except
        for the added ``edit_instruction_embeds`` which carries the UMT5-encoded
        edit instruction (B, text_seq, 4096).
        """
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        from diffusers.utils import apply_lora_scale

        # --- Replicated from HeliosTransformer3DModel.forward() ---
        # The full forward is copied (not called via super()) because we need to
        # inject adapter calls inside the block loop.  All code before and after
        # the block loop is identical to the parent.

        assert (
            len(
                {
                    x is None
                    for x in [
                        indices_hidden_states,
                        indices_latents_history_short,
                        indices_latents_history_mid,
                        indices_latents_history_long,
                        latents_history_short,
                        latents_history_mid,
                        latents_history_long,
                    ]
                }
            )
            == 1
        ), "All history latents and indices must either all exist or all be None"

        if indices_hidden_states is not None and indices_hidden_states.ndim == 1:
            indices_hidden_states = indices_hidden_states.unsqueeze(0)
        if indices_latents_history_short is not None and indices_latents_history_short.ndim == 1:
            indices_latents_history_short = indices_latents_history_short.unsqueeze(0)
        if indices_latents_history_mid is not None and indices_latents_history_mid.ndim == 1:
            indices_latents_history_mid = indices_latents_history_mid.unsqueeze(0)
        if indices_latents_history_long is not None and indices_latents_history_long.ndim == 1:
            indices_latents_history_long = indices_latents_history_long.unsqueeze(0)

        if gan_mode:
            assert self.is_use_gan

        if isinstance(hidden_states, list):
            assert gan_mode is False and self.is_use_gan is False
            enable_navit = True
            navit_len = len(hidden_states)
            batch_size = hidden_states[0].shape[0]
        else:
            enable_navit = False
            batch_size = hidden_states.shape[0]
        p_t, p_h, p_w = self.config.patch_size

        (
            hidden_states,
            rotary_emb,
            post_patch_height_list,
            post_patch_width_list,
            post_patch_num_frames_list,
            original_context_length_list,
        ) = self.process_input_hidden_states(
            latents=hidden_states,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
        )
        post_patch_num_frames = sum(post_patch_num_frames_list)
        post_patch_height = sum(post_patch_height_list)
        post_patch_width = sum(post_patch_width_list)
        original_context_length = sum(original_context_length_list)
        history_context_length = hidden_states.shape[1] - original_context_length

        if indices_hidden_states is not None and self.zero_history_timestep:
            if isinstance(timestep, list):
                timestep_t0 = torch.zeros((1), dtype=timestep[0].dtype, device=timestep[0].device)
            else:
                timestep_t0 = torch.zeros((1), dtype=timestep.dtype, device=timestep.device)
            temb_t0, timestep_proj_t0, _ = self.condition_embedder(
                timestep_t0, encoder_hidden_states, is_return_encoder_hidden_states=False
            )
            temb_t0 = temb_t0.unsqueeze(1).expand(batch_size, history_context_length, -1)
            timestep_proj_t0 = (
                timestep_proj_t0.unflatten(-1, (6, -1))
                .view(1, 6, 1, -1)
                .expand(batch_size, -1, history_context_length, -1)
            )

        navit_hidden_attention_mask = None
        navit_encoder_attention_mask = None
        if enable_navit:
            from .helios_kernels import create_navit_attention_masks

            assert navit_len == len(original_context_length_list)
            navit_hidden_attention_mask, navit_encoder_attention_mask, navit_history_hidden_attention_mask = (
                create_navit_attention_masks(
                    batch_size=batch_size,
                    original_context_length_list=original_context_length_list[::-1],
                    history_context_length=history_context_length,
                    encoder_hidden_states_seq_len=encoder_hidden_states.shape[1],
                    device=hidden_states.device,
                    restrict_self_attn=self.config.restrict_self_attn,
                    guidance_cross_attn=self.config.guidance_cross_attn,
                )
            )
            navit_hidden_attention_mask = [navit_hidden_attention_mask, navit_history_hidden_attention_mask]

            history_hidden_states, hidden_states = (
                hidden_states[:, :history_context_length],
                hidden_states[:, history_context_length:],
            )
            history_rotary_emb, rotary_emb = (
                rotary_emb[:, :history_context_length],
                rotary_emb[:, history_context_length:],
            )
            timestep = timestep[::-1]

            hidden_states_list = [None] * navit_len
            rotary_emb_list = [None] * navit_len
            temb_list = [None] * navit_len
            timestep_proj_list = [None] * navit_len

            seq_start = 0
            for idx, cur_seq_len in zip(range(navit_len), original_context_length_list[::-1]):
                cur_hidden_states = hidden_states[:, seq_start : seq_start + cur_seq_len, :]
                cur_rotary_emb = rotary_emb[:, seq_start : seq_start + cur_seq_len, :]

                hidden_states_list[idx] = torch.cat([history_hidden_states, cur_hidden_states], dim=1)
                rotary_emb_list[idx] = torch.cat([history_rotary_emb, cur_rotary_emb], dim=1)

                seq_start += cur_seq_len

                if idx == 0:
                    cur_temb, cur_timestep_proj, encoder_hidden_states = self.condition_embedder(
                        timestep[idx], encoder_hidden_states
                    )
                else:
                    cur_temb, cur_timestep_proj, _ = self.condition_embedder(
                        timestep[idx], encoder_hidden_states, is_return_encoder_hidden_states=False
                    )

                cur_temb = cur_temb.view(batch_size, 1, -1).expand(-1, cur_seq_len, -1)
                cur_timestep_proj = cur_timestep_proj.view(batch_size, 6, 1, -1).expand(-1, -1, cur_seq_len, -1)

                if self.zero_history_timestep:
                    temb_list[idx] = torch.cat([temb_t0, cur_temb], dim=1)
                    timestep_proj_list[idx] = torch.cat([timestep_proj_t0, cur_timestep_proj], dim=2)
                else:
                    temb_list[idx] = cur_temb
                    timestep_proj_list[idx] = cur_timestep_proj

            hidden_states = torch.cat(hidden_states_list, dim=1)
            rotary_emb = torch.cat(rotary_emb_list, dim=1)
            temb = torch.cat(temb_list, dim=1)
            timestep_proj = torch.cat(timestep_proj_list, dim=2)
        else:
            temb, timestep_proj, encoder_hidden_states = self.condition_embedder(timestep, encoder_hidden_states)
            timestep_proj = timestep_proj.unflatten(-1, (6, -1))

            if indices_hidden_states is not None and not self.zero_history_timestep:
                main_repeat_size = hidden_states.shape[1]
            else:
                main_repeat_size = original_context_length
            temb = temb.view(batch_size, 1, -1).expand(batch_size, main_repeat_size, -1)
            timestep_proj = timestep_proj.view(batch_size, 6, 1, -1).expand(batch_size, 6, main_repeat_size, -1)

            if indices_hidden_states is not None and self.zero_history_timestep:
                temb = torch.cat([temb_t0, temb], dim=1)
                timestep_proj = torch.cat([timestep_proj_t0, timestep_proj], dim=2)

        if timestep_proj.ndim == 4:
            timestep_proj = timestep_proj.permute(0, 2, 1, 3)

        # 4. Transformer blocks — with Edit Adapter injection
        logits_hidden = []
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        rotary_emb = rotary_emb.contiguous()

        has_adapter = edit_instruction_embeds is not None

        # Compute sigma modulation once (shared across all adapter layers).
        sigma_modulation = None
        if has_adapter:
            _ts = timestep[0] if isinstance(timestep, list) else timestep
            if _ts.ndim == 0:
                _ts = _ts.unsqueeze(0).expand(batch_size)
            sigma_modulation = self.edit_adapter.compute_sigma_modulation(_ts)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for iidx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    navit_hidden_attention_mask,
                    navit_encoder_attention_mask,
                    original_context_length,
                    original_context_length_list,
                    is_first_denoising_step,
                )
                # --- Edit Adapter injection ---
                if has_adapter:
                    hidden_states = self._gradient_checkpointing_func(
                        self.edit_adapter.adapter_blocks[iidx],
                        hidden_states,
                        edit_instruction_embeds,
                        original_context_length,
                        original_context_length_list,
                        post_patch_num_frames_list,
                        post_patch_height_list,
                        post_patch_width_list,
                        sigma_modulation,
                    )
                if gan_mode and self.is_use_gan and self.is_use_gan_hooks and iidx in self.gan_hooks:
                    logits_hidden.append(hidden_states[:, -original_context_length:, :])
        else:
            for iidx, block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    navit_hidden_attention_mask,
                    navit_encoder_attention_mask,
                    original_context_length,
                    original_context_length_list,
                    is_first_denoising_step,
                )
                # --- Edit Adapter injection ---
                if has_adapter:
                    hidden_states = self.edit_adapter.adapter_blocks[iidx](
                        hidden_states,
                        edit_instruction_embeds,
                        original_context_length,
                        original_context_length_list,
                        post_patch_num_frames_list,
                        post_patch_height_list,
                        post_patch_width_list,
                        sigma_modulation=sigma_modulation,
                    )
                if gan_mode and self.is_use_gan and self.is_use_gan_hooks and iidx in self.gan_hooks:
                    logits_hidden.append(hidden_states[:, -original_context_length:, :])

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            if not enable_navit:
                temb = temb[:, -original_context_length:, :]
            shift, scale = (self.norm_out.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(
                2, dim=2
            )
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.norm_out.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        if enable_navit:
            hidden_states = (self.norm_out.norm(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)

            output = []
            seq_start = 0
            for (
                cur_original_context_length,
                cur_post_patch_num_frames,
                cur_post_patch_height,
                cur_post_patch_width,
            ) in zip(
                reversed(original_context_length_list),
                reversed(post_patch_num_frames_list),
                reversed(post_patch_height_list),
                reversed(post_patch_width_list),
            ):
                cur_hidden_states = hidden_states[
                    :, seq_start : seq_start + cur_original_context_length + history_context_length, :
                ]
                cur_hidden_states = cur_hidden_states[:, history_context_length:, :]
                cur_hidden_states = self.proj_out(cur_hidden_states)
                seq_start += cur_original_context_length + history_context_length

                cur_hidden_states = cur_hidden_states.reshape(
                    batch_size,
                    cur_post_patch_num_frames,
                    cur_post_patch_height,
                    cur_post_patch_width,
                    p_t,
                    p_h,
                    p_w,
                    -1,
                )
                cur_hidden_states = cur_hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                cur_hidden_states = cur_hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                output.append(cur_hidden_states)

            output = output[::-1]
        else:
            hidden_states = hidden_states[:, -original_context_length:, :]
            hidden_states = (self.norm_out.norm(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(
                batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
            )
            hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
            output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        logits = []
        if gan_mode and self.is_use_gan:
            if self.is_use_gan_final:
                logits.append(self.gradient_checkpointing_method(self.gan_final_head, output))
            if self.is_use_gan_hooks:
                from einops import rearrange

                for idx, (_, gan_head) in enumerate(self.gan_heads.items()):
                    activation = rearrange(
                        logits_hidden[idx],
                        "b (f h w) c -> b c f h w",
                        f=post_patch_num_frames,
                        h=post_patch_height,
                        w=post_patch_width,
                    )
                    logits.append(self.gradient_checkpointing_method(gan_head, activation.contiguous()))
            logits = torch.cat(logits, dim=1) if len(logits) > 1 else logits[0]
            logits_hidden = None
            del logits_hidden

        if not return_dict:
            return (output, logits)

        return Transformer2DModelOutput(sample=output, logits=logits)
