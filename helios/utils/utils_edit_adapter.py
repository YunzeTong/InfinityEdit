"""
Training utilities for Edit Adapter.

Flow-matching loss that passes edit_instruction_embeds to the transformer, plus
the mixture-of-Gaussians sigma sampler used to concentrate training on the sigma
values the inference schedule actually visits.
"""

import torch
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from .utils_base import apply_schedule_shift


def compute_density_mixture_gaussian(batch_size, centers, stds, weights,
                                     sigma_min=0.001, sigma_max=0.999,
                                     use_rejection_sampling=False):
    """Sample raw sigma from a mixture of Gaussians.

    Args:
        centers: list of Gaussian centers, e.g. [0.999, 0.500]
        stds:    list of per-component stds,  e.g. [0.025, 0.025]
        weights: list of mixture weights (unnormalized), e.g. [1.0, 1.0]
        use_rejection_sampling: if True, resample out-of-range values instead
            of clamping, avoiding boundary density accumulation.

    Returns:
        Tensor of shape (batch_size,) with sampled sigma values.
    """
    weights_t = torch.tensor(weights, dtype=torch.float32)
    weights_t = weights_t / weights_t.sum()
    centers_t = torch.tensor(centers, dtype=torch.float32)
    stds_t = torch.tensor(stds, dtype=torch.float32)

    if not use_rejection_sampling:
        component_idx = torch.multinomial(weights_t, batch_size, replacement=True)
        samples = torch.normal(centers_t[component_idx], stds_t[component_idx])
        samples = samples.clamp(sigma_min, sigma_max)
        return samples

    result = torch.empty(batch_size)
    remaining = batch_size
    offset = 0
    for _ in range(50):
        idx = torch.multinomial(weights_t, remaining, replacement=True)
        candidates = torch.normal(centers_t[idx], stds_t[idx])
        valid = (candidates >= sigma_min) & (candidates <= sigma_max)
        n_valid = valid.sum().item()
        result[offset:offset + n_valid] = candidates[valid]
        offset += n_valid
        remaining = batch_size - offset
        if remaining == 0:
            break
    if remaining > 0:
        result[offset:] = (sigma_min + sigma_max) / 2
    return result


def flow_loss_edit_adapter(
    args,
    accelerator,
    transformer,
    prompt_embeds,
    edit_prompt_embeds,
    noisy_model_input,
    sigmas,
    timesteps,
    target,
    indices_hidden_states,
    latents_history_short,
    indices_latents_history_short,
    latents_history_mid,
    indices_latents_history_mid,
    latents_history_long,
    indices_latents_history_long,
):
    """Compute flow-matching loss with edit adapter injection."""
    model_pred = transformer(
        hidden_states=noisy_model_input,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        indices_hidden_states=indices_hidden_states,
        indices_latents_history_short=indices_latents_history_short,
        indices_latents_history_mid=indices_latents_history_mid,
        indices_latents_history_long=indices_latents_history_long,
        latents_history_short=latents_history_short,
        latents_history_mid=latents_history_mid,
        latents_history_long=latents_history_long,
        edit_instruction_embeds=edit_prompt_embeds,
        return_dict=False,
    )[0]

    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme=args.training_config.weighting_scheme, sigmas=sigmas
    )

    per_elem_loss = weighting.float() * (model_pred.float() - target.float()) ** 2

    # Optional: increasing temporal weights along the latent-frame axis so the
    # model focuses more on the later frames (which tend to degrade in quality).
    # Weights are linearly ramped from 1.0 to `temporal_weight_max_ratio` across
    # the T latent frames, then normalized to mean=1 so the overall loss
    # magnitude is unchanged.
    if getattr(args.training_config, "reweighting_along_time", False):
        T = target.shape[2]
        max_ratio = getattr(args.training_config, "temporal_weight_max_ratio", 2.0)
        temporal_weights = torch.linspace(
            1.0, max_ratio, steps=T,
            device=target.device, dtype=torch.float32,
        )
        temporal_weights = temporal_weights / temporal_weights.mean()
        temporal_weights = temporal_weights.view(1, 1, T, 1, 1)
        per_elem_loss = per_elem_loss * temporal_weights

    loss = torch.mean(
        per_elem_loss.reshape(target.shape[0], -1),
        1,
    ).mean()

    assert loss.requires_grad, f"Loss should have gradient! Got {loss.requires_grad}"
    accelerator.backward(loss)

    return loss


def prepare_noise_input_edit_adapter(
    args,
    model_input,
    noise_scheduler,
):
    """Simplified noise input preparation for edit adapter (no error recycling, no corruption)."""
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]

    # Sample timesteps
    sigma_sampling = getattr(args.training_config, "sigma_sampling_strategy", "logit_normal")
    training_steps = noise_scheduler.config.num_train_timesteps

    if sigma_sampling == "mixture_gaussian":
        centers = list(args.training_config.mixture_centers)
        stds = list(args.training_config.mixture_stds)
        weights = list(args.training_config.mixture_weights)
        floor_center = getattr(args.training_config, "raw_final_sigma_floor", None)
        if floor_center is not None:
            centers.append(floor_center)
            stds.append(stds[-1])
            weights.append(weights[-1])
        use_rejection = getattr(args.training_config, "mixture_rejection_sampling", False)
        sampled_sigmas = compute_density_mixture_gaussian(
            batch_size=bsz,
            centers=centers,
            stds=stds,
            weights=weights,
            use_rejection_sampling=use_rejection,
        )
        # scheduler.sigmas is descending linspace(1.0, 0.001, 1000), step=(1.0-0.001)/999
        # Convert sampled sigma to nearest grid index
        sigmas_grid = noise_scheduler.sigmas  # (1000,) descending
        indices = torch.argmin(
            (sigmas_grid.unsqueeze(0) - sampled_sigmas.unsqueeze(1)).abs(), dim=1
        )
    else:
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.training_config.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.training_config.logit_mean,
            logit_std=args.training_config.logit_std,
            mode_scale=args.training_config.mode_scale,
        )
        indices = (u * training_steps).long()

    noise_scheduler.temp_sigmas = noise_scheduler.sigmas
    noise_scheduler.temp_timesteps = noise_scheduler.timesteps

    if args.training_config.use_dynamic_shifting:
        noise_scheduler.temp_sigmas = apply_schedule_shift(
            noise_scheduler.sigmas,
            noise,
            base_seq_len=args.training_config.base_seq_len,
            max_seq_len=args.training_config.max_seq_len,
            base_shift=args.training_config.base_shift,
            max_shift=args.training_config.max_shift,
        )
        noise_scheduler.temp_timesteps = noise_scheduler.temp_sigmas * 1000.0
        while noise_scheduler.temp_timesteps.ndim > 1:
            noise_scheduler.temp_timesteps = noise_scheduler.temp_timesteps.squeeze(-1)

    timesteps = noise_scheduler.temp_timesteps[indices].to(
        device=model_input.device, non_blocking=True
    )

    sigmas = noise_scheduler.temp_sigmas[indices].flatten()
    while len(sigmas.shape) < model_input.ndim:
        sigmas = sigmas.unsqueeze(-1)
    sigmas = sigmas.to(model_input.device, dtype=model_input.dtype)

    # Flow matching: zt = (1 - t) * x + t * z
    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
    target = noise - model_input

    return noisy_model_input, sigmas, timesteps, target
