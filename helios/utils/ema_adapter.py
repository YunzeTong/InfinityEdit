"""
Lightweight EMA (Exponential Moving Average) for Edit Adapter.

No deepspeed dependency — works with pure accelerate multi-GPU training.
Based on EMAModel_Zero3, stripped of all ZeRO-3 partition logic.
"""

import copy
from typing import Iterable, Union

import torch


class EMAAdapter:
    """
    Exponential Moving Average of edit adapter weights.

    Maintains a shadow copy of the adapter module. After each optimizer step,
    call `step(parameters)` to update the shadow weights.

    For validation, use `store()` / `copy_to()` / `restore()` to temporarily
    swap in EMA weights.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
    ):
        self.shadow_model = copy.deepcopy(model)
        self.shadow_model.requires_grad_(False)
        self.shadow_model.to("cpu")

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None

        self.temp_stored_params = None

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        parameters = list(parameters)
        self.optimization_step += 1

        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        for s_param, param in zip(self.shadow_model.parameters(), parameters):
            if param.requires_grad:
                param_cpu = param.data.to(device="cpu", dtype=s_param.dtype)
                s_param.data.sub_(one_minus_decay * (s_param.data - param_cpu))
            else:
                s_param.data.copy_(param.data.to("cpu"))

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Copy EMA weights into the given parameters (for validation)."""
        for s_param, param in zip(self.shadow_model.parameters(), parameters):
            param.data.copy_(s_param.to(param.device).data)

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Save current live parameters for later restoration."""
        self.temp_stored_params = [p.detach().cpu().clone() for p in parameters]

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Restore parameters saved by `store()`."""
        if self.temp_stored_params is None:
            raise RuntimeError("No stored params to restore. Call store() first.")
        for stored, param in zip(self.temp_stored_params, parameters):
            param.data.copy_(stored.data)
        self.temp_stored_params = None

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "shadow_model": self.shadow_model.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict.get("decay", self.decay)
        self.min_decay = state_dict.get("min_decay", self.min_decay)
        self.optimization_step = state_dict.get("optimization_step", self.optimization_step)
        self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        self.power = state_dict.get("power", self.power)

        shadow_sd = state_dict.get("shadow_model", None)
        if shadow_sd is not None:
            self.shadow_model.load_state_dict(shadow_sd)
