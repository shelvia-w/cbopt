"""Implementation of the uCBOpt optimizer."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

ClosureType = Callable[[], Tensor]

class uCBOpt(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 0.2,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        hess_init: float = 0.5,         
        weight_decay: float = 0.0,   
        cand_curvature: float = 0.0,
        eps: float = 1e-8,
        rescale_lr: bool = True,
    ):

        if lr < 0.0:
            raise ValueError(f"learning rate must be >= 0, got {lr}")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0,1), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0,1), got {beta2}")
        if hess_init <= 0.0:
            raise ValueError(f"hess_init must be > 0, got {hess_init}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if cand_curvature < 0.0:
            raise ValueError(f"cand_curvature must be >= 0, got {cand_curvature}")
        if cand_curvature > weight_decay:
            raise ValueError(
                "cand_curvature must be <= weight_decay "
                f"(got cand_curvature={cand_curvature}, weight_decay={weight_decay})"
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if not isinstance(rescale_lr, bool):
            raise TypeError(f"rescale_lr must be bool, got {type(rescale_lr).__name__}")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            hess_init=hess_init,
            weight_decay=weight_decay,
            cand_curvature=cand_curvature,
            eps=eps,
            rescale_lr=rescale_lr,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[ClosureType] = None) -> Optional[Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr_base: float = group["lr"]
            b1: float = group["beta1"]
            b2: float = group["beta2"]
            hess_init: float = group["hess_init"]
            wd: float = group["weight_decay"]
            cand: float = group["cand_curvature"]
            eps: float = group["eps"]
            rescale_lr: bool = group["rescale_lr"]
            lr_eff = lr_base * (hess_init + wd) if rescale_lr else lr_base

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            m_list: list[Tensor] = []
            v_list: list[Tensor] = []
            step_counts: list[int] = []

            for p in group["params"]:
                if p is None or p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("uCBOpt does not support sparse gradients.")

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.full_like(p, float(hess_init))

                state["step"] += 1
                step_counts.append(state["step"])
                m_list.append(state["m"])
                v_list.append(state["v"])

            if len(params_with_grad) == 0:
                continue

            h_list = torch._foreach_mul(grads, grads)
            g_for_m = grads

            torch._foreach_mul_(m_list, b1)
            torch._foreach_add_(m_list, g_for_m, alpha=1.0 - b1)

            torch._foreach_mul_(v_list, b2)
            torch._foreach_add_(v_list, h_list, alpha=1.0 - b2)

            bias_corr_m = [1.0 - (b1 ** t) for t in step_counts]
            m_bar = torch._foreach_div(m_list, bias_corr_m)

            denom = torch._foreach_add(v_list, wd)
            torch._foreach_sub_(denom, cand)
            torch._foreach_clamp_min_(denom, eps)
            numer = torch._foreach_add(m_bar, params_with_grad, alpha=wd)

            step_dir = torch._foreach_div(numer, denom)

            torch._foreach_add_(params_with_grad, step_dir, alpha=-lr_eff)

        return loss
