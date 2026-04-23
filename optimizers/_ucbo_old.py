from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

ClosureType = Callable[[], Tensor]


class uCBO(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 1e-1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        hess_init: float = 0.0,         
        weight_decay: float = 0.0,   
        cand_curvature: float = 0.0, 
        coupled: bool = False,
        bias_correction_m: bool = True,
        bias_correction_v: bool = True,
        clip_radius: float = float("inf"),
        eps: float = 1e-8,
        rescale_lr: bool = True,
    ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0,1), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0,1), got {beta2}")
        if hess_init < 0.0:
            raise ValueError(f"hess_init must be >= 0, got {hess_init}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if cand_curvature < 0.0:
            raise ValueError(f"cand_curvature must be >= 0, got {cand_curvature}")
        if clip_radius <= 0.0:
            raise ValueError(f"clip_radius must be > 0 or inf, got {clip_radius}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            hess_init=hess_init,
            weight_decay=weight_decay,
            cand_curvature=cand_curvature,
            coupled=coupled,
            bias_correction_m=bias_correction_m,
            bias_correction_v=bias_correction_v,
            clip_radius=clip_radius,
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
            coupled: bool = group["coupled"]
            bc_m: bool = group["bias_correction_m"]
            bc_v: bool = group["bias_correction_v"]
            clip_radius: float = group["clip_radius"]
            eps: float = group["eps"]
            rescale_lr: bool = group["rescale_lr"]
            lr_eff = lr_base * (hess_init + wd - cand) if rescale_lr else lr_base

            for p in group["params"]:
                if p is None or p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("CBO does not support sparse gradients.")

                g = p.grad
                h = g.mul(g)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.full_like(p, float(hess_init))

                state["step"] += 1
                t = state["step"]

                m: Tensor = state["m"]
                v: Tensor = state["v"]

                g_for_m = g.add(p, alpha=wd) if (coupled and wd != 0.0) else g
                m.mul_(b1).add_(g_for_m, alpha=1.0 - b1)
                v.mul_(b2).add_(h, alpha=1.0 - b2)

                m_use = m / (1.0 - (b1 ** t)) if bc_m else m
                v_use = v / (1.0 - (b2 ** t)) if bc_v else v

                denom = v_use.add(wd).sub(cand).clamp_min(eps)
                numer = m_use if coupled else m_use.add(p, alpha=wd)

                step_dir = numer.div(denom)
                if clip_radius != float("inf"):
                    step_dir = step_dir.clamp(min=-clip_radius, max=clip_radius)

                p.add_(step_dir, alpha=-lr_eff)

        return loss
