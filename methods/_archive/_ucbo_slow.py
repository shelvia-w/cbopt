"""Archived slower reference implementation of the uCBOpt optimizer."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

ClosureType = Callable[[], Tensor]


class uCBO_slow(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 0.2,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        hess_init: float = 0.5,         
        weight_decay: float = 0.0,   
        cand_curvature: float = 0.0, 
        denom_floor: float = 1e-8,
        rescale_lr: bool = True,
    ):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0,1), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0,1), got {beta2}")
        if hess_init < 0.0:
            raise ValueError(f"hess_init must be > 0, got {hess_init}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if cand_curvature < 0.0:
            raise ValueError(f"cand_curvature must be >= 0, got {cand_curvature}")
        if cand_curvature > weight_decay:
            raise ValueError(f"cand_curvature must be <= weight_decay (got cand_curvature={cand_curvature}, weight_decay={weight_decay})")
        if denom_floor < 0.0:
            raise ValueError(f"denom_floor must be >= 0, got {denom_floor}")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            hess_init=hess_init,
            weight_decay=weight_decay,
            cand_curvature=cand_curvature,
            denom_floor=denom_floor,
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
            denom_floor: float = group["denom_floor"]
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

                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).add_(h, alpha=1.0 - b2)

                m_bc = m / (1.0 - (b1 ** t))

                denom = v.add(wd).sub(cand).clamp_min(denom_floor)
                numer = m_bc.add(p, alpha=wd)

                step_dir = numer.div(denom)
                p.add_(step_dir, alpha=-lr_eff)

        return loss
