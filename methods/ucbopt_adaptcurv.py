"""uCBOpt Adaptive Curvature optimizer.

Extends uCBOpt by replacing the fixed scalar curvature with a per-element
adaptive curvature proxy derived from a decayed running minimum of the
squared-gradient EMA (exp_avg_sq).
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor

ClosureType = Callable[[], Tensor]


class uCBOptAdaptCurv(torch.optim.Optimizer):
    """uCBOpt with an adaptive diagonal curvature proxy.

    State per parameter tensor:
        step           -- update count
        exp_avg        -- EMA of gradients (m_t)
        exp_avg_sq     -- EMA of squared gradients (h_t), init to hess_init
        min_exp_avg_sq -- decayed running minimum of exp_avg_sq (c_t), init to +inf

    Update rules:
        h_t = beta2 * h_{t-1} + (1 - beta2) * g_t^2          (not bias-corrected)
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t             (optionally bias-corrected)
        c_t = min(beta3 * c_{t-1}, h_t)
        denom = h_t - gamma * c_t + weight_decay
        numer = m_t + weight_decay * param
        lr_eff = lr * (hess_init + weight_decay)  if rescale_lr else lr
        param -= lr_eff * numer / denom

    When gamma=0, rescale_lr=True and bias_corr=True, this reduces to original uCBOpt
    (given the same lr, beta1, beta2, hess_init, weight_decay).
    rescale_lr=True uses the same scalar LR rescaling as uCBOpt; the per-element
    adaptive curvature term (gamma * c_t) is not included in the scalar rescale.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.999),
        weight_decay: float = 1e-4,
        hess_init: float = 0.5,
        gamma: float = 0.1,
        eps: float = 1e-8,
        maximize: bool = False,
        clip_radius: float = float("inf"),
        bias_corr: bool = True,
        rescale_lr: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        beta1, beta2, beta3 = betas
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"betas[0] (beta1) must be in [0,1), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"betas[1] (beta2) must be in [0,1), got {beta2}")
        if not (0.0 <= beta3 < 1.0):
            raise ValueError(f"betas[2] (beta3) must be in [0,1), got {beta3}")
        if hess_init <= 0.0:
            raise ValueError(f"hess_init must be > 0, got {hess_init}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if not isinstance(rescale_lr, bool):
            raise TypeError(f"rescale_lr must be bool, got {type(rescale_lr).__name__}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            hess_init=hess_init,
            gamma=gamma,
            eps=eps,
            maximize=maximize,
            clip_radius=clip_radius,
            bias_corr=bias_corr,
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
            lr: float = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            wd: float = group["weight_decay"]
            hess_init: float = group["hess_init"]
            gamma: float = group["gamma"]
            eps: float = group["eps"]
            maximize: bool = group["maximize"]
            clip_radius: float = group["clip_radius"]
            bias_corr: bool = group["bias_corr"]
            rescale_lr: bool = group["rescale_lr"]
            lr_eff: float = lr * (hess_init + wd) if rescale_lr else lr

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            min_exp_avg_sqs: list[Tensor] = []
            step_counts: list[int] = []

            for p in group["params"]:
                if p is None or p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("uCBOptAdaptCurv does not support sparse gradients.")

                params_with_grad.append(p)
                grads.append(p.grad if not maximize else p.grad.neg())

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.full_like(p, float(hess_init))
                    # decayed running minimum of exp_avg_sq; tracks adaptive curvature
                    state["min_exp_avg_sq"] = torch.full_like(p, float("inf"))

                state["step"] += 1
                step_counts.append(state["step"])
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                min_exp_avg_sqs.append(state["min_exp_avg_sq"])

            if not params_with_grad:
                continue

            # h_t = beta2 * h_{t-1} + (1 - beta2) * g_t^2
            grad_sq = torch._foreach_mul(grads, grads)
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_add_(exp_avg_sqs, grad_sq, alpha=1.0 - beta2)

            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)

            # c_t = min(beta3 * c_{t-1}, h_t)  — decayed running minimum
            torch._foreach_mul_(min_exp_avg_sqs, beta3)
            for c, h in zip(min_exp_avg_sqs, exp_avg_sqs):
                torch.minimum(c, h, out=c)

            # bias-correct only m, matching original uCBOpt; h is not bias-corrected
            if bias_corr:
                bc_m = [1.0 - beta1 ** t for t in step_counts]
                m_hat = torch._foreach_div(exp_avgs, bc_m)
            else:
                m_hat = list(exp_avgs)
            h_hat = list(exp_avg_sqs)

            # denom = h_t - gamma * c_t + weight_decay
            denom = torch._foreach_add(h_hat, min_exp_avg_sqs, alpha=-gamma)
            torch._foreach_add_(denom, wd)
            torch._foreach_clamp_min_(denom, eps)

            # Coupled weight decay: matches original uCBOpt when gamma=0
            # and the same beta1, beta2, hess_init, lr, and weight_decay are used.
            numer = torch._foreach_add(m_hat, params_with_grad, alpha=wd)

            update = torch._foreach_div(numer, denom)

            # optional elementwise clipping of the update
            if math.isfinite(clip_radius):
                torch._foreach_clamp_(update, -clip_radius, clip_radius)

            torch._foreach_add_(params_with_grad, update, alpha=-lr_eff)

        return loss
