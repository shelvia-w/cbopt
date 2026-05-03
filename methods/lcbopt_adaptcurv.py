"""lCBOpt Adaptive Curvature optimizer.

Lower-CBO variant with decayed running maximum curvature tracking.

The curvature proxy incorporates weight decay:

    H_t      = exp_avg_sq                      (EMA of squared gradients)
    H_curv_t = H_t + weight_decay
    C_t      = max(beta3 * C_{t-1}, H_curv_t)  (decayed running max of H_curv)

    denom = gamma * C_t - H_curv_t + eps

Because C_t >= H_curv_t (the running max is never below the current value),
denom >= (gamma - 1) * H_curv_t + eps > 0 when gamma > 1.

gamma > 1 is required and enforced in __init__.

This intentionally differs from uCBOptAdaptCurv (upper-CBO), which uses:

    denom = H_curv_t - gamma * C_t

where C_t is a decayed running *minimum*.  Here the roles are reversed:
C_t is a running *maximum* and the sign in the denominator is flipped.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor

ClosureType = Callable[[], Tensor]


class lCBOptAdaptCurv(torch.optim.Optimizer):
    """Lower-CBO optimizer with decayed running maximum curvature tracking.

    State per parameter tensor:
        step            -- update count
        exp_avg         -- EMA of gradients (m_t)
        exp_avg_sq      -- EMA of squared gradients (h_t), init to hess_init
        max_exp_avg_sq  -- decayed running max of exp_avg_sq (c_t), init to hess_init

    Update rules:
        h_t      = beta2 * h_{t-1} + (1 - beta2) * g_t^2
        h_curv_t = h_t + weight_decay
        c_t      = max(beta3 * c_{t-1}, h_curv_t)
        m_t      = beta1 * m_{t-1} + (1 - beta1) * g_t
        m_hat    = m_t / (1 - beta1^t)      [bias-corrected]
        denom    = gamma * c_t - h_curv_t + eps   [lower-CBO, always > 0 when gamma > 1]
        numer    = m_hat + weight_decay * param
        update   = numer / denom
        param   -= lr_eff * update

    gamma > 1 is required (enforced in __init__).
    lr_eff = lr * (hess_init + weight_decay) if rescale_lr else lr
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.99999, 0.999),
        weight_decay: float = 1e-4,
        hess_init: float = 1.0,
        gamma: float = 1.05,
        eps: float = 1e-6,
        maximize: bool = False,
        clip_radius: float = 1.0,
        rescale_lr: bool = False,
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
        if gamma <= 1.0:
            raise ValueError(f"gamma must be > 1 (required for denom = gamma*c - h_curv + eps > 0), got {gamma}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if not (math.isinf(clip_radius) or clip_radius >= 0.0):
            raise ValueError(f"clip_radius must be >= 0 or inf, got {clip_radius}")
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
            rescale_lr: bool = group["rescale_lr"]
            lr_eff: float = lr * (hess_init + wd) if rescale_lr else lr

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            step_counts: list[int] = []

            for p in group["params"]:
                if p is None or p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "uCBOptLowerAdaptCurv does not support sparse gradients."
                    )

                params_with_grad.append(p)
                grads.append(p.grad if not maximize else p.grad.neg())

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.full_like(p, float(hess_init))
                    # running max of h_curv = h + wd; init consistent with h_curv
                    state["max_exp_avg_sq"] = torch.full_like(p, float(hess_init) + wd)

                state["step"] += 1
                step_counts.append(state["step"])
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])

            if not params_with_grad:
                continue

            # h_t = beta2 * h_{t-1} + (1 - beta2) * g_t^2
            grad_sq = torch._foreach_mul(grads, grads)
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_add_(exp_avg_sqs, grad_sq, alpha=1.0 - beta2)

            # h_curv_t = h_t + weight_decay
            h_curv = torch._foreach_add(exp_avg_sqs, wd)

            # c_t = max(beta3 * c_{t-1}, h_curv_t)  — decayed running maximum
            torch._foreach_mul_(max_exp_avg_sqs, beta3)
            for c, hc in zip(max_exp_avg_sqs, h_curv):
                torch.maximum(c, hc, out=c)

            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)

            # bias-correct m only; h is not bias-corrected
            bc_m = [1.0 - beta1 ** t for t in step_counts]
            m_hat = torch._foreach_div(exp_avgs, bc_m)

            # Lower-CBO denominator:
            #   H_curv_t = h_t + wd,  C_t = max_exp_avg_sq (tracks H_curv)
            #   denom = gamma * C_t - H_curv_t + eps
            # C_t >= H_curv_t by construction, so denom >= (gamma-1)*H_curv_t + eps > 0.
            denom = torch._foreach_mul(max_exp_avg_sqs, gamma)
            torch._foreach_add_(denom, h_curv, alpha=-1.0)
            torch._foreach_add_(denom, eps)
            torch._foreach_clamp_min_(denom, eps)

            # Coupled weight decay: numer = m_hat + weight_decay * param
            numer = torch._foreach_add(m_hat, params_with_grad, alpha=wd)

            update = torch._foreach_div(numer, denom)

            # optional elementwise clipping of the update
            if math.isfinite(clip_radius):
                torch._foreach_clamp_(update, -clip_radius, clip_radius)

            torch._foreach_add_(params_with_grad, update, alpha=-lr_eff)

        return loss
