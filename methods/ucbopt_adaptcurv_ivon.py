"""IVON with an adaptive curvature term subtracted from the update denominator.

Extends uCBOptIVON by replacing the fixed scalar cand_curvature with a
per-element adaptive proxy: gamma * c_t, where c_t is a decayed running
minimum of IVON's hessian estimate.

Update rules:
    hess updated via IVON Price/GradSq method
    c_t = min(beta3 * c_{t-1}, hess_t)
    denom = hess_t + wd - gamma * c_t          (clamped to eps)
    numer = momentum / debias + wd * param
    param -= lr_eff * clip(numer / denom, ±clip_radius)

When gamma=0 this reduces to standard IVON.
"""

from math import pow
from typing import Callable, Optional, Tuple
from contextlib import contextmanager
import torch
import torch.optim
import torch.distributed as dist
from torch import Tensor


ClosureType = Callable[[], Tensor]


def _welford_mean(avg: Optional[Tensor], newval: Tensor, count: int) -> Tensor:
    return newval if avg is None else avg + (newval - avg) / count


class uCBOptAdaptCurvIVON(torch.optim.Optimizer):
    hessian_approx_methods = (
        'price',
        'gradsq',
    )

    def __init__(
        self,
        params,
        lr: float,
        ess: float,
        hess_init: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        beta3: float = 0.999,
        weight_decay: float = 1e-4,
        gamma: float = 0.1,
        eps: float = 1e-8,
        mc_samples: int = 1,
        hess_approx: str = 'price',
        clip_radius: float = float("inf"),
        sync: bool = False,
        debias: bool = True,
        rescale_lr: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 1 <= mc_samples:
            raise ValueError(f"Invalid number of MC samples: {mc_samples}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if not 0.0 < hess_init:
            raise ValueError(f"Invalid Hessian initialization: {hess_init}")
        if not 0.0 < ess:
            raise ValueError(f"Invalid effective sample size: {ess}")
        if not 0.0 < clip_radius:
            raise ValueError(f"Invalid clipping radius: {clip_radius}")
        if not 0.0 <= beta1 <= 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 <= 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if not 0.0 <= beta3 < 1.0:
            raise ValueError(f"beta3 must be in [0, 1), got {beta3}")
        if hess_approx not in self.hessian_approx_methods:
            raise ValueError(f"Invalid hess_approx parameter: {hess_approx}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")

        defaults = dict(
            lr=lr,
            mc_samples=mc_samples,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            weight_decay=weight_decay,
            gamma=gamma,
            eps=eps,
            hess_init=hess_init,
            ess=ess,
            clip_radius=clip_radius,
        )
        super().__init__(params, defaults)

        self.mc_samples = mc_samples
        self.hess_approx = hess_approx
        self.sync = sync
        self._numel, self._device, self._dtype = self._get_param_configs()
        self.current_step = 0
        self.debias = debias
        self.rescale_lr = rescale_lr

        self._reset_samples()
        self._init_buffers()

    def _get_param_configs(self):
        all_params = []
        for pg in self.param_groups:
            pg["numel"] = sum(p.numel() for p in pg["params"] if p is not None)
            all_params += [p for p in pg["params"] if p is not None]
        if len(all_params) == 0:
            return 0, torch.device("cpu"), torch.get_default_dtype()
        devices = {p.device for p in all_params}
        if len(devices) > 1:
            raise ValueError(
                f"Parameters are on different devices: {[str(d) for d in devices]}"
            )
        device = next(iter(devices))
        dtypes = {p.dtype for p in all_params}
        if len(dtypes) > 1:
            raise ValueError(
                f"Parameters are on different dtypes: {[str(d) for d in dtypes]}"
            )
        dtype = next(iter(dtypes))
        total = sum(pg["numel"] for pg in self.param_groups)
        return total, device, dtype

    def _reset_samples(self):
        self.state['count'] = 0
        self.state['avg_grad'] = None
        self.state['avg_nxg'] = None
        self.state['avg_gsq'] = None

    def _init_buffers(self):
        for group in self.param_groups:
            hess_init, numel = group["hess_init"], group["numel"]
            group["momentum"] = torch.zeros(numel, device=self._device, dtype=self._dtype)
            group["hess"] = torch.zeros(
                numel, device=self._device, dtype=self._dtype
            ).add(torch.as_tensor(hess_init))
            # decayed running minimum of hess; tracks adaptive curvature proxy
            group["min_hess"] = torch.full(
                (numel,), float("inf"), device=self._device, dtype=self._dtype
            )

    @contextmanager
    def sampled_params(self, train: bool = False):
        param_avg, noise = self._sample_params()
        yield
        self._restore_param_average(train, param_avg, noise)

    def _restore_param_average(self, train: bool, param_avg: Tensor, noise: Tensor):
        param_grads = []
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue
                p_slice = slice(offset, offset + p.numel())
                p.data = param_avg[p_slice].view(p.shape)
                if train:
                    if p.requires_grad:
                        param_grads.append(p.grad.flatten())
                    else:
                        param_grads.append(torch.zeros_like(p).flatten())
                offset += p.numel()
        assert offset == self._numel

        if train:
            grad_sample = torch.cat(param_grads, 0)
            count = self.state["count"] + 1
            self.state["count"] = count
            self.state["avg_grad"] = _welford_mean(self.state["avg_grad"], grad_sample, count)
            if self.hess_approx == 'price':
                self.state['avg_nxg'] = _welford_mean(
                    self.state['avg_nxg'], noise * grad_sample, count)
            elif self.hess_approx == 'gradsq':
                self.state['avg_gsq'] = _welford_mean(
                    self.state['avg_gsq'], grad_sample.square(), count)

    @torch.no_grad()
    def step(self, closure: ClosureType = None) -> Optional[Tensor]:
        if closure is None:
            loss = None
        else:
            losses = []
            for _ in range(self.mc_samples):
                with torch.enable_grad():
                    loss = closure()
                losses.append(loss)
            loss = sum(losses) / self.mc_samples
        if self.sync and dist.is_initialized():
            self._sync_samples()
        self._update()
        self._reset_samples()
        return loss

    def _sync_samples(self):
        world_size = dist.get_world_size()
        dist.all_reduce(self.state["avg_grad"])
        self.state["avg_grad"].div_(world_size)
        dist.all_reduce(self.state["avg_nxg"])
        self.state["avg_nxg"].div_(world_size)

    def _sample_params(self) -> Tuple[Tensor, Tensor]:
        noise_samples = []
        param_avgs = []
        offset = 0
        for group in self.param_groups:
            gnumel = group["numel"]
            noise_sample = (
                torch.randn(gnumel, device=self._device, dtype=self._dtype)
                / (group["ess"] * (group["hess"] + group["weight_decay"])).sqrt()
            )
            noise_samples.append(noise_sample)

            goffset = 0
            for p in group["params"]:
                if p is None:
                    continue
                p_avg = p.data.flatten()
                numel = p.numel()
                p_noise = noise_sample[goffset : goffset + numel]
                param_avgs.append(p_avg)
                p.data = (p_avg + p_noise).view(p.shape)
                goffset += numel
                offset += numel
            assert goffset == group["numel"]
        assert offset == self._numel

        return torch.cat(param_avgs, 0), torch.cat(noise_samples, 0)

    def _update(self):
        self.current_step += 1
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            b3 = group["beta3"]
            pg_slice = slice(offset, offset + group["numel"])

            param_avg = torch.cat(
                [p.flatten() for p in group["params"] if p is not None], 0
            )

            group["momentum"] = self._new_momentum(
                self.state["avg_grad"][pg_slice], group["momentum"], b1
            )

            group["hess"] = self._new_hess(
                self.hess_approx,
                group["hess"],
                self.state["avg_nxg"],
                self.state['avg_gsq'],
                pg_slice,
                group["ess"],
                b2,
                group["weight_decay"],
            )

            # c_t = min(beta3 * c_{t-1}, hess_t)
            group["min_hess"].mul_(b3)
            torch.minimum(group["min_hess"], group["hess"], out=group["min_hess"])

            param_avg = self._new_param_averages(
                param_avg,
                group["hess"],
                group["min_hess"],
                group["momentum"],
                lr * (group["hess_init"] + group["weight_decay"]) if self.rescale_lr else lr,
                group["weight_decay"],
                group["gamma"],
                group["eps"],
                group["clip_radius"],
                1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0,
            )

            pg_offset = 0
            for p in group["params"]:
                if p is not None:
                    p.data = param_avg[pg_offset : pg_offset + p.numel()].view(p.shape)
                    pg_offset += p.numel()
            assert pg_offset == group["numel"]
            offset += group["numel"]
        assert offset == self._numel

    @staticmethod
    def _get_nll_hess(method: str, hess, avg_nxg, avg_gsq, pg_slice, ess) -> Tensor:
        if method == 'price':
            return avg_nxg[pg_slice] * hess * ess
        elif method == 'gradsq':
            return avg_gsq[pg_slice]
        else:
            raise NotImplementedError(f'unknown hessian approx.: {method}')

    @staticmethod
    def _new_momentum(avg_grad, m, b1) -> Tensor:
        return b1 * m + (1.0 - b1) * avg_grad

    @staticmethod
    def _new_hess(method, hess, avg_nxg, avg_gsq, pg_slice, ess, beta2, wd) -> Tensor:
        f = uCBOptAdaptCurvIVON._get_nll_hess(
            method, hess + wd, avg_nxg, avg_gsq, pg_slice, ess
        )
        return beta2 * hess + (1.0 - beta2) * f + \
            (0.5 * (1 - beta2) ** 2) * (hess - f).square() / (hess + wd)

    @staticmethod
    def _new_param_averages(
        param_avg, hess, min_hess, momentum, lr, wd, gamma, eps, clip_radius, debias
    ) -> Tensor:
        denom = (hess + wd - gamma * min_hess).clamp_min(eps)
        return param_avg - lr * torch.clip(
            (momentum / debias + wd * param_avg) / denom,
            min=-clip_radius,
            max=clip_radius,
        )
