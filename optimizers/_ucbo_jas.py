'''
Basically, IVON but
    1. without sampling (line 2 in the pseudocode),
    2. with squared gradients (line 3),
    3. without correction to h (line 5), and
    4. denominator reduced by min_exp_avg_sq moving estimate.
'''


# mypy: allow-untyped-defs
from types import NoneType
from typing import cast, Optional, Union
import functools

import torch
from torch import Tensor

from torch.optim.optimizer import (
    _get_value,
    _to_scalar,
    DeviceDict,
    Optimizer,
)
import torch.nn as nn


def _use_grad_for_differentiable(func):
    def _use_grad(*args, **kwargs):
        import torch._dynamo

        self = cast(Optimizer, args[0])  # assume first positional arg is `self`
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults.get("differentiable", False))
            torch._dynamo.graph_break()
            ret = func(*args, **kwargs)
        finally:
            torch._dynamo.graph_break()
            torch.set_grad_enabled(prev_grad)
        return ret

    functools.update_wrapper(_use_grad, func)
    return _use_grad


class uCBOpt(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        rescale_lr: bool = False,
        betas: tuple[Union[float, Tensor], Union[float, Tensor], Union[NoneType, float, Tensor]] = (0.9, 0.999, None),
        weight_decay: float = 1e-2,
        decoupled_wd: bool = False,
        hess_init: float = 0.,
        bias_corr: bool = False,
        gamma: float = 0.,
        perturb_rad: float = 0.,
        eps: float = 1e-8,
        clip_radius: float = torch.inf,
        *,
        foreach: bool = True,       # Unused
        maximize: bool = False,
    ):
        if isinstance(lr, Tensor):
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if betas[2] is not None and not 1.0 <= betas[2]:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= hess_init:
            raise ValueError("Invalid Hessian initialization: {}".format(hess_init))
        if not 0.0 < clip_radius:
            raise ValueError("Invalid clipping radius: {}".format(clip_radius))
        if not 0.0 <= gamma <= 1.:
            raise ValueError("Invalid curvature dampening factor: {}".format(gamma))
        if not 0.0 <= perturb_rad:
            raise ValueError("Invalid perturbation radius: {}".format(perturb_rad))
        
        if not (
            (isinstance(betas[0], float) and isinstance(betas[1], float))
            or (isinstance(betas[0], Tensor) and isinstance(betas[1], Tensor))
        ):
            raise ValueError("betas must be either both floats or both Tensors")
        if isinstance(betas[0], Tensor):
            if betas[0].numel() != 1:
                raise ValueError("Tensor betas[0] must be 1-element")
        if isinstance(betas[1], Tensor):
            if betas[1].numel() != 1:
                raise ValueError("Tensor betas[1] must be 1-element")
        if isinstance(betas[2], Tensor):
            if betas[2].numel() != 1:
                raise ValueError("Tensor betas[2] must be 1-element")
        if (betas[2] is None) != (gamma == 0.):
            raise ValueError(f"gamma is 0. if and only if betas[2] is None. Received {gamma = } and {betas[2] = }")

        defaults = dict(
            lr=lr,
            rescale_lr=rescale_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            foreach=True,
            decoupled_wd=decoupled_wd,
            hess_init=hess_init,
            bias_corr=bias_corr,
            clip_radius=clip_radius,
            gamma=gamma,
            perturb_rad=perturb_rad,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("decoupled_wd", False)
            scalar_dtype = torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(step_val, dtype=scalar_dtype)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        min_exp_avg_sqs,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    scalar_dtype = torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
                    state["step"] = torch.tensor(0.0, dtype=scalar_dtype)
                    # Exponential moving average of gradient values -- initialize with hess_init to copy IVON
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.full_like(
                        p, group["hess_init"], memory_format=torch.preserve_format
                    )
                    if group["betas"][2] is not None:
                        # Maintains min of all exp. moving avg. of sq. grad. values
                        state["min_exp_avg_sq"] = torch.full_like(
                            p, torch.inf, memory_format=torch.preserve_format
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                # Foreach without capturable does not support a tensor lr
                if (
                    group["foreach"]
                    and torch.is_tensor(group["lr"])
                ):
                    raise RuntimeError(
                        "lr as a Tensor is not supported for capturable=False and foreach=True"
                    )

                state_steps.append(state["step"])
        return

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            min_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2, beta3 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                min_exp_avg_sqs,
                state_steps,
            )

            _multi_tensor_adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                min_exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                rescale_lr=group["rescale_lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                decoupled_wd=group["decoupled_wd"],
                hess_init=group["hess_init"],
                bias_corr=group["bias_corr"],
                beta3=beta3,
                gamma=group["gamma"],
                perturb_rad=group["perturb_rad"],
                clip_radius=group["clip_radius"],
            )

        return loss


def _multi_tensor_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    min_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: Union[float, Tensor],
    beta2: Union[float, Tensor],
    lr: Union[float, Tensor],
    rescale_lr: bool,
    weight_decay: float,
    eps: float,
    maximize: bool,
    decoupled_wd: bool,
    hess_init: float,
    bias_corr: bool,
    beta3: float,
    gamma: float,
    perturb_rad: float,
    clip_radius: float,
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor):
        raise RuntimeError(
            "lr as a Tensor is not supported for capturable=False and foreach=True"
        )
        
    if isinstance(beta1, Tensor):
        raise ValueError(
            "beta1 as a Tensor is not supported for capturable=False and foreach=True"
        )
        
    if isinstance(beta2, Tensor):
        raise ValueError(
            "beta2 as a Tensor is not supported for capturable=False and foreach=True"
        )

    lr = _to_scalar(lr)
    # TODO: Support nonzero-dim Tensor betas, see #147921

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, min_exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )

    # We only shuffle around the beta when it is a Tensor and on CUDA, otherwise, we prefer
    # treating it as a scalar.
    beta1_dict: Optional[DeviceDict] = (  # type: ignore[attr-defined]
        {beta1.device: beta1}
        if isinstance(beta1, Tensor) and str(beta1.device) != "cpu"
        else None
    )

    for (
        device_params_,
        device_grads_,
        device_exp_avgs_,
        device_exp_avg_sqs_,
        device_min_exp_avg_sqs_,
        device_state_steps_,
    ), _ in grouped_tensors.values():
        
        device_params = cast(list[Tensor], device_params_)
        device_grads = cast(list[Tensor], device_grads_)
        device_exp_avgs = cast(list[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(list[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(list[Tensor], device_state_steps_)

        device = device_params[0].device
        if beta1_dict is not None and device not in beta1_dict:
            beta1_dict[device] = beta1.to(device=device, non_blocking=True)  # type: ignore[union-attr, attr-defined]
        device_beta1 = beta1_dict[device] if beta1_dict else beta1

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        torch._foreach_add_(device_state_steps, 1)
        # print(f"{device_state_steps = }")

        # Update the EMA of the second moment -- never includes the weight decay term!
        # Update the second moment first because first moment may need us to add the weight decay term to the gradient
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        if isinstance(beta2, torch.Tensor):
            scaled_device_grads = torch._foreach_mul(device_grads, 1 - beta2)  # type: ignore[assignment]
            value = 1.0
        else:
            scaled_device_grads = device_grads  # type: ignore[assignment]
            value = 1 - beta2
        torch._foreach_addcmul_(
            device_exp_avg_sqs, scaled_device_grads, device_grads, value
        )
        # print(f"{device_exp_avg_sqs = }")

        # If maintaining an EMA of the regularized loss -- include weight decay from prior
        if not decoupled_wd and weight_decay != 0:
            # print("not using decoupled_wd -- updating gradients")
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=weight_decay
                )

        # Update the EMA of the first moment
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - device_beta1)
        # print(f"{device_exp_avgs = }")

        # Delete the local intermediate(s) since they won't be used anymore to save on peak memory
        del device_grads
        del scaled_device_grads

        bias_correction1: Union[tuple[Tensor, ...], list[Tensor]]
        bias_correction2: Union[tuple[Tensor, ...], list[Tensor]]
        
        bias_correction1 = [
            1 - torch.as_tensor(beta1, device=device) ** _get_value(step) for step in device_state_steps
        ]
        # print(f"{bias_correction1 = }")

        # Start computing the update with denominator to reduce peak memory usage when
        #       incorporating the numerator to compute the full update
        if bias_corr:
            # print("using bias correction in the second moment computation")
            bias_correction2 = [
                1 - torch.as_tensor(beta2, device=device) ** _get_value(step) for step in device_state_steps
            ]
            denom = torch._foreach_div(device_exp_avg_sqs, bias_correction2)
            torch._foreach_add_(denom, scalar=weight_decay)
        else:
            denom = torch._foreach_add(device_exp_avg_sqs, scalar=weight_decay)
        # print(f"{denom = }")

        # This is an AMSgrad analogue -- only track the curvature dampening if beta3 is not None
        if beta3 is not None and 0. < gamma:
            print(f"dampening curvature with {beta3 = } and {gamma = }")
            device_min_exp_avg_sqs = cast(list[Tensor], device_min_exp_avg_sqs_)
            torch._foreach_mul_(device_min_exp_avg_sqs, beta3)
            torch._foreach_minimum_(device_min_exp_avg_sqs, device_exp_avg_sqs)
            torch._foreach_sub_(denom, device_min_exp_avg_sqs, alpha=gamma)
            del device_min_exp_avg_sqs
            # print(f"{denom = }")

        if 0. < perturb_rad:
            print(f"perturbing the curvature using {perturb_rad = }")
            torch._foreach_add_(
                denom,
                [perturb_rad*torch.randn_like(d) for d in denom],
            )
            torch._foreach_clamp_min_(denom, eps)
            # print(f"{denom = }")

        # If using decoupled weight decay, like in IVON, scale the parameters
        #       before adding because learning rate was scaled down for bias correction
        if decoupled_wd and weight_decay != 0:
            num = torch._foreach_addcmul(
                device_exp_avgs,
                device_params,
                bias_correction1,
                value=weight_decay,
            )
            # print(f"{num = }")
            updates = torch._foreach_div(num, denom)
            # print(f"{updates = }")
            del num
        else:
            # print("not using decoupled_wd for numerator computation")
            updates = torch._foreach_div(device_exp_avgs, denom)
        del denom

        if 0. < clip_radius < torch.inf:
            # print(f"clipping the update size with {clip_radius = }")
            torch._foreach_clamp_min_(updates, -clip_radius)
            torch._foreach_clamp_max_(updates, clip_radius)
            # print(f"{updates = }")

        if rescale_lr:
            step_size = lr * (hess_init+weight_decay)
        else:
            # print("not rescaling learning rate for step size computation")
            step_size = lr
        torch._foreach_addcdiv_(
            device_params,
            updates,
            bias_correction1,
            value=-step_size,
        )
        # print(f"{device_params = }")
        # import sys; sys.exit()


class SampleParameters(object):

    def __init__(self, model: nn.Module):
        self.model = model

    def _perturb_parameters(self, mean: Union[list[Tensor], float] = 0., std: Union[list[Tensor], float] = 1.):
        if isinstance(mean, list):
            assert all(isinstance(m, Tensor) for m in mean)
            mean = nn.utils.parameters_to_vector(mean)
        if isinstance(std, list):
            assert all(isinstance(s, Tensor) for s in std)
            std = nn.utils.parameters_to_vector(std)
        self.stash = nn.utils.parameters_to_vector(self.model.parameters())
        perturbation = torch.randn_like(self.stash)
        if True:    # mean != 0. and not (isinstance(mean, list) and torch.all(mean == 0.)):
            perturbation += mean
        if True:    # std != 1. and not (isinstance(std, list) and torch.all(std == 1.)):
            perturbation *= std
        new_params = self.stash + perturbation
        del perturbation
        nn.utils.vector_to_parameters(new_params, self.model.parameters())

    def _restore_parameters(self):
        if self.stash is not None:
            nn.utils.vector_to_parameters(self.stash, self.model.parameters())
            self.stash = None


class uCBOptSampling(SampleParameters):

    def __init__(self, model: nn.Module, optimizer: Optimizer):
        super().__init__(model)
        self.optimizer = optimizer

    @torch.no_grad()
    def __enter__(self):
        mean = 0.
        std = list()
        for group in self.optimizer.param_groups:
            wd = group.get("weight_decay", 0)
            for p in group["params"]:
                state = self.optimizer.state[p]
                if "exp_avg_sq" in state:
                    std.append(state["exp_avg_sq"]+wd)
                else:
                    std.append(torch.zeros_like(p))
        self._perturb_parameters(mean, std)

    @torch.no_grad()
    def __exit__(self, type, value, traceback):
        self._restore_parameters()