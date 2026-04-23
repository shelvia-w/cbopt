import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

class FilterResponseNorm(nn.Module):
    def __init__(self, num_filters, eps=1e-6):
        super(FilterResponseNorm, self).__init__()
        self.eps = eps
        par_shape = (1, num_filters, 1, 1)  # [1,C,1,1]
        self.tau = torch.nn.Parameter(torch.zeros(par_shape))
        self.beta = torch.nn.Parameter(torch.zeros(par_shape))
        self.gamma = torch.nn.Parameter(torch.ones(par_shape))

    def forward(self, x):
        nu2 = torch.mean(torch.square(x), dim=[2, 3], keepdim=True)
        x = x * 1 / torch.sqrt(nu2 + self.eps)
        y = self.gamma * x + self.beta
        z = torch.max(y, self.tau)
        return z

# ===================================================================
# for SNGP
# ===================================================================

class BoundedScale(nn.Module):
    def __init__(self, bound: float = 1.0):
        super().__init__()
        if bound <= 0:
            raise ValueError(f"bound must be > 0, got {bound}")
        self.bound = float(bound)

    def forward(self, gamma: Tensor) -> Tensor:
        max_abs = gamma.abs().max()
        scale = torch.clamp(max_abs / self.bound, min=1.0)
        return gamma / scale

class FilterResponseNormLipschitz(nn.Module):
    def __init__(self, num_filters: int, eps: float = 1e-6,
                 gamma_bound: float = 1.0):
        super().__init__()
        self.eps = eps
        par_shape = (1, num_filters, 1, 1)
        self.tau = nn.Parameter(torch.zeros(par_shape))
        self.beta = nn.Parameter(torch.zeros(par_shape))
        self.gamma = nn.Parameter(torch.ones(par_shape))

        parametrize.register_parametrization(
            self, "gamma", BoundedScale(bound=gamma_bound)
        )

    def forward(self, x: Tensor) -> Tensor:
        nu2 = torch.mean(torch.square(x), dim=[2, 3], keepdim=True)
        x = x * (1.0 / torch.sqrt(nu2 + self.eps))
        y = self.gamma * x + self.beta
        z = torch.max(y, self.tau)
        return z