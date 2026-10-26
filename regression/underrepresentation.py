import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class UnderrepresentationExperiment:
    def __init__(self, config, device, seed):
        self.config = dict(config)
        self.device = torch.device(device)
        self.seed = seed
        self.input_dimension = config["n_pca_components"]
        self.hidden_size = config["hidden_size"]
        self.n_parameters = self.input_dimension * self.hidden_size + 2 * self.hidden_size + 1

    def unpack_parameters(self, parameters):
        i = 0
        n_w1 = self.input_dimension * self.hidden_size
        w1 = parameters[i:i+n_w1].reshape(self.input_dimension, self.hidden_size)
        i += n_w1
        b1 = parameters[i:i+self.hidden_size]
        i += self.hidden_size
        w2 = parameters[i:i+self.hidden_size]
        i += self.hidden_size
        b2 = parameters[i]
        return w1, b1, w2, b2

    def network_forward(self, parameters, features):
        w1, b1, w2, b2 = self.unpack_parameters(parameters)
        return torch.tanh(features @ w1 + b1) @ w2 + b2

    def initialize_parameters(self, seed):
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        return 0.1 * torch.randn(self.n_parameters, generator=g, device=self.device)

    def prepare_split(self, raw_x, raw_y, seed):
        x_fit, x_test, y_fit_raw, y_test_raw = train_test_split(
            raw_x, raw_y, test_size=self.config["test_fraction"], random_state=seed
        )

        scaler = StandardScaler()
        x_fit_std = scaler.fit_transform(x_fit)
        x_test_std = scaler.transform(x_test)

        n_components = min(self.config["n_pca_components"], x_fit_std.shape[1], x_fit_std.shape[0])

        pca = PCA(n_components=n_components, random_state=seed)
        x_fit_pca = pca.fit_transform(x_fit_std)
        x_test_pca = pca.transform(x_test_std)

        n_missing = self.config["n_pca_components"] - n_components
        if n_missing > 0:
            x_fit_pca = np.pad(x_fit_pca, ((0, 0), (0, n_missing)))
            x_test_pca = np.pad(x_test_pca, ((0, 0), (0, n_missing)))

        target_mean = float(np.mean(y_fit_raw))
        target_std = float(np.std(y_fit_raw))
        if not np.isfinite(target_std) or target_std < 1e-12:
            target_std = 1.0

        y_fit = (y_fit_raw - target_mean) / target_std
        y_test = (y_test_raw - target_mean) / target_std
        threshold_raw = float(np.quantile(y_fit_raw, self.config["high_risk_quantile"]))
        threshold_scaled = (threshold_raw - target_mean) / target_std

        return {
            "x_fit": x_fit_pca,
            "y_fit": y_fit,
            "x_test": torch.tensor(x_test_pca, device=self.device),
            "y_test": torch.tensor(y_test, device=self.device),
            "y_test_raw": np.asarray(y_test_raw),
            "target_mean": target_mean,
            "target_std": target_std,
            "high_risk_threshold_raw": threshold_raw,
            "high_risk_threshold_scaled": threshold_scaled,
        }

    def build_underrepresented_training_data(self, raw_x, raw_y, seed):
        prepared = self.prepare_split(raw_x, raw_y, seed)
        rng = np.random.default_rng(seed + 11)
        y_fit, threshold = prepared["y_fit"], prepared["high_risk_threshold_scaled"]

        common_idx = np.flatnonzero(y_fit < threshold)
        severe_idx = np.flatnonzero(y_fit >= threshold)

        n_severe_keep = max(1, int(round(self.config["high_risk_train_keep_fraction"] * len(severe_idx))))
        severe_keep = rng.choice(severe_idx, size=min(n_severe_keep, len(severe_idx)), replace=False)

        max_train = min(int(self.config["max_train_samples"]), len(common_idx) + len(severe_keep))
        n_common_keep = max(0, max_train - len(severe_keep))
        common_keep = rng.choice(common_idx, size=min(n_common_keep, len(common_idx)), replace=False)

        train_idx = np.concatenate([common_keep, severe_keep])
        rng.shuffle(train_idx)

        return {
            **prepared,
            "x_train": torch.tensor(prepared["x_fit"][train_idx], device=self.device),
            "y_train": torch.tensor(prepared["y_fit"][train_idx], device=self.device),
            "n_train": len(train_idx),
        }

    def data_loss(self, parameters, data):
        residuals = self.network_forward(parameters, data["x_train"]) - data["y_train"]
        return 0.5 * residuals.square().mean()

    def per_example_data_losses(self, parameters, data):
        residuals = self.network_forward(parameters, data["x_train"]) - data["y_train"]
        return 0.5 * residuals.square()

    def regularized_loss(self, parameters, data):
        return self.data_loss(parameters, data) + 0.5 * self.config["prior_precision"] * parameters.square().sum()

    def inverse_softplus(self, value):
        value = torch.as_tensor(value, device=self.device)
        return torch.log(torch.expm1(value))

    def gaussian_kl_to_prior(self, mean, std):
        delta = self.config["prior_precision"]
        return 0.5 * torch.sum(delta * (mean.square() + std.square()) - 1.0 - torch.log(delta * std.square()))

    def create_antithetic_noise(self, seed):
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        half = max(1, self.config["vi_mc_samples"] // 2)
        noise = torch.randn(half, self.n_parameters, generator=g, device=self.device)
        return torch.cat([noise, -noise], dim=0)

    def negative_elbo(self, mean, rho, data, fixed_noise):
        std = torch.nn.functional.softplus(rho) + 1e-8
        samples = mean[None, :] + std[None, :] * fixed_noise
        expected_loss = torch.stack([self.data_loss(sample, data) for sample in samples]).mean()
        return expected_loss + self.gaussian_kl_to_prior(mean, std) / len(data["y_train"])

    def value_gradient_hessian(self, parameters, objective_function):
        p = parameters.detach().clone().requires_grad_(True)
        value = objective_function(p)
        gradient = torch.autograd.grad(value, p, create_graph=True)[0]
        hessian = torch.autograd.functional.hessian(objective_function, p)
        return value.detach(), gradient.detach(), (0.5 * (hessian + hessian.T)).detach()

    def clip_step(self, step):
        norm = torch.linalg.norm(step)
        return self.config["trust_radius"] * step / norm if norm > self.config["trust_radius"] else step

    def make_positive_definite(self, matrix):
        matrix = 0.5 * (matrix + matrix.T)
        eigenvalues = torch.linalg.eigvalsh(matrix)
        shift = torch.clamp(self.config["minimum_eigenvalue"] - eigenvalues.min(), min=0.0)
        return matrix + (shift + self.config["matrix_damping"]) * torch.eye(matrix.shape[0], device=self.device)

    def obtain_warm_start(self, initial_parameters, data):
        p = initial_parameters.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([p], lr=self.config["warmup_learning_rate"])
        for _ in range(self.config["warmup_steps"]):
            optimizer.zero_grad()
            self.regularized_loss(p, data).backward()
            optimizer.step()
        return p.detach()

    def exact_cbo_update(self, parameters, data, kind):
        _, gradient, hessian = self.value_gradient_hessian(parameters, lambda p: self.regularized_loss(p, data))
        identity = torch.eye(self.n_parameters, device=self.device)
        eigenvalues = torch.linalg.eigvalsh(hessian)

        if kind == "upper":
            precision = max(self.config["upper_precision_floor"], eigenvalues.min().item() - self.config["curvature_margin"])
            curvature = hessian - precision * identity
        elif kind == "lower":
            precision = eigenvalues.max().item() + self.config["curvature_margin"]
            curvature = precision * identity - hessian
        else:
            raise ValueError("kind must be 'upper' or 'lower'")

        direction = -torch.linalg.solve(self.make_positive_definite(curvature), gradient)
        return parameters + self.clip_step(self.config["cbo_learning_rate"] * direction)

    def run_cbo(self, warm_start, data, kind):
        p = warm_start.detach().clone()
        for _ in range(self.config["n_update_steps"]):
            p = self.exact_cbo_update(p, data, kind)
        return p.detach()

    def run_bbb(self, warm_start, data, seed):
        mean = warm_start.detach().clone().requires_grad_(True)
        rho = torch.full_like(mean, self.inverse_softplus(self.config["vi_initial_std"]).item(), requires_grad=True)
        fixed_noise = self.create_antithetic_noise(seed + 101)
        optimizer = torch.optim.Adam([mean, rho], lr=self.config["bbb_learning_rate"])
        for _ in range(self.config["n_update_steps"]):
            optimizer.zero_grad()
            self.negative_elbo(mean, rho, data, fixed_noise).backward()
            optimizer.step()
        return mean.detach()

    def diagonal_empirical_fisher(self, parameters, data):
        losses = self.per_example_data_losses(parameters, data)
        squared_gradients = []
        for loss_i in losses:
            grad_i = torch.autograd.grad(loss_i, parameters, retain_graph=True)[0]
            squared_gradients.append(grad_i.square())
        return torch.stack(squared_gradients).mean(dim=0)

    def run_vogn(self, warm_start, data, seed):
        mean, curvature = warm_start.detach().clone(), torch.ones_like(warm_start)
        n, delta, beta = len(data["y_train"]), self.config["prior_precision"], self.config["vogn_beta"]
        g = torch.Generator(device=self.device)
        g.manual_seed(seed + 202)

        for _ in range(self.config["n_update_steps"]):
            posterior_std = torch.rsqrt(torch.clamp(n * curvature + delta, min=1e-8))
            sampled = (mean + posterior_std * torch.randn(self.n_parameters, generator=g, device=self.device)).detach().requires_grad_(True)
            gradient = torch.autograd.grad(self.data_loss(sampled, data), sampled, create_graph=True)[0]
            fisher = self.diagonal_empirical_fisher(sampled, data).detach()
            curvature = beta * curvature + (1.0 - beta) * fisher
            direction = -(gradient.detach() + delta * mean / n) / torch.clamp(curvature + delta / n, min=1e-8)
            mean = mean + self.clip_step(self.config["vogn_learning_rate"] * direction)
        return mean.detach()

    def run_vadam(self, warm_start, data, seed):
        mean = warm_start.detach().clone()
        m, v = torch.zeros_like(mean), torch.zeros_like(mean)
        n, delta = len(data["y_train"]), self.config["prior_precision"]
        b1, b2, eps = self.config["vadam_beta1"], self.config["vadam_beta2"], self.config["vadam_epsilon"]
        g = torch.Generator(device=self.device)
        g.manual_seed(seed + 303)

        for t in range(1, self.config["n_update_steps"] + 1):
            v_prev_hat = v / (1.0 - b2 ** max(t - 1, 1)) if t > 1 else v
            precision = n * (torch.sqrt(torch.clamp(v_prev_hat, min=0.0)) + delta / n)
            posterior_std = torch.rsqrt(torch.clamp(precision, min=1e-6))
            sampled = (mean + posterior_std * torch.randn(self.n_parameters, generator=g, device=self.device)).detach().requires_grad_(True)
            gradient = torch.autograd.grad(self.data_loss(sampled, data), sampled)[0].detach() + delta * mean / n

            m = b1 * m + (1.0 - b1) * gradient
            v = b2 * v + (1.0 - b2) * gradient.square()
            m_hat, v_hat = m / (1.0 - b1 ** t), v / (1.0 - b2 ** t)
            mean = mean + self.clip_step(-self.config["vadam_learning_rate"] * m_hat / (torch.sqrt(v_hat) + eps))
        return mean.detach()

    def run_adam(self, warm_start, data):
        p = warm_start.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([p], lr=self.config["adam_learning_rate"])
        for _ in range(self.config["n_update_steps"]):
            optimizer.zero_grad()
            self.regularized_loss(p, data).backward()
            optimizer.step()
        return p.detach()

    def fit_all_methods(self, data, seed):
        start = self.obtain_warm_start(self.initialize_parameters(seed), data)
        return {
            "Adam": self.run_adam(start, data),
            "BBB-Adam": self.run_bbb(start, data, seed),
            "VOGN": self.run_vogn(start, data, seed),
            "Vadam": self.run_vadam(start, data, seed),
            "Lower CBO": self.run_cbo(start, data, "lower"),
            "Upper CBO": self.run_cbo(start, data, "upper"),
        }

    def evaluate_method(self, method_name, parameters, data):
        with torch.no_grad():
            predictions_scaled = self.network_forward(parameters, data["x_test"])
            predictions = predictions_scaled * data["target_std"] + data["target_mean"]
            targets = torch.tensor(data["y_test_raw"], device=self.device)

        errors = predictions - targets
        absolute_errors = errors.abs()
        true_high_risk = targets >= data["high_risk_threshold_raw"]
        severe_errors = absolute_errors[true_high_risk]
        if not len(severe_errors):
            raise ValueError("LT-MAE requires test targets at or above the fit-pool target quantile")

        return {
            "Method": method_name,
            "LT-MAE": severe_errors.mean().item(),
        }

    def build_data(self, raw_x, raw_y, seed):
        return self.build_underrepresented_training_data(raw_x, raw_y, seed)

    def evaluate(self, fitted, data):
        return pd.DataFrame([self.evaluate_method(name, parameters, data) for name, parameters in fitted.items()])
