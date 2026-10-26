import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ContaminationExperiment:
    def __init__(self, config, device, seed):
        self.config = dict(config)
        self.device = torch.device(device)
        self.seed = seed
        self.input_dimension = config["n_pca_components"]
        self.hidden_size = config["hidden_size"]
        self.n_parameters = self.input_dimension * self.hidden_size + 2 * self.hidden_size + 1
        self.vi_noise = self.create_antithetic_noise()
        self.cbo_updates = {"Upper CBO": self.upper_cbo_update, "Lower CBO": self.lower_cbo_update}
        self.vi_runners = {"BBB-Adam": self.run_bbb_adam, "VOGN": self.run_vogn, "Vadam": self.run_vadam}
        self.all_methods = ["BBB-Adam", "VOGN", "Vadam", "Upper CBO", "Lower CBO", "Adam"]

    def unpack_parameters(self, parameters):
        index, first_layer_size = 0, self.input_dimension * self.hidden_size
        weight_input_hidden = parameters[index:index + first_layer_size].reshape(self.input_dimension, self.hidden_size)
        index += first_layer_size
        bias_hidden = parameters[index:index + self.hidden_size]
        index += self.hidden_size
        weight_hidden_output = parameters[index:index + self.hidden_size]
        index += self.hidden_size
        return weight_input_hidden, bias_hidden, weight_hidden_output, parameters[index]

    def network_forward(self, parameters, features):
        weight_input_hidden, bias_hidden, weight_hidden_output, bias_output = self.unpack_parameters(parameters)
        return torch.tanh(features @ weight_input_hidden + bias_hidden) @ weight_hidden_output + bias_output

    def initialize_parameters(self, seed):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        return 0.1 * torch.randn(self.n_parameters, generator=generator, device=self.device)

    def prepare_split(self, raw_x, raw_y, seed):
        x_fit, x_holdout, y_fit, y_holdout = train_test_split(
            raw_x, raw_y, test_size=0.40, random_state=seed
        )

        feature_scaler = StandardScaler()
        x_fit_standardized = feature_scaler.fit_transform(x_fit)
        x_holdout_standardized = feature_scaler.transform(x_holdout)

        n_components = min(self.config["n_pca_components"], x_fit_standardized.shape[1], x_fit_standardized.shape[0])
        pca = PCA(n_components=n_components, random_state=seed)
        x_fit_pca = pca.fit_transform(x_fit_standardized)
        x_holdout_pca = pca.transform(x_holdout_standardized)

        n_missing = self.config["n_pca_components"] - n_components
        if n_missing > 0:
            x_fit_pca = np.pad(x_fit_pca, ((0, 0), (0, n_missing)))
            x_holdout_pca = np.pad(x_holdout_pca, ((0, 0), (0, n_missing)))

        target_mean = float(np.mean(y_fit))
        target_std = float(np.std(y_fit))
        if not np.isfinite(target_std) or target_std < 1e-12:
            target_std = 1.0

        return {
            "x_fit": x_fit_pca,
            "y_fit": (y_fit - target_mean) / target_std,
            "x_holdout": torch.tensor(x_holdout_pca, device=self.device),
            "y_holdout": torch.tensor((y_holdout - target_mean) / target_std, device=self.device),
            "target_mean": target_mean,
            "target_std": target_std,
            "actual_pca_components": n_components,
        }

    def choose_train_test(self, prepared, seed):
        rng = np.random.default_rng(seed)
        n_train = min(int(self.config["n_train"]), len(prepared["x_fit"]))
        train_indices = rng.choice(len(prepared["x_fit"]), size=n_train, replace=False)
        x_train, y_train = prepared["x_fit"][train_indices], prepared["y_fit"][train_indices]
        return {
            "x_train": torch.tensor(x_train, device=self.device),
            "y_train_clean": torch.tensor(y_train, device=self.device),
            "x_test": prepared["x_holdout"],
            "y_test": prepared["y_holdout"],
            "train_feature_norm": np.linalg.norm(x_train, axis=1),
            "target_mean": prepared["target_mean"],
            "target_std": prepared["target_std"],
            "actual_pca_components": prepared["actual_pca_components"],
            "n_train": n_train,
        }

    def contaminate_targets(self, clean_targets, feature_norm, contamination_fraction, contamination_type, seed):
        """Shift targets at random locations or high-norm training examples."""
        rng, contaminated = np.random.default_rng(seed), clean_targets.detach().clone()
        n_outliers = int(round(contamination_fraction * len(contaminated)))
        if n_outliers == 0:
            return contaminated, np.array([], dtype=int)
        if contamination_type == 'random':
            indices = rng.choice(len(contaminated), size=n_outliers, replace=False)
        elif contamination_type == 'targeted':
            indices = np.argsort(feature_norm)[-n_outliers:]
        else:
            raise ValueError("contamination_type must be 'random' or 'targeted'.")
        shifts = rng.choice([-1.0, 1.0], size=n_outliers) * self.config['outlier_scale']
        contaminated[torch.tensor(indices, device=self.device)] += torch.tensor(shifts, device=self.device)
        return contaminated, indices

    def build_experiment_data(self, raw_x, raw_y, contamination_fraction, contamination_type, seed):
        prepared = self.prepare_split(raw_x, raw_y, seed)
        data = self.choose_train_test(prepared, seed + 10)
        contaminated_targets, outlier_indices = self.contaminate_targets(
            data["y_train_clean"], data["train_feature_norm"], contamination_fraction, contamination_type, seed + 20
        )
        data.update({
            "y_train": contaminated_targets,
            "outlier_indices": outlier_indices,
            "contamination_fraction": contamination_fraction,
            "contamination_type": contamination_type,
        })
        return data

    def data_loss(self, parameters, data):
        residuals = self.network_forward(parameters, data['x_train']) - data['y_train']
        return 0.5 * torch.mean(residuals.square())

    def per_example_data_losses(self, parameters, data):
        residuals = self.network_forward(parameters, data['x_train']) - data['y_train']
        return 0.5 * residuals.square()

    def regularized_loss(self, parameters, data):
        return self.data_loss(parameters, data) + 0.5 * self.config['prior_precision'] * torch.sum(parameters.square())

    def inverse_softplus(self, value):
        value = torch.as_tensor(value, dtype=torch.get_default_dtype(), device=self.device)
        return torch.log(torch.expm1(value))

    def gaussian_kl_to_prior(self, mean, std):
        """KL[N(mean,diag(std^2)) || N(0, prior_precision^-1 I)]."""
        delta = self.config['prior_precision']
        return 0.5 * torch.sum(delta * (mean.square() + std.square()) - 1.0 - torch.log(delta * std.square()))

    def create_antithetic_noise(self, seed=12345):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        half = self.config['vi_mc_samples'] // 2
        noise = torch.randn(half, self.n_parameters, generator=generator, device=self.device)
        return torch.cat([noise, -noise], dim=0)

    def negative_elbo(self, mean, rho, data):
        """Negative ELBO using average data loss plus KL divided by training-set size."""
        std = torch.nn.functional.softplus(rho) + 1e-8
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * self.vi_noise
        expected_data_loss = torch.stack([self.data_loss(sample, data) for sample in samples]).mean()
        return expected_data_loss + self.gaussian_kl_to_prior(mean, std) / len(data['y_train'])

    def value_gradient_hessian(self, parameters, objective_function):
        local_parameters = parameters.detach().clone().requires_grad_(True)
        objective = objective_function(local_parameters)
        gradient = torch.autograd.grad(objective, local_parameters, create_graph=True)[0]
        hessian = torch.autograd.functional.hessian(objective_function, local_parameters)
        return objective.detach(), gradient.detach(), (0.5 * (hessian + hessian.T)).detach()

    def data_loss_derivatives(self, parameters, data):
        return self.value_gradient_hessian(parameters, lambda current: self.data_loss(current, data))

    def clip_step(self, step):
        norm = torch.linalg.norm(step)
        return self.config['trust_radius'] * step / norm if norm > self.config['trust_radius'] else step

    def make_positive_definite(self, matrix):
        matrix = 0.5 * (matrix + matrix.T)
        eigenvalues = torch.linalg.eigvalsh(matrix)
        shift = torch.clamp(self.config['minimum_eigenvalue'] - eigenvalues.min(), min=0.0)
        identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        return matrix + (shift + self.config['matrix_damping']) * identity

    def obtain_warm_start(self, initial_parameters, data):
        parameters = initial_parameters.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([parameters], lr=self.config['warmup_learning_rate'])
        for _ in range(self.config['warmup_steps']):
            optimizer.zero_grad()
            objective = self.regularized_loss(parameters, data)
            objective.backward()
            optimizer.step()
        return parameters.detach().clone()

    def upper_cbo_update(self, parameters, data):
        _, data_gradient, data_hessian = self.data_loss_derivatives(parameters, data)
        delta, identity = self.config['prior_precision'], torch.eye(self.n_parameters, device=self.device)
        regularized_gradient, regularized_hessian = data_gradient + delta * parameters, data_hessian + delta * identity
        smallest_eigenvalue = torch.linalg.eigvalsh(regularized_hessian).min().item()
        candidate_precision = max(self.config['upper_precision_floor'], smallest_eigenvalue - self.config['curvature_margin']) if smallest_eigenvalue > self.config['curvature_margin'] else self.config['upper_precision_floor']
        raw_curvature = regularized_hessian - candidate_precision * identity
        effective_curvature = self.make_positive_definite(raw_curvature)
        step = self.clip_step(self.config['cbo_learning_rate'] * -torch.linalg.solve(effective_curvature, regularized_gradient))
        return step

    def lower_cbo_update(self, parameters, data):
        _, data_gradient, data_hessian = self.data_loss_derivatives(parameters, data)
        delta, identity = self.config['prior_precision'], torch.eye(self.n_parameters, device=self.device)
        regularized_gradient, regularized_hessian = data_gradient + delta * parameters, data_hessian + delta * identity
        candidate_precision = torch.linalg.eigvalsh(regularized_hessian).max().item() + self.config['curvature_margin']
        raw_signed_curvature = -(regularized_hessian - candidate_precision * identity)
        effective_curvature = self.make_positive_definite(raw_signed_curvature)
        step = self.clip_step(self.config['cbo_learning_rate'] * -torch.linalg.solve(effective_curvature, regularized_gradient))
        return step

    def run_cbo_method(self, method_name, warm_start, data):
        parameters = warm_start.detach().clone()
        for _ in range(self.config['n_update_steps']):
            step = self.cbo_updates[method_name](parameters, data)
            parameters = parameters + step
        return parameters

    def run_bbb_adam(self, warm_start, data):
        """Mean-field Gaussian VI with reparameterized Adam updates."""
        mean = warm_start.detach().clone().requires_grad_(True)
        rho = torch.full_like(mean, self.inverse_softplus(self.config['vi_initial_std']).item(), requires_grad=True)
        optimizer = torch.optim.Adam([mean, rho], lr=self.config['bbb_learning_rate'])
        for _ in range(self.config['n_update_steps']):
            optimizer.zero_grad()
            objective = self.negative_elbo(mean, rho, data)
            objective.backward()
            optimizer.step()
        return mean.detach().clone()

    def diagonal_empirical_fisher(self, parameters, data):
        """Average squared per-example gradients for the diagonal Fisher estimate."""
        squared_gradients = []
        for loss_i in self.per_example_data_losses(parameters, data):
            gradient_i = torch.autograd.grad(loss_i, parameters, retain_graph=True)[0]
            squared_gradients.append(gradient_i.square())
        return torch.stack(squared_gradients).mean(dim=0)

    def run_vogn(self, warm_start, data):
        """Diagonal VOGN-style update."""
        mean, curvature = warm_start.detach().clone(), torch.ones_like(warm_start)
        n, delta, beta = len(data['y_train']), self.config['prior_precision'], self.config['vogn_beta']
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + 123)
        for _ in range(self.config['n_update_steps']):
            posterior_precision = n * curvature + delta
            posterior_std = torch.rsqrt(torch.clamp(posterior_precision, min=1e-8))
            epsilon = torch.randn(self.n_parameters, generator=generator, device=self.device)
            sampled = (mean + posterior_std * epsilon).detach().requires_grad_(True)
            loss = self.data_loss(sampled, data)
            gradient = torch.autograd.grad(loss, sampled, create_graph=True)[0]
            fisher_diagonal = self.diagonal_empirical_fisher(sampled, data).detach()
            curvature = beta * curvature + (1.0 - beta) * fisher_diagonal
            natural_denominator = curvature + delta / n
            direction = -(gradient.detach() + delta * mean / n) / torch.clamp(natural_denominator, min=1e-8)
            step = self.clip_step(self.config['vogn_learning_rate'] * direction)
            mean = mean + step
        return mean.detach().clone()

    def run_vadam(self, warm_start, data):
        """Vadam-style noisy Adam with a diagonal Gaussian posterior."""
        mean, first_moment, second_moment = warm_start.detach().clone(), torch.zeros_like(warm_start), torch.zeros_like(warm_start)
        n, delta = len(data['y_train']), self.config['prior_precision']
        beta1, beta2, eps = self.config['vadam_beta1'], self.config['vadam_beta2'], self.config['vadam_epsilon']
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + 456)
        for step_index in range(1, self.config['n_update_steps'] + 1):
            corrected_second = second_moment / (1.0 - beta2 ** max(step_index - 1, 1)) if step_index > 1 else second_moment
            posterior_precision = n * (torch.sqrt(torch.clamp(corrected_second, min=0.0)) + delta / n)
            posterior_std = torch.rsqrt(torch.clamp(posterior_precision, min=1e-6))
            epsilon = torch.randn(self.n_parameters, generator=generator, device=self.device)
            sampled = (mean + posterior_std * epsilon).detach().requires_grad_(True)
            loss = self.data_loss(sampled, data)
            gradient = torch.autograd.grad(loss, sampled)[0].detach() + delta * mean / n
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient.square()
            corrected_first, corrected_second = first_moment / (1.0 - beta1 ** step_index), second_moment / (1.0 - beta2 ** step_index)
            direction = -corrected_first / (torch.sqrt(corrected_second) + eps)
            step = self.clip_step(self.config['vadam_learning_rate'] * direction)
            mean = mean + step
        return mean.detach().clone()

    def run_deterministic_adam(self, warm_start, data):
        """Adam reference without parameter sampling."""
        parameters = warm_start.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([parameters], lr=self.config['deterministic_adam_learning_rate'])
        for _ in range(self.config['n_update_steps']):
            optimizer.zero_grad()
            loss = self.regularized_loss(parameters, data)
            loss.backward()
            optimizer.step()
        return parameters.detach().clone()

    def fit_all_methods(self, data, seed):
        initial_parameters = self.initialize_parameters(seed)
        warm_start = self.obtain_warm_start(initial_parameters, data)
        fitted = {name: runner(warm_start, data) for name, runner in self.vi_runners.items()}
        fitted.update({name: self.run_cbo_method(name, warm_start, data) for name in self.cbo_updates})
        fitted['Adam'] = self.run_deterministic_adam(warm_start, data)
        return fitted

    def evaluate_method(self, method_name, parameters, data):
        with torch.no_grad():
            errors = self.network_forward(parameters, data['x_test']) - data['y_test']
        return {'Method': method_name,
                'RMSE': torch.sqrt(torch.mean(errors.square())).item() * data['target_std']}

    def evaluate_all_methods(self, fitted, data):
        return pd.DataFrame([self.evaluate_method(method, fitted[method], data) for method in self.all_methods])

    def build_data(self, raw_x, raw_y, seed):
        return self.build_experiment_data(raw_x, raw_y, self.config["contamination_fraction"], "targeted", seed)

    def evaluate(self, fitted, data):
        return self.evaluate_all_methods(fitted, data)
