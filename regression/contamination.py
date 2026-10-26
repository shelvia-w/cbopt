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

    def clean_training_loss(self, parameters, data):
        residuals = self.network_forward(parameters, data['x_train']) - data['y_train_clean']
        return 0.5 * torch.mean(residuals.square())

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
        return matrix + (shift + self.config['matrix_damping']) * identity, eigenvalues, float(shift.item())

    def local_surrogate_change(self, gradient, curvature, step):
        return float((torch.dot(gradient, step) + 0.5 * torch.dot(step, curvature @ step)).item())

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
        objective, data_gradient, data_hessian = self.data_loss_derivatives(parameters, data)
        delta, identity = self.config['prior_precision'], torch.eye(self.n_parameters, device=self.device)
        regularized_gradient, regularized_hessian = data_gradient + delta * parameters, data_hessian + delta * identity
        smallest_eigenvalue = torch.linalg.eigvalsh(regularized_hessian).min().item()
        candidate_precision = max(self.config['upper_precision_floor'], smallest_eigenvalue - self.config['curvature_margin']) if smallest_eigenvalue > self.config['curvature_margin'] else self.config['upper_precision_floor']
        raw_curvature = regularized_hessian - candidate_precision * identity
        raw_eigenvalues = torch.linalg.eigvalsh(raw_curvature)
        effective_curvature, _, shift = self.make_positive_definite(raw_curvature)
        step = self.clip_step(self.config['cbo_learning_rate'] * -torch.linalg.solve(effective_curvature, regularized_gradient))
        return {'step': step, 'gradient': regularized_gradient, 'curvature': effective_curvature,
                'candidate_precision': candidate_precision,
                'raw_condition_valid': bool(raw_eigenvalues.min().item() > self.config['minimum_eigenvalue']),
                'matrix_shift': shift, 'objective': objective.item(),
                'raw_min_eigenvalue': raw_eigenvalues.min().item(),
                'raw_max_eigenvalue': raw_eigenvalues.max().item()}

    def lower_cbo_update(self, parameters, data):
        objective, data_gradient, data_hessian = self.data_loss_derivatives(parameters, data)
        delta, identity = self.config['prior_precision'], torch.eye(self.n_parameters, device=self.device)
        regularized_gradient, regularized_hessian = data_gradient + delta * parameters, data_hessian + delta * identity
        candidate_precision = torch.linalg.eigvalsh(regularized_hessian).max().item() + self.config['curvature_margin']
        raw_signed_curvature = -(regularized_hessian - candidate_precision * identity)
        raw_eigenvalues = torch.linalg.eigvalsh(raw_signed_curvature)
        effective_curvature, _, shift = self.make_positive_definite(raw_signed_curvature)
        step = self.clip_step(self.config['cbo_learning_rate'] * -torch.linalg.solve(effective_curvature, regularized_gradient))
        return {'step': step, 'gradient': regularized_gradient, 'curvature': effective_curvature,
                'candidate_precision': candidate_precision,
                'raw_condition_valid': bool(raw_eigenvalues.min().item() > self.config['minimum_eigenvalue']),
                'matrix_shift': shift, 'objective': objective.item(),
                'raw_min_eigenvalue': raw_eigenvalues.min().item(),
                'raw_max_eigenvalue': raw_eigenvalues.max().item()}

    def run_cbo_method(self, method_name, warm_start, data):
        parameters, records = warm_start.detach().clone(), []
        for step_index in range(self.config['n_update_steps']):
            result = self.cbo_updates[method_name](parameters, data)
            step, new_parameters = result['step'], parameters + result['step']
            local_change = self.local_surrogate_change(result['gradient'], result['curvature'], step)
            records.append({'Method': method_name, 'Step': step_index, 'Objective': result['objective'],
                            'Step norm': torch.linalg.norm(step).item(), 'Gradient norm': torch.linalg.norm(result['gradient']).item(),
                            'Posterior mean std': np.nan, 'Posterior max std': np.nan,
                            'Candidate precision': result['candidate_precision'],
                            'Raw curvature valid': result['raw_condition_valid'], 'Matrix shift': result['matrix_shift'],
                            'Raw minimum eigenvalue': result['raw_min_eigenvalue'], 'Raw maximum eigenvalue': result['raw_max_eigenvalue'],
                            'Local surrogate change': local_change, 'Local surrogate improved': local_change < 0,
                            'Contaminated train loss': self.regularized_loss(new_parameters, data).item(),
                            'Clean train loss': self.clean_training_loss(new_parameters, data).item()})
            parameters = new_parameters
        return {'parameters': parameters, 'posterior_std': None, 'records': pd.DataFrame(records)}

    def run_bbb_adam(self, warm_start, data):
        """Mean-field Gaussian VI with reparameterized Adam updates."""
        mean = warm_start.detach().clone().requires_grad_(True)
        rho = torch.full_like(mean, self.inverse_softplus(self.config['vi_initial_std']).item(), requires_grad=True)
        optimizer, records = torch.optim.Adam([mean, rho], lr=self.config['bbb_learning_rate']), []
        for step_index in range(self.config['n_update_steps']):
            old_mean = mean.detach().clone()
            optimizer.zero_grad()
            objective = self.negative_elbo(mean, rho, data)
            objective.backward()
            gradient_norm = torch.sqrt(sum(parameter.grad.square().sum() for parameter in [mean, rho])).item()
            optimizer.step()
            std, step = (torch.nn.functional.softplus(rho) + 1e-8).detach(), mean.detach() - old_mean
            records.append({'Method': 'BBB-Adam', 'Step': step_index, 'Objective': objective.item(),
                            'Step norm': torch.linalg.norm(step).item(), 'Gradient norm': gradient_norm,
                            'Posterior mean std': std.mean().item(), 'Posterior max std': std.max().item(),
                            'Candidate precision': float((1.0 / std.square()).mean().item()),
                            'Raw curvature valid': np.nan, 'Matrix shift': np.nan,
                            'Raw minimum eigenvalue': np.nan, 'Raw maximum eigenvalue': np.nan,
                            'Local surrogate change': np.nan, 'Local surrogate improved': np.nan,
                            'Contaminated train loss': self.regularized_loss(mean.detach(), data).item(),
                            'Clean train loss': self.clean_training_loss(mean.detach(), data).item()})
        return {'parameters': mean.detach().clone(), 'posterior_std': std, 'records': pd.DataFrame(records)}

    def diagonal_empirical_fisher(self, parameters, data):
        """Average squared per-example gradients for the diagonal Fisher estimate."""
        squared_gradients = []
        for loss_i in self.per_example_data_losses(parameters, data):
            gradient_i = torch.autograd.grad(loss_i, parameters, retain_graph=True)[0]
            squared_gradients.append(gradient_i.square())
        return torch.stack(squared_gradients).mean(dim=0)

    def run_vogn(self, warm_start, data):
        """Diagonal VOGN-style update."""
        mean, curvature, records = warm_start.detach().clone(), torch.ones_like(warm_start), []
        n, delta, beta = len(data['y_train']), self.config['prior_precision'], self.config['vogn_beta']
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + 123)
        for step_index in range(self.config['n_update_steps']):
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
            posterior_precision = n * curvature + delta
            posterior_std = torch.rsqrt(torch.clamp(posterior_precision, min=1e-8))
            objective = self.data_loss(mean, data) + self.gaussian_kl_to_prior(mean, posterior_std) / n
            records.append({'Method': 'VOGN', 'Step': step_index, 'Objective': objective.item(),
                            'Step norm': torch.linalg.norm(step).item(), 'Gradient norm': torch.linalg.norm(gradient).item(),
                            'Posterior mean std': posterior_std.mean().item(), 'Posterior max std': posterior_std.max().item(),
                            'Candidate precision': posterior_precision.mean().item(),
                            'Raw curvature valid': bool(torch.all(curvature > 0).item()), 'Matrix shift': 0.0,
                            'Raw minimum eigenvalue': curvature.min().item(), 'Raw maximum eigenvalue': curvature.max().item(),
                            'Local surrogate change': np.nan, 'Local surrogate improved': np.nan,
                            'Contaminated train loss': self.regularized_loss(mean, data).item(),
                            'Clean train loss': self.clean_training_loss(mean, data).item()})
        return {'parameters': mean.detach().clone(), 'posterior_std': posterior_std.detach().clone(), 'records': pd.DataFrame(records)}

    def run_vadam(self, warm_start, data):
        """Vadam-style noisy Adam with a diagonal Gaussian posterior."""
        mean, first_moment, second_moment = warm_start.detach().clone(), torch.zeros_like(warm_start), torch.zeros_like(warm_start)
        n, delta = len(data['y_train']), self.config['prior_precision']
        beta1, beta2, eps = self.config['vadam_beta1'], self.config['vadam_beta2'], self.config['vadam_epsilon']
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + 456)
        records = []
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
            posterior_precision = n * (torch.sqrt(corrected_second) + delta / n)
            posterior_std = torch.rsqrt(torch.clamp(posterior_precision, min=1e-6))
            objective = self.data_loss(mean, data) + self.gaussian_kl_to_prior(mean, posterior_std) / n
            records.append({'Method': 'Vadam', 'Step': step_index - 1, 'Objective': objective.item(),
                            'Step norm': torch.linalg.norm(step).item(), 'Gradient norm': torch.linalg.norm(gradient).item(),
                            'Posterior mean std': posterior_std.mean().item(), 'Posterior max std': posterior_std.max().item(),
                            'Candidate precision': posterior_precision.mean().item(),
                            'Raw curvature valid': bool(torch.all(corrected_second >= 0).item()), 'Matrix shift': 0.0,
                            'Raw minimum eigenvalue': corrected_second.min().item(), 'Raw maximum eigenvalue': corrected_second.max().item(),
                            'Local surrogate change': np.nan, 'Local surrogate improved': np.nan,
                            'Contaminated train loss': self.regularized_loss(mean, data).item(),
                            'Clean train loss': self.clean_training_loss(mean, data).item()})
        return {'parameters': mean.detach().clone(), 'posterior_std': posterior_std.detach().clone(), 'records': pd.DataFrame(records)}

    def run_deterministic_adam(self, warm_start, data):
        """Adam reference without parameter sampling."""
        parameters = warm_start.detach().clone().requires_grad_(True)
        optimizer, records = torch.optim.Adam([parameters], lr=self.config['deterministic_adam_learning_rate']), []
        for step_index in range(self.config['n_update_steps']):
            old_parameters = parameters.detach().clone()
            optimizer.zero_grad()
            loss = self.regularized_loss(parameters, data)
            loss.backward()
            gradient_norm = torch.linalg.norm(parameters.grad).item()
            optimizer.step()
            step = parameters.detach() - old_parameters
            records.append({'Method': 'Adam', 'Step': step_index, 'Objective': loss.item(),
                            'Step norm': torch.linalg.norm(step).item(), 'Gradient norm': gradient_norm,
                            'Posterior mean std': np.nan, 'Posterior max std': np.nan, 'Candidate precision': np.nan,
                            'Raw curvature valid': np.nan, 'Matrix shift': np.nan,
                            'Raw minimum eigenvalue': np.nan, 'Raw maximum eigenvalue': np.nan,
                            'Local surrogate change': np.nan, 'Local surrogate improved': np.nan,
                            'Contaminated train loss': self.regularized_loss(parameters, data).item(),
                            'Clean train loss': self.clean_training_loss(parameters, data).item()})
        return {'parameters': parameters.detach().clone(), 'posterior_std': None, 'records': pd.DataFrame(records)}

    def fit_all_methods(self, data, seed):
        initial_parameters = self.initialize_parameters(seed)
        warm_start = self.obtain_warm_start(initial_parameters, data)
        fitted = {name: runner(warm_start, data) for name, runner in self.vi_runners.items()}
        fitted.update({name: self.run_cbo_method(name, warm_start, data) for name in self.cbo_updates})
        fitted['Adam'], fitted['warm_start'] = self.run_deterministic_adam(warm_start, data), warm_start
        return fitted

    def evaluate_method(self, method_name, fitted_method, data):
        parameters, posterior_std, records = fitted_method['parameters'], fitted_method['posterior_std'], fitted_method['records']
        with torch.no_grad():
            errors = self.network_forward(parameters, data['x_test']) - data['y_test']
        curvature = records['Raw curvature valid'].dropna()
        improvement = records['Local surrogate improved'].dropna()
        shifts = records['Matrix shift'].dropna()
        return {'Method': method_name,
                'Test RMSE': torch.sqrt(torch.mean(errors.square())).item() * data['target_std'],
                'Test MAE': torch.mean(torch.abs(errors)).item() * data['target_std'],
                'Test 95th error': torch.quantile(torch.abs(errors), 0.95).item() * data['target_std'],
                'Posterior mean std': posterior_std.mean().item() if posterior_std is not None else np.nan,
                'Posterior max std': posterior_std.max().item() if posterior_std is not None else np.nan,
                'Raw curvature valid (%)': 100 * curvature.astype(float).mean() if len(curvature) else np.nan,
                'Local surrogate improved (%)': 100 * improvement.astype(float).mean() if len(improvement) else np.nan,
                'Mean matrix shift': shifts.mean() if len(shifts) else np.nan}

    def evaluate_all_methods(self, fitted, data):
        return pd.DataFrame([self.evaluate_method(method, fitted[method], data) for method in self.all_methods])

    def build_data(self, raw_x, raw_y, seed):
        return self.build_experiment_data(raw_x, raw_y, self.config["contamination_fraction"], "targeted", seed)

    def evaluate(self, fitted, data):
        return self.evaluate_all_methods(fitted, data)
