import math
import re

DATASET_NAMES = (
    "Concrete", "Airfoil", "Real Estate Valuation", "Power Plant",
    "Energy Heating", "Auto MPG", "Automobile", "Computer Hardware",
    "Wine Red", "Diabetes",
)
METHODS = ("Adam", "BBB-Adam", "VOGN", "Vadam", "Lower CBO", "Upper CBO")
METRIC_COLUMNS = {"contamination": "RMSE", "underrepresentation": "LT-MAE"}
METHOD_LABELS = dict(zip(METHODS, ("Adam", "BBB", "VOGN", "Vadam", "lCBOpt", "uCBOpt")))
DATASET_LABELS = {"Real Estate Valuation": "Real Estate"}


def dataset_slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def repetition_seed(seed, dataset_name, repetition):
    return seed + 10000 * DATASET_NAMES.index(dataset_name) + 1000 * repetition


def normalize_numbers(value):
    if isinstance(value, dict):
        return {key: normalize_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_numbers(item) for item in value]
    if isinstance(value, str) and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
        return float(value)
    return value


def validate_config(config, experiment):
    if config.get("experiment") != experiment:
        raise ValueError(f"Expected a {experiment} config")
    names = config["datasets"]
    if not names or len(names) != len(set(names)) or any(name not in DATASET_NAMES for name in names):
        raise ValueError("datasets must contain unique supported dataset names")
    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")
    params = config["parameters"]
    common = {
        "n_pca_components", "hidden_size", "warmup_steps", "warmup_learning_rate",
        "n_update_steps", "cbo_learning_rate", "trust_radius", "prior_precision",
        "vi_initial_std", "vi_mc_samples", "bbb_learning_rate", "vogn_learning_rate",
        "vogn_beta", "vadam_learning_rate", "vadam_beta1", "vadam_beta2", "vadam_epsilon",
        "curvature_margin", "upper_precision_floor", "minimum_eigenvalue", "matrix_damping", "n_repetitions",
    }
    specific = ({"n_train", "contamination_fraction", "outlier_scale", "deterministic_adam_learning_rate"}
                if experiment == "contamination" else
                {"test_fraction", "max_train_samples", "high_risk_quantile", "high_risk_train_keep_fraction",
                 "adam_learning_rate"})
    missing = (common | specific) - set(params)
    if missing:
        raise ValueError(f"Missing experiment parameters: {sorted(missing)}")
    integer_keys = ("n_pca_components", "hidden_size", "n_update_steps", "n_repetitions", "vi_mc_samples")
    integer_keys += ("n_train",) if experiment == "contamination" else ("max_train_samples",)
    for key in integer_keys:
        if not isinstance(params[key], int) or params[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(params["warmup_steps"], int) or params["warmup_steps"] < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    if params["vi_mc_samples"] < 2 or params["vi_mc_samples"] % 2:
        raise ValueError("vi_mc_samples must be even and at least two for antithetic sampling")
    maximum_seed = repetition_seed(config["seed"], DATASET_NAMES[-1], params["n_repetitions"] - 1)
    if config["seed"] < 0 or maximum_seed >= 2**32:
        raise ValueError("The repetition seed schedule must remain in NumPy's uint32 seed range")
    for key, value in params.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    for key in ("trust_radius", "prior_precision", "vi_initial_std", "matrix_damping", "minimum_eigenvalue"):
        if params[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in params:
        if key.endswith("learning_rate") and params[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("vogn_beta", "vadam_beta1", "vadam_beta2"):
        if not 0 <= params[key] < 1:
            raise ValueError(f"{key} must lie in [0, 1)")
    if experiment == "contamination":
        if not 0 <= params["contamination_fraction"] <= 1 or params["outlier_scale"] < 0:
            raise ValueError("Invalid contamination fraction or outlier scale")
    else:
        for key in ("test_fraction", "high_risk_quantile"):
            if not 0 < params[key] < 1:
                raise ValueError(f"{key} must be between zero and one")
        if not 0 <= params["high_risk_train_keep_fraction"] <= 1:
            raise ValueError("high_risk_train_keep_fraction must lie in [0, 1]")
