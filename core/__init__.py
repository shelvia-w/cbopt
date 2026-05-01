"""Convenient re-exports for the core package public surface."""

from .engine import (
    do_epoch,
    do_trainbatch,
    do_evalbatch,
    SummaryWriter,
)

from .logging import (
    coro_log_timed,
    coro_log_metrics,
)

from .utils import (
    check_cuda,
    deterministic_run,
    mkdir,
    mkdirp,
    get_outputsaver,
    summarize_csv,
)

from .checkpoint import (
    savecheckpoint,
    loadcheckpoint,
)

from .coroutines import (
    autoinitcoroutine,
    coro_timer,
    coro_trackavg_weighted,
    coro_dict2csv,
    coro_npybatchgatherer,
)

from .metrics import (
    AUROC,
    top5corrects,
    cumconfidence,
    cumentropy,
    cumnll,
    cumbrier,
)

from .calibration import (
    bins2ece,
    bins2acc,
    bins2conf,
    data2bins,
    bins2diagram
)

from .evaluation import (
    predict_proba_batch,
    predict_proba_von,
    predict_proba_swag,
    predict_proba_duq,
    predict_proba_sngp,
)

from .ood import (
    confidence_from_prediction_npy,
    coro_epochlog_ood,
    coro_log_ood,
)

__all__ = [
    "do_epoch", "do_trainbatch", "do_evalbatch",
    "SummaryWriter",
    "coro_log_timed", "coro_log_metrics",
    "check_cuda", "deterministic_run",
    "mkdir", "mkdirp", "get_outputsaver", "summarize_csv",
    "savecheckpoint", "loadcheckpoint",
    "autoinitcoroutine", "coro_timer", "coro_trackavg_weighted",
    "coro_dict2csv", "coro_npybatchgatherer",
    "AUROC", "top5corrects", "cumconfidence", "cumentropy", "cumnll", "cumbrier",
    "bins2ece", "bins2acc", "bins2conf", "data2bins", "bins2diagram",
    "predict_proba_batch", "predict_proba_von", "predict_proba_swag",
    "predict_proba_duq", "predict_proba_sngp",
    "confidence_from_prediction_npy", "coro_epochlog_ood", "coro_log_ood",
]
