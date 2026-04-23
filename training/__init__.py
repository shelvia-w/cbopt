"""Convenient re-exports for the training package public surface."""

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
from data.data_utils import corrupt_labels
from .utils import (
    check_cuda,
    deterministic_run,
    coro_timer,
    mkdir,
    mkdirp,
    get_outputsaver,
    savecheckpoint,
    loadcheckpoint,
    summarize_csv,
)
from .coroutines import (
    autoinitcoroutine,
    coro_trackavg_weighted,
    coro_dict2csv,
    coro_npybatchgatherer,
)
from .metrics import (
    AUROC,
    top5corrects,
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
    do_evalbatch_ood,
    do_evalbatch_von,
    do_evalbatch_swag,
    do_evalbatch_duq,
    do_evalbatch_sngp,
    coro_log_ood,
)

__all__ = [
    "do_epoch", "do_trainbatch", "do_evalbatch",
    "SummaryWriter",
    "coro_log_timed", "coro_log_metrics",
    "check_cuda", "deterministic_run",
    "coro_timer", "mkdir", "mkdirp", "get_outputsaver",
    "savecheckpoint", "loadcheckpoint",
    "corrupt_labels", "summarize_csv",
    "autoinitcoroutine", "coro_trackavg_weighted",
    "coro_dict2csv", "coro_npybatchgatherer",
    "AUROC", "top5corrects", "cumentropy", "cumnll", "cumbrier",
    "bins2ece", "bins2acc", "bins2conf", "data2bins", "bins2diagram",
    "do_evalbatch_ood", "do_evalbatch_von", "do_evalbatch_swag",
    "do_evalbatch_duq", "do_evalbatch_sngp", "coro_log_ood",
]
