#!/bin/bash
# Generic training runner. Reads a YAML config and launches one training
# process per seed in parallel, each writing to its own savedir/stdout.log.
# Usage: run_train.sh <config.yaml>
set -euo pipefail

CONFIG_FILE="${1:?Usage: run_train.sh <config.yaml>}"

DATADIR="/scratch/$USER/cbo_results/data"
mkdir -p "$DATADIR"

python3 - "$CONFIG_FILE" "$DATADIR" <<'PYEOF'
import sys, yaml, os, subprocess
from datetime import datetime

config_file = sys.argv[1]
datadir     = sys.argv[2]

with open(config_file) as f:
    c = yaml.safe_load(f)

optimizer = c['optimizer']
dataset   = c['dataset']
model     = c['model']
seeds     = c.get('seeds', [0, 1, 2, 3, 4])
if 'device' not in c:
    raise KeyError(f"{config_file} must define device, e.g. device: cuda")
device    = c['device']
scratch   = f"/scratch/{os.environ['USER']}/cbo_results"
traindir  = f"{scratch}/{c['traindir']}"

# Derive project root from config location: <root>/configs/<file>.yaml
project_root = os.path.dirname(os.path.dirname(os.path.abspath(config_file)))
code_base    = c.get('code_base', project_root)

def to_module_name(value):
    module = value.replace('\\', '.').replace('/', '.')
    if module.endswith('.py'):
        module = module[:-3]
    if module.startswith('cbo.scripts.'):
        module = f"experiments.{module.split('.')[-1]}"
    elif module.startswith('experiments.'):
        pass
    elif '.' not in module:
        module = f"experiments.{module}"
    return module

# Map optimizer names to training modules (baseline methods share one module)
BASELINE_OPTS = {'sgd', 'adamw', 'adahessian', 'mcdrop', 'swag'}
_script_map = {opt: 'experiments.train_standard' for opt in BASELINE_OPTS}
_script_map.update(
    {
        'ivon': 'experiments.train_ivon',
        'ucbopt': 'experiments.train_ucbopt',
        'duq': 'experiments.train_duq',
        'sngp': 'experiments.train_sngp',
    }
)
script = to_module_name(c.get('train_script', _script_map.get(optimizer, f"experiments.train_{optimizer}")))

# Flag convention: 1-2 char key -> single dash (-lr), longer -> double dash (--beta1).
# Keys that already start with '-' are used verbatim (e.g. "--wd" for ucbopt_mcdrop).
def to_flag(key):
    if key.startswith('-'):
        return key
    return f"-{key}" if len(key) <= 2 else f"--{key}"

extra = []
# Standard methods: inject --optimizer flag so experiments.train_standard knows which to use
if optimizer in BASELINE_OPTS and not c.get('train_script'):
    extra += ['--optimizer', optimizer]
for key, val in c.get('train_args', {}).items():
    extra += [to_flag(key), str(val)]
for flag in c.get('train_flags', []):
    extra.append(to_flag(flag))

timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
procs = []
for seed in seeds:
    savedir = f"{traindir}/seed={seed}/{timestamp}"
    os.makedirs(savedir, exist_ok=True)
    cmd = ['python', '-u', '-m', script, model, dataset,
           '-s', str(seed), '-d', device, '-dd', datadir, '-sd', savedir] + extra
    print(f"Launching seed={seed} -> {savedir}", flush=True)
    log = open(f"{savedir}/stdout.log", 'w')
    proc = subprocess.Popen(cmd, cwd=code_base, stdout=log, stderr=log)
    procs.append((seed, proc, log))

failed = []
for seed, proc, log in procs:
    ret = proc.wait()
    log.close()
    status = "OK" if ret == 0 else f"FAILED (exit={ret})"
    print(f"  seed={seed}: {status}", flush=True)
    if ret != 0:
        failed.append(seed)

if failed:
    print(f"ERROR: seed(s) {failed} failed!", flush=True)
    sys.exit(1)
print("All training complete!", flush=True)
PYEOF
