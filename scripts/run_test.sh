#!/bin/bash
# Generic evaluation runner. Reads a YAML config and evaluates a trained model.
# Output is written to both PBS stdout and <traindir>/eval/stdout.log (tee behaviour).
# Usage: run_test.sh <config.yaml>
set -euo pipefail

CONFIG_FILE="${1:?Usage: run_test.sh <config.yaml>}"

DATADIR="/scratch/$USER/cbo_results/data"
mkdir -p "$DATADIR"

python3 - "$CONFIG_FILE" "$DATADIR" <<'PYEOF'
import sys, yaml, os, subprocess

config_file = sys.argv[1]
datadir     = sys.argv[2]

with open(config_file) as f:
    c = yaml.safe_load(f)

optimizer   = c['optimizer']
dataset     = c['dataset']
scratch     = f"/scratch/{os.environ['USER']}/cbo_results"
traindir    = f"{scratch}/{c['traindir']}"

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

# DUQ has its own test module (different positional signature); everything else uses experiments.test
_test_script_map = {'duq': 'experiments.test_duq'}
script      = to_module_name(c.get('test_script', _test_script_map.get(optimizer, 'experiments.test')))
seed_start  = c.get('seed_start', 0)
seed_end    = c.get('seed_end', 4)
device      = c.get('device', 'cuda')
eval_subdir = c.get('eval_subdir', 'eval')

def to_flag(key):
    if key.startswith('-'):
        return key
    return f"-{key}" if len(key) <= 2 else f"--{key}"

extra = []
for key, val in c.get('test_args', {}).items():
    extra += [to_flag(key), str(val)]
for flag in c.get('test_flags', []):
    extra.append(to_flag(flag))

savedir = f"{traindir}/{eval_subdir}"
os.makedirs(savedir, exist_ok=True)

# Some test scripts (e.g. duq) take MODEL as a 2nd positional: TRAINDIR MODEL DATASET
model = c.get('model', '')
positionals = [traindir, model, dataset] if c.get('test_positional_model', False) else [traindir, dataset]

cmd = ['python', '-u', '-m', script] + positionals + [
      '-dd', datadir, '-sd', savedir,
      '-d', device, '-ss', str(seed_start), '-se', str(seed_end)] + extra

print(f"Running: {' '.join(cmd)}", flush=True)
log_path = f"{savedir}/stdout.log"

# Tee output to both this process's stdout (captured by PBS log) and the savedir log.
with open(log_path, 'a') as log:
    p = subprocess.Popen(cmd, cwd=code_base,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()
        log.write(line.decode(errors='replace'))
    p.wait()

print(f"Saved to: {savedir}", flush=True)
if p.returncode != 0:
    sys.exit(p.returncode)
print("Evaluation complete!", flush=True)
PYEOF
