#!/bin/bash
# OOD evaluation runner. Reads a YAML config and runs OOD evaluation across all seeds.
# Output is written to both PBS stdout and <traindir>/<ood_subdir>/stdout.log (tee behaviour).
# Usage: run_ood.sh <config.yaml>
set -euo pipefail

CONFIG_FILE="${1:?Usage: run_ood.sh <config.yaml>}"

DATADIR="/scratch/$USER/cbo_results/data"
mkdir -p "$DATADIR"

python3 - "$CONFIG_FILE" "$DATADIR" <<'PYEOF'
import sys, yaml, os, subprocess

config_file = sys.argv[1]
datadir     = sys.argv[2]

with open(config_file) as f:
    c = yaml.safe_load(f)

dataset   = c['dataset']
scratch   = f"/scratch/{os.environ['USER']}/cbo_results"
traindir  = f"{scratch}/{c['traindir']}"

project_root = os.path.dirname(os.path.dirname(os.path.abspath(config_file)))
code_base    = c.get('code_base', project_root)

if 'device' not in c:
    raise KeyError(f"{config_file} must define device, e.g. device: cuda")
device     = c['device']
ood_subdir = c.get('ood_subdir', 'ood_eval')

def to_flag(key):
    if key.startswith('-'):
        return key
    return f"-{key}" if len(key) <= 2 else f"--{key}"

extra = []
for key, val in c.get('ood_args', {}).items():
    extra += [to_flag(key), str(val)]
for flag in c.get('ood_flags', []):
    extra.append(to_flag(flag))

savedir = f"{traindir}/{ood_subdir}"
os.makedirs(savedir, exist_ok=True)

cmd = ['python', '-u', '-m', 'experiments.ood',
       traindir,
       '--indomain_dataset', dataset,
       '-dd', datadir, '-sd', savedir, '-d', device] + extra

print(f"Running: {' '.join(cmd)}", flush=True)
log_path = f"{savedir}/stdout.log"

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
print("OOD evaluation complete!", flush=True)
PYEOF
