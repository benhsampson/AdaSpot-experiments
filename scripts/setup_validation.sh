#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
mkdir -p artifacts validation/results
if [ ! -x .venv/bin/python ]; then
    uv venv --system-site-packages .venv
fi
# uv does not count inherited system packages when resolving/installing.
# Resolve once, then install only locked versions absent from this interpreter.
.venv/bin/python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
missing = []
for line in Path('validation/requirements.lock').read_text().splitlines():
    if not line or line.startswith('#') or line.startswith(' '):
        continue
    name, expected = line.split('==')
    try:
        actual = version(name)
    except PackageNotFoundError:
        actual = None
    if actual != expected:
        missing.append(line)
Path('artifacts/install-requirements.txt').write_text('\n'.join(missing) + '\n')
PY
uv pip install --python .venv/bin/python --no-deps --torch-backend cu128 -r artifacts/install-requirements.txt
.venv/bin/python - <<'PY'
import importlib.metadata as m
import json, platform, subprocess
import torch, torchvision, timm, cv2
from pathlib import Path
from packaging.requirements import Requirement
errors=[]
for line in Path('validation/requirements.lock').read_text().splitlines():
    if not line or line.startswith(('#',' ')): continue
    name, expected=line.split('==')
    assert m.version(name)==expected, (name,m.version(name),expected)
    for text in m.requires(name) or []:
        req=Requirement(text)
        if req.marker and not req.marker.evaluate(): continue
        try: actual=m.version(req.name)
        except m.PackageNotFoundError: errors.append(f'{name}: missing {req}'); continue
        if req.specifier and actual not in req.specifier: errors.append(f'{name}: {req} but {actual}')
assert not errors, errors
assert torch.cuda.is_available()
x=torch.randn(32,32,device='cuda'); assert torch.isfinite(x@x).all()
report={'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'repository_status':subprocess.check_output(['git','status','--short'],text=True),
 'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
 'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
 'gpu_bytes':torch.cuda.get_device_properties(0).total_memory,'cuda_arches':torch.cuda.get_arch_list(),
 'packages':{name:m.version(name) for name in sorted({d.metadata['Name'] for d in m.distributions()})},
 'package_locations':{name:str(m.distribution(name).locate_file('')) for name in ['torch','torchvision','numpy','timm','opencv-python']},
 'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),
 'lscpu':subprocess.check_output(['lscpu'],text=True),
 'memory':subprocess.check_output(['free','-h'],text=True),
 'disk':subprocess.check_output(['df','-h','/workspace','/tmp'],text=True),
 'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]}
Path('validation/results/environment.json').write_text(json.dumps(report,indent=2)+'\n')
print('Environment imports, dependencies, and CUDA matrix multiplication passed.')
PY
