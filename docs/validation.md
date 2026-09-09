# AdaSpot validation work log

Implementation in progress. This guide will be replaced with the verified commands and results after the data, inference, and training checks finish.

## Environment

The image has an NVIDIA RTX PRO 4500 Blackwell with 32 GB VRAM (not the initially expected 24 GB), 48 logical CPU cores, approximately 251 GiB RAM, Python 3.12.3, uv 0.9.0, FFmpeg, and PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128. Use the installed CUDA stack.

Run `bash scripts/setup_validation.sh`. The script creates `.venv` with system site packages, installs only missing or mismatched locked packages, validates dependency metadata and imports, and exercises CUDA. The full dependency lock is `validation/requirements.lock`.

Learning: uv 0.9.0 does not consider inherited system packages when installing into a virtual environment. Directly installing requirements with CUDA-specific pins also needs the PyTorch CUDA index. Resolve the complete lock using `uv pip compile validation/requirements.in -c validation/constraints.txt --torch-backend cu128 -o validation/requirements.lock`, compare the lock against `importlib.metadata` in the target interpreter, then install only missing versions with `--no-deps`. This avoids downloading and duplicating existing PyTorch/CUDA packages.

## Intended checks

Strict loading into the actual full PyTorch module; two-clip inference smoke test; 20 labeled held-out clips; separate training clips for one optimizer step; predictions, spotting metrics, memory measurements, and visuals. Preserve the original checkpoint and use an unpruned instance for training, since `clean_modules()` deletes the auxiliary heads.

## Data access findings

The official FineDiving source requires an emailed access agreement. Its `JNU-SmartEducation/FineDiving-Test` Hugging Face mirror returned HTTP 401 for the archive. Tennis source access and publicly downloadable FineGym archives are being checked before choosing the final source.
