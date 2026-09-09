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

## Verified milestone: checkpoint and data

Both `Tennis_big` (15,300,637 parameters) and `FineGym_big` (15,369,355 parameters) passed strict loading and tensor equality with the published checkpoint. No model source changes were needed with the locked environment.

YouTube rejected both Tennis probes with a sign-in/bot check. The final subset therefore uses FineGym footage from the Sports-QA authors (`HopLeeTop/Sports-QA`), selecting 20 test clips (five per apparatus, all 32 classes) and two separate training clips. HTTP range access to ZIP members reduces the payload to 194,432,641 bytes. `validation/subset.json` pins the source revision, members, original labels, and video hashes.

JPEG extraction is cached at `/tmp/adaspot-finegym-frames` because network-mounted per-frame files were much slower. Persistent input videos remain in `artifacts/finegym/videos`; rerunning the downloader regenerates and verifies frames. Override with `ADASPOT_FRAME_DIR` if needed.

The two-clip GPU smoke test processed 2,340 frames, with a peak allocation of 1,525,325,312 bytes. It yielded soft-NMS mAP of 21.60%, 38.15%, 39.60%, and 40.55% at tolerances 0/1/2/4 frames. These are preliminary subset measurements, with the archive-trimming caveat documented in the manifest.
