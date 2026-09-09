# Reproduce AdaSpot checkpoint, inference, and training validation

This workflow validates the released **FineGym_big** model against the actual AdaSpot PyTorch implementation. It uses real FineGym videos, the repository's original train/test assignments and event labels, native inference, and one native training update. It also records a successful strict load of Tennis_big before switching datasets because YouTube access was blocked.

The goal is to establish code/checkpoint compatibility, inspect prediction quality, and check that fine-tuning can execute. The small, transformed subset is not a reproduction of the paper's benchmark.

## 1. Machine and environment

The inspected image has:

| Component | Observed value |
| --- | --- |
| GPU | NVIDIA RTX PRO 4500 Blackwell; 32 GB VRAM, not the initially expected 24 GB |
| Driver / driver CUDA capability | 580.159.04 / CUDA 13.0 |
| Installed PyTorch runtime | torch 2.8.0+cu128; CUDA 12.8 |
| torchvision | 0.23.0+cu128 |
| GPU architecture | sm_120, supported by this PyTorch build |
| CPU | AMD EPYC 7443P; 24 cores / 48 logical CPUs |
| System memory | Approximately 251 GiB |
| Python / uv | 3.12.3 / 0.9.0 |
| Storage | Network-mounted `/workspace`; approximately 28 GB initially free on local `/tmp` filesystem |

`validation/results/environment.json` contains the machine inventory, FFmpeg version, repository HEAD/status at inventory time, package locations, and complete active package inventory. The metrics also record SHA-256 values of the exact native source and validation code used, including changes awaiting the final commit. The environment uses the image's existing Python, PyTorch, CUDA libraries, and FFmpeg. The locked additions include timm, OpenCV, NumPy requirements, plotting/evaluation dependencies, SoccerNet, and Hugging Face tools. SoccerNet is required even for FineGym because `util/eval.py` imports it unconditionally.

Start from the repository root:

```bash
cd /workspace/adaspot
bash scripts/setup_validation.sh
```

Expected final message:

```text
Environment imports, dependencies, and CUDA matrix multiplication passed.
```

The script creates `.venv` with `--system-site-packages`, compares each locked package against the active interpreter, and installs only missing or mismatched versions. It validates dependency metadata, imports the actual model dependencies, and executes CUDA matrix multiplication.

**Learning:** uv 0.9.0's package installer does not count inherited system packages. A direct virtual-environment install would duplicate the CUDA stack. Resolve the complete lock, then use `importlib.metadata` to filter already satisfied packages and install the remainder with `--no-deps`. The exact resolution is checked in as `validation/requirements.lock`; normal reproduction does not regenerate it.

If intentionally updating dependencies, regenerate the lock with:

```bash
uv pip compile validation/requirements.in \
  -c validation/constraints.txt --torch-backend cu128 \
  -o validation/requirements.lock
```

Then rerun every validation stage. `torch==2.8.0+cu128` needs the CUDA 12.8 PyTorch package index; the default PyPI index alone did not resolve it. On an otherwise compatible Python 3.12 image without preinstalled PyTorch, the setup script installs the pinned missing CUDA packages from that index. CUDA driver support and FFmpeg remain machine prerequisites.

## 2. Reproduce everything

```bash
bash scripts/run_validation.sh
```

The driver runs setup, strict checkpoint loading, data download/verification, two-clip smoke inference, 20-clip evaluation, one training update, and the final evidence audit. It uses `pipefail`; any failed stage stops the driver. Logs are retained in `artifacts/logs/`.

For individual stages, use the numbered commands below. All scripts resolve paths relative to the repository root. Downloads are cached and source revisions are pinned. No authentication was needed for the final chosen checkpoint or data source.

## 3. Download and strictly load the largest released backbone variant

```bash
.venv/bin/python scripts/validate_adaspot.py checkpoint --model FineGym_big
```

The source is the [AdaSpot author's Hugging Face release](https://huggingface.co/arturxe/AdaSpot/tree/main). `validation/FineGym_big.json` records its immutable revision and `FineGym_big-1/checkpoint_best.pt` path. The checkpoint is cached under `artifacts/huggingface/`.

The release offers small and big variants for each supported dataset. FineGym_big uses the released big RegNetY-004/GSF backbone, a 100-frame temporal encoding, and 32 foreground classes plus background. The full trainable module has **15,369,355 parameters**; its checkpoint is **62,069,100 bytes**. Larger RegNet mappings exist in source, but no corresponding larger pretrained checkpoint is released. Class-head sizes differ slightly between datasets; this selection uses the largest released backbone for FineGym.

The checker constructs the actual `AdaSpot.Impl` (`torch.nn.Module`), downloads the source's default ImageNet backbone initialization if absent, loads the AdaSpot state dict with `strict=True`, and compares every loaded tensor against the original checkpoint. It uses CPU `torch.load(..., weights_only=True)` before loading to CUDA. Results include SHA-256, parameter count, state-tensor count, complete configuration, and compatibility findings in `validation/results/FineGym_big_checkpoint.json`.

Expected message:

```text
Strict loading and equality of all checkpoint tensors passed.
```

No core model, dataset, evaluation, or configuration source changes were needed. `timm==1.0.29` in the locked environment is compatible with these weights. Tennis_big also passed, with 15,300,637 parameters; its source and compatibility report are retained separately.

## 4. Download and verify the real dataset subset

```bash
export ADASPOT_FRAME_DIR=/tmp/adaspot-finegym-frames
.venv/bin/python scripts/download_subset.py
```

The input videos come from the [Sports-QA authors' release](https://huggingface.co/datasets/HopLeeTop/Sports-QA/tree/main), linked by their [official repository](https://github.com/HopLee6/Sports-QA). These are FineGym event clips, not unrelated substitute videos. Labels come from AdaSpot's existing `data/finegym/train.json` and `test.json`, not Sports-QA question/answer labels. See the [FineGym project](https://sdolivia.github.io/FineGym/) and existing dataset README for attribution and dataset terms.

The frozen selection in `validation/subset.json` contains:

- 20 held-out test clips: five each from balance beam, floor exercise, uneven bars, and vault; together they cover all 32 event classes.
- Two training clips with source videos disjoint from the evaluation source videos.
- Original labels, immutable archive URLs, archive identifiers, ZIP member names and CRCs, byte sizes, and individual video SHA-256 values.

Selection uses seed 1, integer 25/30 FPS sources, greedy coverage of unseen classes, then shortest duration, with five test clips per apparatus. The checked-in manifest takes precedence on reproduction; the script does not resample it.

**Selective download:** the complete author archives total tens of gigabytes. `scripts/remote_zip.py` reads their ZIP directories and only the selected members with HTTP Range requests. It requires HTTP 206 and verifies the returned byte range. Python's ZIP reader verifies member CRCs. Only **194,432,641 bytes (185.43 MiB)** of selected video payload are retained, plus small directory transfers; the allowed 50 GB download ceiling is unnecessary.

Videos persist in `artifacts/finegym/videos/`. Frames are resized to **796×448**, saved as JPEG quality 95, and named with six-digit absolute source-frame indices under the source video ID, matching the native FineGym loader's layout. Their regenerable cache defaults to local `/tmp/adaspot-finegym-frames`, approximately 2.8 GiB for this subset. Network-mounted per-frame files were markedly slower; use the same `ADASPOT_FRAME_DIR` for download, inference, and audit if overriding it. Rerunning the downloader reconstructs a missing local frame cache from persistent MP4s.

The downloader verifies FPS, decoded frame count, image dimensions, event bounds, split separation, hashes, and class coverage. Reports are saved in `validation/results/data.json`; adapted annotations are in `validation/test_labels.json` and `validation/train_labels.json`. The original repository labels remain unchanged.

### Frame alignment and its limits

AdaSpot's FineGym labels include surrounding padding, while the released Sports-QA videos are trimmed. The conversion preserves absolute event-frame coordinates:

```text
original_origin = original.start_frame - original.pad[0]
absolute_event_frame = original_origin + original_event.frame
download_origin = integer_start_seconds_in_event_ID × FPS
adapted_event.frame = absolute_event_frame - download_origin
```

Each selected clip's measured duration matches `(integer_end - integer_start + 1)` seconds. Original and adapted labels and the applied frame shift are retained. No shifts are selected by optimizing model accuracy.

The integer-second trim origin is inferred from clip IDs and durations; identity against the complete original source video was not independently established. Visuals show many correctly aligned event peaks, but exact-frame scores should still be interpreted as **measurements on this transformed subset**, with different boundary context and possible trimming/re-encoding effects. Obtain original source frames before treating these numbers as a benchmark reproduction.

### Access failures that informed the choice

- Tennis probes `wZnCcqm_g-E` and `LVjO-dH0P6c` returned YouTube's sign-in/bot check through yt-dlp.
- FineDiving's official instructions require an emailed access agreement. An alternative `JNU-SmartEducation/FineDiving-Test` archive returned HTTP 401.
- A public FineGym mirror established that selective ZIP extraction worked. The same probe clip was byte-identical to the authors' copy; final manifest URLs use the authors' release directly.

## 5. Smoke-test two clips, then evaluate all 20

```bash
.venv/bin/python scripts/validate_adaspot.py smoke
.venv/bin/python scripts/validate_adaspot.py evaluate
```

Both stages load and verify the full checkpoint before calling `clean_modules()` for inference. This operation deletes auxiliary training heads; never reuse that pruned instance for training.

Inference uses the native `ActionSpotVideoDataset` and `AdaSpot.predict`: batch size 1, 100-frame windows, stride 1, 50-frame overlap, native start/end padding, native resizing/cropping/normalization, BF16 autocast, and FP32 overlap accumulation. Every prediction must have shape `[1, 100, 33]`, be finite, have approximately normalized class probabilities, and contribute coverage to every real frame. The wrapper checks coverage before the upstream evaluator can silently replace missing support with one.

The smoke stage always runs inference on the first two manifest test clips. The evaluation stage handles all 20 and can reuse previously validated scores. Cached arrays in `artifacts/scores/` are keyed by a signature of the checkpoint hash, model config, original source-code hashes, input video hash, adapted labels, and pipeline version. Per-video timing and allocated/reserved CUDA peaks are retained alongside those arrays. The two smoke clips' timings include initial kernel/setup overhead; this is not a rigorous throughput benchmark.

Outputs in `validation/results/` include:

| Artifact | Contents |
| --- | --- |
| `smoke_metrics.json`, `evaluate_metrics.json` | Accuracy, class support, class F1, raw/NMS/soft-NMS mAP and per-class AP at 0/1/2/4-frame tolerances, provenance, memory, timing |
| `*_predictions.json` | Scored event predictions for all postprocessing variants |
| `*_argmax_events.json` | Framewise argmax foreground events |
| `*_timeline.png` | First two clips: label times, per-frame class scores, and soft-NMS events above 0.2 |
| `*_frames.jpg` | First two clips: sampled labeled frames and the predicted score at that frame |

Report frame accuracy alongside the always-background baseline: rare events make accuracy alone misleading. Native foreground-detection F1 ignores foreground class identity; per-class F1 and spotting AP supply the class-specific evidence.

## 6. Verify one native training update

```bash
.venv/bin/python scripts/validate_adaspot.py train
```

This starts a **fresh, unpruned** model from the original checkpoint. It selects a labeled 100-frame window from each training clip, uses a batch of two, and runs native `AdaSpot.epoch` with augmentation, MixUp, foreground weight 5, BF16 autocast, the main and both auxiliary losses, and one AdamW step at learning rate 0.0008.

Instrumentation records all three actual cross-entropy losses without changing the loss calculation, captures parameter-gradient finiteness/norms, checks both backbones and all prediction heads receive nonzero gradients, and verifies parameter updates remain finite. It hashes the downloaded checkpoint again after the update; no updated checkpoint is saved. Evidence is in `validation/results/training.json`.

This checks the actual optimizer path rather than only calling backward on artificial inputs. It does not establish convergence, a useful fine-tuning learning rate, or the largest training batch size.

## 7. Audit the saved evidence

```bash
.venv/bin/python scripts/audit_validation.py
```

The audit checks all 22 video hashes and extracted-frame hashes, original/adapted label consistency, train/test source separation, all 32 classes, finite fully covered prediction arrays, recomputed frame accuracy and native mAP, source hashes, checkpoint-loading evidence, gradient/update evidence, memory fit, and readable visualizations. It writes `validation/results/audit.json` and fails on any inconsistency.

The large input assets and virtual environment are ignored by Git. The manifest, lock, scripts, small results, and visuals are committed. Reproduction regenerates the ignored assets.

## Measured results and fine-tuning assessment

The individual stages and then `bash scripts/run_validation.sh` both completed successfully on **2026-09-09**. The complete rerun reproduced identical accuracy/mAP values and training loss, and its final audit passed. Logs from the documented driver are in `artifacts/logs/`.

The validation verified 20 test clips, **25,290 frames**, **258 events**, all **32 classes**, and two separate training clips. Native source code and configuration remained unchanged.

| Metric | Raw scores | NMS, window 1 | Soft-NMS, window 2 |
| --- | ---: | ---: | ---: |
| mAP, exact frame | 26.54% | 18.38% | 22.86% |
| mAP, ±1 frame | 41.03% | 48.66% | 48.76% |
| mAP, ±2 frames | 45.20% | 56.19% | 55.94% |
| mAP, ±4 frames | 47.53% | 60.62% | 60.35% |
| Mean across these tolerances | 40.08% | 45.96% | **46.98%** |

Frame accuracy was **98.68%**, versus **98.98%** for always predicting background. Native exact-frame foreground-detection F1 was **33.40%**. Per-class support ranges from **1 to 17 events**. Mean per-class soft-NMS AP at ±4 frames is **35.30% for balance beam**, **78.39% for floor exercise**, **64.85% for uneven bars**, and **68.86% for vault**. The wide-camera balance-beam example has weak scores at several labeled events, while the uneven-bars example shows many closely aligned peaks. Inspect per-class AP and support in the JSON before drawing conclusions.

Inference at batch size 1 allocated at most **1.42 GiB** and reserved **1.70 GiB** of CUDA memory. The final combined per-clip inference timings, including two cached smoke-run measurements, totaled approximately **107 seconds** for 498 windows. Timing varies with startup and cache state.

One native training step at batch size 2 allocated **8.78 GiB** and reserved **8.89 GiB**. Total loss was **0.34973**, with main/high-resolution/low-resolution losses **0.32218 / 0.34270 / 0.38431**. All **741** parameter-gradient tensors were finite; no trainable parameter lacked a gradient. Both backbones and all three prediction heads received nonzero gradient norms and finite parameter updates. The original checkpoint's SHA-256 remained unchanged.

**Assessment:** this is a working starting point for fine-tuning: released weights strictly match the source, GPU inference produces meaningful event peaks, and the native augmented/MixUp training path executes a real optimizer update with ample memory headroom on this machine. Prediction quality is mixed and depends on tolerance; the transformed small subset and inferred archive trim origins limit accuracy claims. Before substantive fine-tuning, validate the intended target domain's labeling/frame alignment and evaluate on its held-out data. Neither convergence nor paper-level performance is established here.

The first two clips' visuals are available directly:

- [Balance-beam event timeline](../validation/results/5Bfx6Wz3KKs_E_019366_019418_timeline.png) and [labeled frames](../validation/results/5Bfx6Wz3KKs_E_019366_019418_frames.jpg).
- [Uneven-bars event timeline](../validation/results/nEoD3t5O3Hg_E_008416_008445_timeline.png) and [labeled frames](../validation/results/nEoD3t5O3Hg_E_008416_008445_frames.jpg).

## Useful implementation findings

- Initial shell inspection failed with `bwrap: No permissions to create new namespace`. The user subsequently changed the runner permission profile, after which commands worked. This is a runner/sandbox issue, not a missing Python dependency.

- `clean_modules()` permanently removes auxiliary modules from that instance. Loading fresh weights into an already pruned instance does not recreate the heads; construct a fresh model for training.
- Native training's `step()` explicitly disables the passed scaler and uses BF16 autocast with ordinary backward. The validation intentionally exercises that existing behavior.
- The upstream `ErrorStat.get_acc()` refers to a nonexistent `_get()` method. The validation uses `1 - err.get()` instead; no core patch was needed.
- The repository README contains stale CLI examples (`train_tdeed.py`, `--model`, and `--frame_width`). The actual standalone video CLI uses `--model_name` and has no `--frame_width`. Use the commands in this guide for this experiment.
- GPU peak allocations differ greatly between inference and training. Checkpoint size alone is not a VRAM estimate.
- An interrupted data download can be resumed by rerunning the downloader. Existing MP4 hashes are checked; extracted frames are verified or regenerated. A server that ignores byte ranges fails explicitly rather than silently downloading a full archive.
- The final audit verifies saved evidence and actual inputs. Its trimming caveat remains part of the result even when every technical check passes.
