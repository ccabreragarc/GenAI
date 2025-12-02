# GenAI

AI-based facial generation system with a DCGAN baseline and a DDPM diffusion model. The project now includes device-agnostic training, centralized data transforms, and a Streamlit app that can sample from either generator.

## What's Inside
- DCGAN generator/discriminator for quick baselines (`train.py --model dcgan`).
- DDPM UNet with a linear noise scheduler plus iterative sampling (`train.py --model ddpm` or `train_ddpm.py`).
- Streaming CelebA loader with shared transforms (resize -> tensor -> normalize to `[-1, 1]`).
- Metrics for FID/mode collapse and a Streamlit UI that can toggle between DCGAN and DDPM with sampling progress.

## Device Agnostic by Design
The code selects a device in the order CUDA -> MPS -> CPU and never hard-codes `.cuda()`:
```python
from utils import get_device
device = get_device()  # respects AMD/ROCm and CPU-only setups
```
Pass `--device cpu` to force CPU if ROCm is not available on Windows.

## Training
- DCGAN: `python train.py --model dcgan --batch-size 64 --resolution 64` (DCGAN fixed to 64×64)
- DDPM: `python train.py --model ddpm --batch-size 32 --timesteps 1000 --resolution 64`
- Dedicated DDPM entrypoint (same args): `python train_ddpm.py --batch-size 32 --timesteps 1000`

Checkpoints:
- DCGAN -> `models/dcgan_latest.pt` (legacy alias: `models/latest_checkpoint.pt`)
- DDPM -> `models/ddpm_latest.pt`
Samples land in `outputs/samples/<model>/`.

## Data & Preprocessing
- Source: `nielsr/CelebA-faces` via HuggingFace streaming (fallback to local folder `data/celeba/` when offline).
- Transforms centralized in `data_loader.py` using `build_transforms(resolution)`.
- Loader always emits `batch["pixel_values"]` shaped `[B, 3, H, W]` in `[-1, 1]`, so GAN, VAE, and DDPM pipelines share the same input contract.

## Deployment (Streamlit)
Run `streamlit run app_streamlit.py`.
- Toggle DCGAN/DDPM generation.
- DDPM shows iterative sampling progress.
- Gracefully handles missing checkpoints and optional env downloads (`CHECKPOINT_URL` for DCGAN, `CHECKPOINT_URL_DDPM` for DDPM).

## Responsibilities & Status
- Dataset & preprocessing - streaming loader + normalization.
- Model training - DCGAN baseline and DDPM pipeline/support.
- Debugging & stability - metrics, logging, safer checkpoints.
- Deployment - Streamlit/dashboard updated for both models.
- Device portability - AMD/CPU friendly; no CUDA-only calls.

## Metrics & Evaluation
- FID and collapse heuristics in `metrics.py`; run `python metrics_runner.py --fake_dir outputs/samples/<model>/ --real_dir <reference> --device cpu --weights <local_inception_weights>`.
- Outputs stored under `outputs/logs/metrics.pt` for dashboard consumption.

## Smoke Tests (CPU-only)
- Data pipeline: `python data_loader.py` (runs verification block, prints shape/min/max).
- DCGAN quick run: `python train.py --model dcgan --batch-size 16 --epochs 1 --steps-per-epoch 10 --device cpu`. Checkpoints in `models/`, samples in `outputs/samples/dcgan/`.
- DDPM quick run: `python train.py --model ddpm --batch-size 8 --epochs 1 --steps-per-epoch 5 --device cpu`. Checkpoints in `models/ddpm_latest.pt`, samples in `outputs/samples/ddpm/`.
- Metrics: `python metrics_runner.py --device cpu --max_batches 2` (uses synthetic noise if dirs missing).
- Streamlit: `streamlit run app_streamlit.py` (warns if checkpoints missing; toggles should not crash).
