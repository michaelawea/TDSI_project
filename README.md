# TDSI_project

Nasal CT super-resolution and segmentation with comparative study of U‑Net and KAN. This project targets reconstruction of high-resolution (HR) images from low-resolution (LR) nasal CT and evaluates anatomical segmentation both before and after super-resolution.

## Objectives
- 2D single-image super-resolution (SISR): reconstruct 4096×4096 HR from 256×256 LR (16× upscaling).
- Segmentation: assess segmentation quality and stability with and without SR.
- Comparative modeling: benchmark U‑Net against KAN/ConvKAN in terms of accuracy, robustness, memory, and training time.
- Extended goal: 3D CT volume reconstruction, segmentation, and super-resolution.

## Data and Preprocessing
- Dataset: NasalSeg (heterogeneous sizes, typically ~100×100 to ~200×200).
- Normalization: resize/pad all images to 256×256 with zero-padding to enable batched training and evaluation.
- Task specification: input 256×256 LR, target 4096×4096 HR (16×).

## Baselines and Early Experiments
- Synthetic sanity check: `generate_dataset.ipynb` builds paired HR/LR shape images (64×64 ↔ 256×256) for rapid iteration and reproducibility.
- Training notebooks:
  - U‑Net: `train_unet.ipynb`
  - KAN/ConvKAN: `train_convkan_superres.ipynb`
  - Metrics and examples: `metrics_tutorial.ipynb`
- Initial findings: U‑Net trains stably and performs reliably on SR; KAN/ConvKAN is feasible but more memory- and time-intensive (U‑Net-like topology with ConvKAN replacing conv layers and KAN replacing MLP blocks).

## Evaluation Metrics
- Primary: SSIM (structural similarity).
- Optional: PSNR, LPIPS to complement perceptual and distortion-oriented assessment.

## Compute and Constraints
- Hardware: single RTX 6000 (ENS Lyon).
- Challenge: 16× SR is compute- and memory-intensive; careful engineering and staged training are required.

## Two-Stage Patch-Based Strategy (×8 + ×2)
To balance the high upscaling factor with limited compute, we adopt a patch-based two-stage pipeline:

- Step 1 (data preparation)
  - From each 4096×4096 HR image, sample random 512×512 patches.
  - Downsample 512×512 HR patches to 64×64 LR patches (8×).

- Step 2 (train ×8 model)
  - Input: 64×64 LR patch; Output: 512×512 SR patch.
  - Scale example: 300 images × 10 patches/image ≈ 3000 training samples.

- Step 3 (inference: 256×256 → 2048×2048)
  - Tile the 256×256 input into 4×4 = 16 patches of size 64×64.
  - Run each patch through the ×8 model to obtain 512×512 outputs.
  - Stitch outputs to reconstruct 2048×2048.

- Step 4 (final upscaling: 2048×2048 → 4096×4096)
  - Use bicubic interpolation or a lightweight ×2 model.

Reference code: `train_patch_debug.ipynb`.

## Limitations and Future Work
- Resource sensitivity: KAN/ConvKAN models are more demanding in memory and time than U‑Net.
- Engineering improvements:
  - Memory/compute: AMP (mixed precision), gradient checkpointing, smaller batch/patch sizes, efficient data pipelines.
  - Inference quality: patch overlap and boundary weighting to reduce seams.
  - Model simplification: lighter decoders, depthwise separable or low-rank layers; pruning/sparsity for KAN variants.
  - Metrics/visualization: add PSNR/LPIPS; richer visual QA and distortion maps.

## Notebooks
- Data generation: `generate_dataset.ipynb`
- U‑Net training: `train_unet.ipynb`
- KAN/ConvKAN training: `train_convkan_superres.ipynb`
- Metrics examples: `metrics_tutorial.ipynb`
- Patch strategy: `train_patch_debug.ipynb`

If you use this project in academic work, please acknowledge or cite appropriately.
