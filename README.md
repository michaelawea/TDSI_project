# TDSI_project

Lightweight U-Net for 4× single-image super-resolution on a self-generated shapes dataset.

**Purpose**
- Provide an end-to-end, minimal SISR baseline: data → train → evaluate.
- Be fast to run on CPU/GPU with no external datasets.
- Offer clear, objective feedback via PSNR/SSIM and visualizations.

**Requirements (Project Needs)**
- Generate paired HR/LR images synthetically; avoid data downloads.
- Train a small model that converges quickly (minutes to tens of minutes).
- Save checkpoints and summaries for reproducibility.
- Keep key knobs configurable (HR/LR size, sample count, scale factor).

**Technology**
- Data: OpenCV/NumPy draw random circles/rectangles/triangles at `HR_SIZE=256`; downscale to `LR_SIZE=64` (×4) via bicubic/bilinear/area in `generate_dataset.ipynb`.
- Model: Lightweight U‑Net (~1.9M params) in PyTorch trained in `train_unet.ipynb` with L1 loss and Adam (`lr=2e-4`, `epochs=30`, `batch_size=8`, `train_split=0.9`).
- Outputs: `checkpoints/best_model.pth`, `checkpoints/test_results.png`, `checkpoints/training_summary.txt`.

**Environment**
- Python 3.8+
- Packages: `numpy`, `matplotlib`, `opencv-python`, `Pillow`, `torch`, `torchvision`, `tqdm`, `scikit-image`
- Install: `pip install -r requirements.txt` (or install packages above individually)

Files to run: `generate_dataset.ipynb`, `train_unet.ipynb`. Chinese guide: `README_zh.md`.
