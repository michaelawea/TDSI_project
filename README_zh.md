# TDSI_project

用于 4× 单张图像超分辨率的轻量级 U‑Net 小项目，数据自生成，端到端可复现。

**目的**
- 提供一个从数据→训练→评估的极简 SISR 基线。
- 在无外部数据的前提下，快速在 CPU/GPU 上跑通。
- 以 PSNR/SSIM 与可视化结果给出直观反馈。

**需求**
- 合成 HR/LR 成对数据，避免下载数据集。
- 小模型、短训练时间（数分钟到十几分钟）。
- 保存最佳模型与训练总结，便于复现。
- 关键参数可配（HR/LR 尺寸、样本数、缩放倍数）。

**技术**
- 数据：在 `generate_dataset.ipynb` 中用 OpenCV/NumPy 绘制随机圆/矩形/三角形，`HR_SIZE=256`；下采样到 `LR_SIZE=64`（×4，bicubic/bilinear/area）。
- 模型：在 `train_unet.ipynb` 中训练轻量级 U‑Net（约 1.9M 参数），L1 损失，Adam（`lr=2e-4`，`epochs=30`，`batch_size=8`，`train_split=0.9`）。
- 输出：`checkpoints/best_model.pth`、`checkpoints/test_results.png`、`checkpoints/training_summary.txt`。

**环境**
- Python 3.8+
- 依赖：`numpy`、`matplotlib`、`opencv-python`、`Pillow`、`torch`、`torchvision`、`tqdm`、`scikit-image`
- 安装：`pip install -r requirements.txt`（或分别安装上述依赖）

运行文件：`generate_dataset.ipynb`、`train_unet.ipynb`。英文版说明见 `README.md`。
