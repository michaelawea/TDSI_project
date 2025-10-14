# TDSI_project

鼻腔 CT 图像的超分辨率重建与分割（对比 U-Net 与 KAN）。本项目围绕临床鼻腔 CT 图像，研究从低分辨率图像重建高分辨率图像，并开展解剖区域分割；同时对比传统 U‑Net 与基于 KAN（Kolmogorov–Arnold Networks）的模型在该任务中的可行性与性能。

## 研究目标
- 2D 单幅图像超分辨率（SISR）：从 256×256 低分辨率（LR）重建至 4096×4096 高分辨率（HR），对应 16× 放大。
- 解剖区域分割：在 SR 前后分别评估分割质量与稳定性。
- 模型比较研究：系统对比 U‑Net 与 KAN/ConvKAN 架构的性能、训练效率与资源消耗。
- 拓展目标：对 CT 体数据进行 3D 重建，并在体域空间开展分割与超分辨率研究。

## 数据与预处理
- 数据来源：NasalSeg 数据集（图像尺寸不一，分辨率多在约 100×100 至 200×200 之间）。
- 尺度统一：将所有图像规范化至 256×256，空白区域以 0 填充，以便批量训练与评估。
- 任务定义：输入 256×256 LR，目标 HR 为 4096×4096（16×）。

## 基线与初步实验
- 合成数据验证：在 `generate_dataset.ipynb` 中生成规则形状的 HR/LR 成对数据（64×64 ↔ 256×256）用于快速迭代与可复现实验。
- 模型与脚本：
  - U‑Net 训练：`train_unet.ipynb`
  - KAN/ConvKAN 训练：`train_convkan_superres.ipynb`
  - 评价指标与示例：`metrics_tutorial.ipynb`
- 初步结论：U‑Net 在超分辨率任务上收敛稳定、效果可靠；KAN/ConvKAN 具备可行性，但对显存与训练时长更为敏感（以 ConvKAN 替代卷积层、以 KAN 替代 MLP 的 U‑Net 类结构在资源占用上更高）。

## 评价指标
- 主要指标：SSIM（结构相似性）。
- 可选指标：PSNR、LPIPS 等，可在后续实验中纳入以补充主客观评估。

## 训练资源与约束
- 硬件：ENS Lyon 单卡 RTX 6000。
- 挑战：16× 超分辨率训练难度高，显存与时间成本显著，需要工程化优化与分阶段策略。

## 分阶段训练策略（Patch‑based ×8 + 轻量 ×2）
为兼顾高放大倍率与有限算力，采用基于补丁（patch）的两阶段策略：

- Step 1（数据准备）
  - 从 4096×4096 HR 图像随机裁剪 512×512 patch。
  - 将 512×512 HR patch 下采样为 64×64 LR patch（8×）。

- Step 2（训练 8× 模型）
  - 输入：64×64 LR patch；输出：512×512 SR patch。
  - 规模示例：300 张图 × 10 patch/图 ≈ 3000 个训练样本。

- Step 3（推理：256×256 → 2048×2048）
  - 将输入 256×256 图像划分为 4×4=16 个 64×64 小块。
  - 逐块通过 8× 模型得到 512×512 结果。
  - 按网格拼接复原为 2048×2048。

- Step 4（最终放大：2048×2048 → 4096×4096）
  - 使用双三次插值（bicubic）或额外训练的轻量 2× 模型完成最终放大。

参考代码：`train_patch_debug.ipynb`。

## 现存问题与优化方向
- 显存与时长：KAN/ConvKAN 架构相较 U‑Net 更易受显存与训练时间限制。
- 优化建议：
  - 计算与显存优化：混合精度（AMP）、梯度检查点（gradient checkpointing）、更小 batch/patch、分布式数据加载。
  - 推理平滑：patch 重叠与边界加权（减少拼接接缝）。
  - 模型层面：更轻量的解码器、可分离卷积/低秩近似；对 KAN 进行剪枝与稀疏化探索。
  - 指标与可视化：补充 PSNR/LPIPS，增加可视化质检与失真热图。

## 笔记本与入口
- 数据生成：`generate_dataset.ipynb`
- U‑Net 训练：`train_unet.ipynb`
- KAN/ConvKAN 训练：`train_convkan_superres.ipynb`
- 指标示例：`metrics_tutorial.ipynb`
- Patch 训练策略：`train_patch_debug.ipynb`

如在学术工作中使用本项目，请在合适位置进行致谢或引用。
