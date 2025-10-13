# =============================================================================
# train_convkan_superres.py
# -----------------------------------------------------------------------------
# 这个脚本用于训练一个基于卷积Kolmogorov-Arnold网络(ConvKAN)的图像超分辨率模型。
#
# 包含以下部分:
# 1. 导入所需的库。
# 2. 一个自定义的PyTorch Dataset，用于加载低分辨率和高分辨率的图像对。
# 3. 一个专为超分辨率任务设计的ConvKAN模型架构。
# 4. 完整的训练设置和训练循环。
#
# 如何运行:
# 1. 确保你已经安装了必要的库: pip install torch torchvision pillow tqdm convkan
# 2. 将此文件放置在你的项目根目录。
# 3. 确保你的数据集在 'dataset/low_resolution' 和 'dataset/high_resolution' 文件夹中。
# 4. 在终端中运行: python train_convkan_superres.py
# =============================================================================

# %%
# =============================================================================
# 步骤 1: 导入库
# =============================================================================
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
from tqdm import tqdm

# 尝试导入convkan，如果失败则提示安装
try:
    from convkan import ConvKAN, LayerNorm2D
except ImportError:
    print("错误: convkan 库未找到。")
    print("请通过 'pip install convkan' 命令进行安装。")
    exit()

# %%
# =============================================================================
# 步骤 2: 创建自定义数据集
# =============================================================================
class SuperResolutionDataset(Dataset):
    """
    一个自定义的数据集，用于加载低分辨率(lr)和高分辨率(hr)的图像对。
    它会自动匹配 lr_dir 和 hr_dir 中文件名相似的图像。
    """
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        
        self.lr_images = sorted(glob.glob(os.path.join(self.lr_dir, '*.png')))
        self.hr_images = sorted(glob.glob(os.path.join(self.hr_dir, '*.png')))
        
        # 确保低分辨率和高分辨率图像数量匹配
        if not self.lr_images or not self.hr_images:
            raise IOError(f"错误: 在 '{lr_dir}' 或 '{hr_dir}' 中没有找到图像文件。请检查路径。")
        
        print(f"发现 {len(self.lr_images)} 张低分辨率图像和 {len(self.hr_images)} 张高分辨率图像。")
        # 实际项目中可能需要更复杂的匹配逻辑
        # 这里我们假设文件数量一致且按排序后一一对应
        assert len(self.lr_images) == len(self.hr_images), "低分辨率和高分辨率图像的数量不匹配!"

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image_path = self.lr_images[idx]
        hr_image_path = self.hr_images[idx] # 直接按索引匹配
        
        lr_image = Image.open(lr_image_path).convert("RGB")
        hr_image = Image.open(hr_image_path).convert("RGB")

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
            
        return lr_image, hr_image

# %%
# =============================================================================
# 步骤 3: 定义 ConvKAN 超分辨率模型
# =============================================================================
class ConvKANResBlock(nn.Module):
    """一个使用ConvKAN的残差块"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvKAN(channels, channels, kernel_size=3, padding=1),
            LayerNorm2D(channels),
            ConvKAN(channels, channels, kernel_size=3, padding=1),
            LayerNorm2D(channels)
        )

    def forward(self, x):
        return x + self.block(x) # 残差连接

class ConvKAN_SR(nn.Module):
    """
    一个专为超分辨率设计的ConvKAN模型。
    架构: Head -> Body (Residual Blocks) -> Upsampling -> Tail
    """
    def __init__(self, in_channels=3, out_channels=3, base_filters=64, n_res_blocks=8, upscale_factor=2):
        super().__init__()
        
        # 1. Head: 初步特征提取
        self.head = ConvKAN(in_channels, base_filters, kernel_size=3, padding=1)
        
        # 2. Body: 深度特征提取的残差块
        body = [ConvKANResBlock(base_filters) for _ in range(n_res_blocks)]
        self.body = nn.Sequential(*body)
        
        # 3. Upsampling: 使用PixelShuffle进行上采样
        self.upsample = nn.Sequential(
            ConvKAN(base_filters, base_filters * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            LayerNorm2D(base_filters)
        )
        
        # 4. Tail: 重建最终图像
        self.tail = nn.Conv2d(base_filters, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.head(x)
        
        # 保存残差连接的初始输入
        res = x
        x = self.body(x)
        x = x + res # 添加一个长跳跃连接

        x = self.upsample(x)
        x = self.tail(x)
        return x

# %%
# =============================================================================
# 步骤 4: 主函数 - 训练设置和执行
# =============================================================================
def main():
    # --- 超参数设置 ---
    LR_DIR = "dataset/low_resolution"
    HR_DIR = "dataset/high_resolution"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    UPSCALE_FACTOR = 2 # 假设低分辨率是高分辨率尺寸的一半

    print(f"使用设备: {DEVICE}")

    # --- 数据加载 ---
    transform = transforms.ToTensor()
    try:
        train_dataset = SuperResolutionDataset(lr_dir=LR_DIR, hr_dir=HR_DIR, transform=transform)
    except (IOError, AssertionError) as e:
        print(e)
        return # 如果数据集有问题则退出
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)

    # --- 模型, 损失函数, 优化器 ---
    model = ConvKAN_SR(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    criterion = nn.L1Loss() # L1损失通常能带来更清晰的图像
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("模型和数据加载完毕，开始训练...")

    # --- 训练循环 ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        total_loss = 0.0
        
        for lr_images, hr_images in progress_bar:
            lr_images = lr_images.to(DEVICE)
            hr_images = hr_images.to(DEVICE)
            
            # 前向传播
            optimizer.zero_grad()
            sr_images = model(lr_images)
            
            # 计算损失
            loss = criterion(sr_images, hr_images)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 结束, 平均损失: {avg_loss:.4f}")

    print("训练完成!")

    # --- 保存模型 ---
    torch.save(model.state_dict(), 'convkan_sr_model.pth')
    print("模型已保存至 convkan_sr_model.pth")

# %%
# =============================================================================
# 脚本入口
# =============================================================================
if __name__ == '__main__':
    main()
