import torch
import torch.nn as nn

from src.convlayer import ConvolutionalNetwork
import torch.nn.functional as F

class TransposeConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0, device=None):
        super(TransposeConvLayer, self).__init__()

        # 设置设备
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化反卷积层并将其转移到设备
        self.deconv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding
        ).to(self.device)

    def forward(self, x):
        # 确保输入数据在同一设备上
        x = x.to(self.device)
        return self.deconv(x)


class DeconvolutionalNetwork1(nn.Module):
    def __init__(self):
        super(DeconvolutionalNetwork, self).__init__()

        # 定义多个反卷积层
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 第一层反卷积：从 [1, 16, 16] 到 [1, 16, 32]
        self.deconv1 = nn.ConvTranspose1d(in_channels=4, out_channels=16, kernel_size=4, stride=2, padding=1).to(device)

        # 第二层反卷积：从 [1, 16, 32] 到 [1, 32, 32]
        self.deconv2 = nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0).to(device)

        # 第三层反卷积：从 [1, 32, 32] 到 [1, 64, 64]
        self.deconv3 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1).to(device)

    def forward(self, x):
        # 执行前向传播
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class DeconvolutionalNetwork(nn.Module):
    def __init__(self):
        super(DeconvolutionalNetwork, self).__init__()

        # 目标: Input [B, 4, 4] -> Output [B, 16, 64]
        # 需要 4 次上采样 (2^4 = 16倍长度变化)
        # 通道变化逻辑: 4 -> 8 -> 16 -> 32 -> 16 (Encoder 的逆过程)
        # Layer 1: [4, 4] -> [8, 8]
        # (通道 4->8, 长度 x2)
        self.deconv1 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        # Layer 2: [8, 8] -> [16, 16]
        # (通道 8->16, 长度 x2)
        self.deconv2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        # Layer 3: [16, 16] -> [32, 32]
        # (通道 16->32, 长度 x2)
        self.deconv3 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        # Layer 4: [32, 32] -> [16, 64]
        # (通道 32->16, 长度 x2)
        self.deconv4 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=8, stride=4, padding=2)

    def forward(self, x):
        # 建议在中间层加入激活函数 (ReLU)，与 Encoder 对应
        # Layer 1
        x = F.relu(self.deconv1(x)) # Output: [B, 8, 8]
        # Layer 2
        x = F.relu(self.deconv2(x)) # Output: [B, 16, 16]
        # Layer 3
        x = F.relu(self.deconv3(x)) # Output: [B, 32, 32]
        # Layer 4 (最后一层通常不加 ReLU，或者根据后续模块决定，这里直接输出特征)
        x = self.deconv4(x)         # Output: [B, 16, 64]
        return x

# 测试类
if __name__ == "__main__":
    # 检查当前设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeconvolutionalNetwork().to(device)
    input_data = torch.randn(16, 4, 16).to(device)  # 输入 [batch_size, channels, length]
    output_data = model(input_data)
    print("Output shape:", output_data.shape)  # 期望输出 [1, 64, 64]
