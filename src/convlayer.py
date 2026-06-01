import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, device=None):
        super(ConvLayer, self).__init__()

        # 确保 device 在类初始化时正确设置
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化卷积层并将其转移到指定的设备
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding).to(self.device)

    def forward(self, x):
        # 确保输入也在同一设备
        x = x.to(self.device)
        return self.conv(x)


class ConvolutionalNetwork1(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()

        # 定义多个卷积层
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = ConvLayer(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvLayer(in_channels=32, out_channels=16, kernel_size=4, stride=2,padding=1)  # 减少 stride 增加 padding
        self.conv3 = ConvLayer(in_channels=16, out_channels=4, kernel_size=4, stride=2, padding=1)  # 同样调整

    def forward(self, x):
        # 执行前向传播
        print(f"ConvolutionalNetwork input shape: {x.shape}")
        x = self.conv1(x)
        print(f"After conv1 shape: {x.shape}")
        x = self.conv2(x)
        print(f"After conv2 shape: {x.shape}")
        x = self.conv3(x)
        print(f"After conv3 shape: {x.shape}")
        return x

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()

        # 目标: Input [B, 16, 64] -> Output [B, 4, 4]
        # 每一层 Stride=2 (长度减半)
        
        # Layer 1: [16, 64] -> [32, 16] (升维提取特征)
        self.conv1 = ConvLayer(in_channels=16, out_channels=32, kernel_size=8, stride=4, padding=2)

        # Layer 2: [32, 16] -> [64, 8] 
        self.conv2 = ConvLayer(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        
        # Layer 3: [64, 8] -> [128, 4] 
        self.conv3 = ConvLayer(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)

        # Layer 4: [128, 4]   -> [64, 2]  
        self.conv4 = ConvLayer(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # x: [B, 16, 64]
        x = self.conv1(x) # [B, 32, 16]
        x = self.conv2(x) # [B, 64, 8]
        x = self.conv3(x) # [B, 128, 4]
        x = self.conv4(x) # [B, 64, 2]
        return x
# 测试类
if __name__ == "__main__":
    # 初始化网络
    model = ConvolutionalNetwork()

    # 创建一个假输入，形状为 (batch_size=1, channels=64, length=64)
    x = torch.randn(16, 64, 64)

    # 进行前向传播
    output = model(x)

    print("Output shape:", output.shape)
