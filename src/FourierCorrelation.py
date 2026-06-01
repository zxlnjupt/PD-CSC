import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, n_heads=4, scale=1.0, fs=2800.0):
        super(FourierBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads

        # 1. 频率设定
        self.max_modes = seq_len // 2 + 1

        # 2. 维度检查
        assert in_channels % n_heads == 0
        self.d_keys = in_channels // n_heads
        self.d_out_head = out_channels // n_heads


        # 3. 只保留可学习的 weights1
        self.scale = scale
        self.weights1 = nn.Parameter(
            self.scale * torch.complex(
                torch.rand(n_heads, self.d_keys, self.d_out_head, self.max_modes),
                torch.rand(n_heads, self.d_keys, self.d_out_head, self.max_modes)
            )

        )

        # 4. 感知权重 weights2，shape: (max_modes,)，不参与训练
        weights2 = self._get_perceptual_curve(seq_len, fs)
        weights2 = weights2 / weights2.max()
        weights2 = weights2 + 1e-2
        self.register_buffer('weights2', weights2)

        self.out_projection = nn.Linear(out_channels, out_channels)

    def _get_perceptual_curve(self, seq_len, fs):
        x_original = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 130, 150, 170, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400])
        y_original = np.array([249.7768, 245.5580, 138.9063, 92.9464, 66.1830, 45.5357, 37.7232, 29.6875, 27.2991, 21.5402, 15.7143, 10.0893, 8.8929, 8.6027, 11.0268, 20.2679, 40.9598, 55.4688, 69.9777, 91.7857, 119.7545, 162.0455, 194.7727, 219.0909, 230.2272, 241.8182, 245.4546, 246.8182, 245.4546, 248.1818])

        global_weights = 1.0 / (y_original + 1e-10)
        global_weights_norm = global_weights / global_weights.max()

        max_modes = seq_len // 2 + 1
        index_array = np.arange(max_modes)
        freq_resolution = fs / seq_len
        network_actual_freq_hz = index_array * freq_resolution

        final_weights_array = np.interp(network_actual_freq_hz, x_original, global_weights_norm)
        return torch.tensor(final_weights_array, dtype=torch.float)

    def forward(self, x):
        B, C, L = x.shape
        H = self.n_heads
        E = C // H
        x = x.view(B, H, E, L)

        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)
        # 乘以感知权重 weights2，shape 广播: (B, H, E, max_modes)
        x_ft = x_ft * self.weights2
        # Einsum: 'bhif,hiof->bhof'
        out_ft = torch.einsum('bhif,hiof->bhof', x_ft, self.weights1)
        # IFFT
        x_out = torch.fft.irfft(out_ft, n=L, dim=-1)
        # Reshape & Project
        x_out = x_out.reshape(B, -1, L)
        x_out = x_out.permute(0, 2, 1)
        x_out = self.out_projection(x_out)
        x_out = x_out.permute(0, 2, 1)

        return x_out