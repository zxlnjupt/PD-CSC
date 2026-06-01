import math
import pickle
from functools import partial
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import FSQ

from model.custom_Module import (
    MultiScaleDiscriminator, MLP_with_stats, DataEmbedding,
    CausalConv1d, EncoderBlock, DecoderBlock
)
from model.custom_Loss import hinge_discr_loss, hinge_gen_loss, SSIM_1D_loss
from src.utils import exists, gradient_penalty
from src.FourierCorrelation import FourierBlock
from src.convlayer import ConvolutionalNetwork
from src.Transpose_convlayer import DeconvolutionalNetwork
from vector_quantize_pytorch import ResidualVQ
import numpy as np
class PerceptualWeightedLoss(nn.Module):
    def __init__(self, seq_len=64, fs=2800.0):
        super().__init__()
        self.seq_len = seq_len
        self.fs = fs
        
        # 1. 生成 JND 权重 (逻辑与你之前提供的一致)
        weights = self._get_perceptual_curve(seq_len, fs)
        
        # 2. 注册为 buffer (会自动随模型移动到 GPU，且不作为可训练参数)
        # 归一化并防止除零，同时保留一点点底数(1e-2)给不敏感频段，避免完全无监督
        weights = weights / weights.max()
        weights = weights + 1e-2 
        self.register_buffer('weights', weights)

    def _get_perceptual_curve(self, seq_len, fs):
        # 原始实验数据
        x_original = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 130, 150, 170, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400])
        y_original = np.array([249.7768, 245.5580, 138.9063, 92.9464, 66.1830, 45.5357, 37.7232, 29.6875, 27.2991, 21.5402, 15.7143, 10.0893, 8.8929, 8.6027, 11.0268, 20.2679, 40.9598, 55.4688, 69.9777, 91.7857, 119.7545, 162.0455, 194.7727, 219.0909, 230.2272, 241.8182, 245.4546, 246.8182, 245.4546, 248.1818])

        # 计算灵敏度权重
        global_weights = 1.0 / (y_original + 1e-10)
        global_weights_norm = global_weights / global_weights.max()

        # 对齐到 FFT 频率点
        max_modes = seq_len // 2 + 1 
        index_array = np.arange(max_modes)
        freq_resolution = fs / seq_len
        network_actual_freq_hz = index_array * freq_resolution

        # 插值
        final_weights_array = np.interp(network_actual_freq_hz, x_original, global_weights_norm)
        
        return torch.tensor(final_weights_array, dtype=torch.float)
    
    def forward(self, pred, target):
        """
        输入: [Batch, 1, Seq_Len] 时域信号
        """
        # 1. 转频域 (RFFT) -> [Batch, 1, Freq_Bins] Complex
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        
        # 2. 取幅度 (Magnitude)
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # 3. 计算差异
        diff = torch.abs(pred_mag - target_mag)
        
        # 4. 加权 (广播权重)
        # diff: [B, 1, F], weights: [F] -> 自动广播
        weighted_loss = diff * self.weights
        
        return weighted_loss.mean()
    
class HapticStream(nn.Module):
    def __init__(
        self,
        *,
        channels: int = 16,
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        channel_mults: Tuple[int, ...] = (2, 2, 4, 4),
        pad_mode: str = 'constant',
        rvq_dim: int = 64,            # 输入RVQ的特征维度 (需与上一层输出通道数一致)
        codebook_size: int = 128,    # 码本大小 (通常 256, 1024 等)
        num_quantizers: int = 3,      # 残差层数 (层数越多重建越好，但码率越高)
        kmeans_init: bool = True,     # 是否使用 K-means 初始化
        squeeze_excite: bool = False,
        input_channels: int = 1,
        embedding_dim: int = 16,
        discr_multi_scales: Tuple[float, ...] = (1.0, 0.5, 0.25),
        recon_loss_weight: float = 10.0,
        adversarial_loss_weight: float = 1.0,
        feature_loss_weight: float = 1.0,
        SSIM_loss_weight: float = 0.1,
        perceptual_loss_weight: float = 5.0,
        commitment_weight: float = 1.0, # 承诺损失权重
        d_model: int = 16,
        n_heads: int = 4,
        d_keys: int = 4,
        fs: float = 2800.0
    ):
        super().__init__()

        _locals = locals()
        _locals.pop('self', None)
        _locals.pop('__class__', None)
        self._configs = pickle.dumps(_locals)

        self.recon_loss_weight = recon_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.SSIM_loss_weight = SSIM_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.single_channel = input_channels == 1
        self.strides = strides
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        # --- 判别器配置 (保持不变) ---
        self.discr_multi_scales = discr_multi_scales
        self.discriminators = nn.ModuleList(
            [MultiScaleDiscriminator() for _ in range(len(discr_multi_scales))]
        )
        discr_rel_factors = [
            int(s1 / s2) for s1, s2 in zip(discr_multi_scales[:-1], discr_multi_scales[1:])
        ]
        self.downsamples = nn.ModuleList(
            [nn.Identity()] + [nn.AvgPool1d(2 * factor, stride=factor, padding=factor) for factor in discr_rel_factors]
        )

        # --- 核心网络组件 ---
        
        # 1. 嵌入层
        self.embedding = DataEmbedding(c_in=input_channels, d_model=self.embedding_dim)

        # 2. Fourier Encoder
        # [Corrected]: 移除了不必要的 projection 层
        # 注意：这里 out_channels 设为 embedding_dim // 2，意味着输出维度减半
        self.encoder = FourierBlock(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            seq_len=64,
            scale=1.0 
        )

        # 3. Fourier Decoder
        # Decoder 接收的输入维度应该是 encoder 的输出维度吗？
        # 通常 Decoder 是镜像的，所以输入是 embedding_dim // 2，输出是 embedding_dim
        # 但看你的原始代码，Decoder 输入似乎是 embedding_dim。
        self.decoder = FourierBlock(
            in_channels=self.embedding_dim, # 注意这里，如果上一层输出了8，这里输入必须是8
            out_channels=self.embedding_dim,
            seq_len=64,
            n_heads=n_heads,
            scale=1.0
        )

        # 4. 因果卷积 (外部模块)
        self.encoder_causal_conv = ConvolutionalNetwork()
        self.decoder_causal_conv = DeconvolutionalNetwork()

        self.rvq = ResidualVQ(
            dim = rvq_dim,
            num_quantizers = num_quantizers,
            codebook_size = codebook_size,
            #quantizer_dropout = 0.5,# 训练时随机丢弃部分量化器以增强鲁棒性
            kmeans_init = kmeans_init,
            kmeans_iters = 10,
            threshold_ema_dead_code = 2, # 处理死码
            commitment_weight = 1.0      # 这里设为1，我们在外部再乘 self.commitment_weight
        )
        
        # 6. 最终输出层
        self.final_conv = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)

        # 7. 辅助学习器 (Tau/Delta)
        self.tau_learner = MLP_with_stats(output_dim=1)
        self.delta_learner = MLP_with_stats(output_dim=64)
        self.perceptual_loss_fn = PerceptualWeightedLoss(seq_len=64, fs=fs)
        # [Deleted]: self.query_projection (不再需要)

    # ... (non_discr_parameters 和 _standardize 保持不变) ...
    def non_discr_parameters(self):
        return [
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *self.embedding.parameters(),
            *self.encoder_causal_conv.parameters(),
            *self.decoder_causal_conv.parameters(),
            *(self.tau_learner.parameters() if exists(self.tau_learner) else []),
            *(self.delta_learner.parameters() if exists(self.delta_learner) else []),
            *self.final_conv.parameters(),
            *self.rvq.parameters() # 确保优化器包含 RVQ 参数
        ]
    
    @staticmethod
    def _standardize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = torch.mean(x, dim=-1, keepdim=True)
        centered = x - mean
        std = torch.sqrt(torch.var(centered, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        standardized = centered / std
        return standardized, mean.detach(), std.detach()

    # ... (_compute_discriminator_losses 保持不变) ...
    def _compute_discriminator_losses(self, orig_stationary: torch.Tensor, recon_stationary: torch.Tensor, apply_grad_penalty=False):
        real, fake = orig_stationary, recon_stationary.detach()
        discr_losses = []
        discr_grad_penalties = []
        scaled_real, scaled_fake = real, fake

        for discr, downsample in zip(self.discriminators, self.downsamples):
            scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))
            real_logits, fake_logits = map(partial(discr, return_intermediates=False), (scaled_real.requires_grad_(), scaled_fake))
            one_discr_loss = hinge_discr_loss(fake_logits, real_logits)
            discr_losses.append(one_discr_loss)
            if apply_grad_penalty:
                discr_grad_penalties.append(gradient_penalty(scaled_real, one_discr_loss))

        return discr_losses, discr_grad_penalties


    def forward(
        self,
        x: torch.Tensor,
        *,
        return_encoded: bool = False,
        return_recons_only: bool = False,
        return_discr_loss: bool = False,
        return_discr_losses_separately: bool = False,
        return_loss_breakdown: bool = False,
        apply_grad_penalty: bool = False,
        curtail_from_left: bool = False,
        attn_mask: Optional[torch.Tensor] = None
    ):
        raw_x = x.clone().detach()
        x_stationary, mean_raw_x, std_raw_x = self._standardize(x)

        tau = self.tau_learner(raw_x, std_raw_x).exp()
        delta = self.delta_learner(raw_x, mean_raw_x)

        # ---------------------------------------------------------
        # 1. Embedding
        # ---------------------------------------------------------
        # 假设 DataEmbedding 输出 [B, C, L] (例如 [256, 16, 64])
        x_embedding = self.embedding(x_stationary) 
        
        # ---------------------------------------------------------
        # 2. Encoder (FourierBlock)
        # ---------------------------------------------------------
        # 输入: [B, 16, 64] -> 输出: [B, 16, 64] 
        x_enc = self.encoder(x_embedding) 
        
        # ---------------------------------------------------------
        # 3. Encoder Causal Conv
        # ---------------------------------------------------------
        # [Critical]: 检查你的 ConvolutionalNetwork 期望什么维度
        # 如果它像大多数 Conv1d 模块一样期望 [B, C, L]，则不需要 transpose
        # 但看你之前的代码，你似乎在这里转置了。我假设它是为了适配外部模块。
        
        # 假设 causal_conv 期望 [B, L, C] (基于你旧代码的推断)
        x_enc = self.encoder_causal_conv(x_enc) # [B, 64, 2]
        x_enc = x_enc.permute(0, 2, 1)

        # ---------------------------------------------------------
        # 4. RVQ Quantization
        # ---------------------------------------------------------
        xhat, indices, commit_loss = self.rvq(x_enc)
        #print("indices", indices.shape)indices torch.Size([256, 2, 4])
        #exit()
        xhat = xhat.permute(0, 2, 1)#
        
        if return_encoded:
            return xhat, indices

        # ---------------------------------------------------------
        # 5. Decoder Causal Conv
        # ---------------------------------------------------------
        #这里输入[B,2,64]
        xhat = self.decoder_causal_conv(xhat) #[B ,16 ,64]
        # ---------------------------------------------------------
        # 6. Decoder (FourierBlock)
        # ---------------------------------------------------------
        
        # 调用 Decoder (单参数)
        xhat = self.decoder(xhat) # [B, C, L]

        # ---------------------------------------------------------
        # 7. Final Projection
        # ---------------------------------------------------------
        # self.final_conv 是 Conv1d(16, 1)，所以期望输入是 [B, 16, L]
        
        # 假设 xhat 是 [B, 16, L]
        
        xhat = self.final_conv(xhat) # [B, 1, L]

        # ---------------------------------------------------------
        # Post-Processing
        # ---------------------------------------------------------
        recon_stationary_x = xhat.clone()
        x_recon = recon_stationary_x * std_raw_x + mean_raw_x

        if return_recons_only:
            return x_recon

        # ... (后续 Loss 计算逻辑保持不变) ...
        # 仅复制 Loss 部分逻辑
        if return_discr_loss:
            discr_losses, discr_grad_penalties = self._compute_discriminator_losses(
                orig_stationary=x_stationary,
                recon_stationary=recon_stationary_x,
                apply_grad_penalty=apply_grad_penalty
            )
            if not return_discr_losses_separately:
                return torch.stack(discr_losses).mean()
            pkg = [(f'scale:{scale}', loss) for scale, loss in zip(self.discr_multi_scales, discr_losses)]
            pkg += [(f'scale_grad_penalty:{scale}', gp) for scale, gp in zip(self.discr_multi_scales, discr_grad_penalties)]
            return pkg

        recon_loss = F.mse_loss(x_stationary, recon_stationary_x)
        SSIM_loss = SSIM_1D_loss(recon_stationary_x, x_stationary)
        perceptual_loss = self.perceptual_loss_fn(recon_stationary_x, x_stationary)
        adversarial_losses = []
        discr_intermediates = []
        scaled_real, scaled_fake = x_stationary, recon_stationary_x
        for discr, downsample in zip(self.discriminators, self.downsamples):
            scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))
            (real_logits, real_intermediates), (fake_logits, fake_intermediates) = map(
                partial(discr, return_intermediates=True), (scaled_real, scaled_fake)
            )
            discr_intermediates.append((real_intermediates, fake_intermediates))
            adversarial_losses.append(hinge_gen_loss(fake_logits))

        feature_losses = []
        for real_intermediates, fake_intermediates in discr_intermediates:
            losses = [F.l1_loss(r, f) for r, f in zip(real_intermediates, fake_intermediates)]
            feature_losses.extend(losses)
            
        feature_loss = torch.stack(feature_losses).mean() if len(feature_losses) > 0 else torch.tensor(0.0, device=x.device)
        adversarial_loss = torch.stack(adversarial_losses).mean() if len(adversarial_losses) > 0 else torch.tensor(0.0, device=x.device)
        # commit_loss 来自 RVQ 内部，如果 num_quantizers > 1，这是所有层的 commit loss 之和
        total_commit_loss = commit_loss.sum() if hasattr(commit_loss, 'sum') else commit_loss

        total_loss = (recon_loss * self.recon_loss_weight
                      + adversarial_loss * self.adversarial_loss_weight
                      + feature_loss * self.feature_loss_weight
                      + SSIM_loss * self.SSIM_loss_weight
                      + total_commit_loss * self.commitment_weight
                      + perceptual_loss * self.perceptual_loss_weight)
                      
        if return_loss_breakdown:
            # 正确顺序：把 perceptual_loss 放到最后，和 train_wandb.py 的解包顺序一致
            return total_loss, (recon_loss, adversarial_loss, feature_loss, total_commit_loss, float(tau[0].item()), float(delta[0, 0].item()), perceptual_loss)
        return total_loss