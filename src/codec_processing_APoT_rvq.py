import pickle
import torch
import scipy.io as scio
import numpy as np
import torch.nn as nn
from functools import partial

from vector_quantize_pytorch import FSQ
from model.custom_Loss import hinge_discr_loss
import torch_dct
from model.custom_Module import DataEmbedding
from src.binary_converter import bit2float, float2bit
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import StaticModel
import struct
from itertools import cycle

from src.FourierCorrelation import FourierBlock
from src.convlayer import ConvolutionalNetwork
from src.Transpose_convlayer import DeconvolutionalNetwork

# 【新增】必须导入模型类，否则无法实例化
from model.hapticstream_fourier import HapticStream 

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成从 -128 到 127 的所有整数的 8 位二进制表示
result_list = ['{0:08b}'.format(i & 0xff) for i in range(-128, 128)]
result_list = [item.replace('0', '0').replace('1', '1') for item in result_list]

def mean_float_to_12bit(num):
    if num < -1 or num > 1:
        raise ValueError("Number out of range (-1, 1)")

    # Determine the sign bit (1 for negative, 0 for positive)
    sign_bit = 1 if num < 0 else 0

    # Take the absolute value for further processing
    abs_num = abs(num)

    # Determine the integer bit (1 if abs_num >= 0.5, else 0)
    int_bit = 1 if abs_num >= 0.5 else 0

    # Calculate the fractional part by removing the integer part
    fractional_part = abs_num - int_bit * 0.5

    # Scale the fractional part to fit into 10 bits
    scaled_fraction = int(fractional_part * 1024)

    # Convert to binary representation
    fractional_bits = format(scaled_fraction, '010b')

    # Combine the sign bit, integer bit, and fractional bits
    bit_representation = f'{sign_bit}{int_bit}{fractional_bits}'

    return bit_representation


def mean_bit_to_float(bit_str):
    if len(bit_str) != 12 or not all(c in '01' for c in bit_str):
        raise ValueError("Input must be a 12-bit binary string")

    # Extract sign, integer, and fractional bits
    sign_bit = int(bit_str[0])
    int_bit = int(bit_str[1])
    fractional_bits = bit_str[2:]

    # Convert fractional bits to a number
    fractional_part = int(fractional_bits, 2) / 1024.0

    # Calculate the decimal value
    num = int_bit * 0.5 + fractional_part

    # Apply the sign
    if sign_bit == 1:
        num = -num

    return num


def std_float_to_12bit(num):
    if num < -0.5 or num > 0.5:
        raise ValueError("Number out of range. Must be between -0.5 and 0.5")

    # Determine the sign bit
    if num < 0:
        sign_bit = '1'
        num = -num  # Make the number positive for further processing
    else:
        sign_bit = '0'

    # Scale the number to fit in 11 bits
    scaled_num = int(num * (2 ** 11))

    # Convert the scaled number to binary and pad to 11 bits
    fractional_bits = format(scaled_num, '011b')

    # Combine the sign bit and fractional bits
    binary_representation = sign_bit + fractional_bits

    return binary_representation


def std_bit_to_float(binary_code):
    if len(binary_code) != 12:
        raise ValueError("Binary code must be 12 bits long")

    # Extract the sign bit and fractional bits
    sign_bit = binary_code[0]
    fractional_bits = binary_code[1:]

    # Convert the fractional bits to an integer
    fractional_num = int(fractional_bits, 2)

    # Scale the number back to the original range
    num = fractional_num / (2**11)

    # Apply the sign
    if sign_bit == '1':
        num = -num

    return num




class Codec_processing():
    def __init__(self,
                 model_path,
                 huffman_codes_path,
                 in_channels=16,
                 out_channels=8,
                 seq_len=64,
                 modes=30,
                 mode_select_method='random',
                 n_heads=4,
                 indices_shape=4, # 注意：这里如果 indices 数量变了，decode 里的 view 可能要改
                 tolerance=0.3,
                 embedding_dim=16,
                 FSQ_levels=None,
                 d_model=16,
                 d_keys = 4,
                 residual_quantization_tuner=torch.tensor([0.03, 0.05, 0.075, 0.1, 0.2, 0.3, 0.03, 1]),
                 ):
        super(Codec_processing, self).__init__()

        if FSQ_levels is None:
            FSQ_levels = [8, 5, 5, 5]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tolerance = tolerance
        self.indices_shape = indices_shape

        # =================【核心修改开始】=================
        print(f"Loading model from: {model_path}")
    
    # 1. 加载 checkpoint 字典
        checkpoint = torch.load(model_path, map_location=self.device)

    # 2. 准备配置字典 (新增了处理 bytes 的逻辑)
        config_data = {}
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config_raw = checkpoint['config']
        
        # 【修复关键点】检查是否为 bytes，如果是则反序列化
            if isinstance(config_raw, bytes):
                print("Config found as bytes, decoding via pickle...")
                try:
                    config_data = pickle.loads(config_raw)
                except Exception as e:
                    print(f"Error unpickling config: {e}")
                    config_data = {} # 失败则回退
            elif isinstance(config_raw, dict):
                config_data = config_raw
            else:
                print(f"Warning: Unknown config type: {type(config_raw)}")
            
    # 3. 实例化模型结构
        if config_data:
            print("Found config in checkpoint, initializing model with saved config...")
        # 此时 config_data 必定是字典，不会报错了
            self.model_total = HapticStream(**config_data)
        else:
            print("No valid config found, initializing model with default arguments...")
            self.model_total = HapticStream() 

    # 4. 加载权重
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model_total.load_state_dict(checkpoint['model'])
        else:
            self.model_total.load_state_dict(checkpoint)

    # 5. 转移到 GPU 并开启评估模式
        self.model_total.to(self.device)
        self.model_total.eval()
       # =================【核心修改结束】=================

        try:
            self.huffman_codec = np.load(huffman_codes_path, allow_pickle=True).item()
        except FileNotFoundError:
            print(f"Warning: Huffman codes file not found at {huffman_codes_path}")
            self.huffman_codec = {}


    def encode(self, data_orig):
        x = data_orig.to(self.device) # 使用 self.device 更通用
 
        # Series Stationarization
        mean_raw_x = torch.mean(x, dim=-1, keepdim=True).detach()
        x = x - mean_raw_x
        std_raw_x = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std_raw_x
        x_stationary = x

        # Forward Pass
        # 此时 self.model_total 是 HapticStream 实例，拥有 embedding 方法
        x_embedding = self.model_total.embedding(x) 
       # x_embedding = x_embedding.transpose(1, 2)

       # #batch_size = x_embedding.shape[0]
       # proj = self.model_total.query_projection(x_embedding)
        #seq_len = proj.shape[1]
        
        # 注意：这里硬编码了 4，如果 query 维度变了这里会报错
      #  queries = proj.view(batch_size, seq_len, 4, -1) 

        x_enc =  self.model_total.encoder(x_embedding)

 

        x_enc = self.model_total.encoder_causal_conv(x_enc)
        #print(f"Before RVQ shape: {x_enc.shape}")
        #exit()
        x = x_enc
        x = x.permute(0, 2, 1)
        xhat, indices, commit_loss = self.model_total.rvq(x)
        #xhat = xhat.permute(0, 2, 1)
        return indices, mean_raw_x, std_raw_x, x_stationary

    def decode(self, indices, mean, std):
        xhat = self.model_total.rvq.get_output_from_indices(indices)
        

        #xhat = xhat.view(1, 4, 4) 
        xhat = xhat.permute(0, 2, 1)
        xhat = self.model_total.decoder_causal_conv(xhat)
       # xhat = xhat.transpose(1, 2)

        xhat= self.model_total.decoder(xhat)

       # xhat = xhat.transpose(1, 2)
       # xhat = xhat.reshape(xhat.size(0), -1, xhat.size(-1))
        xhat = self.model_total.final_conv(xhat)
        
        xhat = xhat.clone()
        return xhat

    def discriminator(self, real_data, fake_data):
        vote_score = 0
        value_per_discriminator = [4, 2, 1]
        index = 0
        scaled_real, scaled_fake = real_data, fake_data
        scaled_real, scaled_fake = scaled_real.to(self.device), scaled_fake.to(self.device)
        
        # 此时 self.model_total 已经包含了 discriminators
        for discr, downsample in zip(self.model_total.discriminators, self.model_total.downsamples):
            scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))
            (real_logits, fake_logits) = map(partial(discr, return_intermediates=False), (scaled_real, scaled_fake))
            discr_loss = hinge_discr_loss(fake_logits, real_logits)
            if abs(discr_loss - 2) <= self.tolerance:
                vote_score = vote_score + value_per_discriminator[index]
            index = index + 1
        return vote_score

    def indices_huffman_encode(self, indices):
        """
        Huffman 编码
        indices: 输入张量，形状可以是 (1, T) 或 (1, T, Layers)
        """
        len_bits = 0
        string_bits = ''
        
        valid_keys = list(self.huffman_codec.keys())
        if not valid_keys: # 防止空字典报错
             print("Error: Huffman codec is empty!")
             return "", 0
             
        fallback_key = max(valid_keys) 
        fallback_bits = self.huffman_codec[fallback_key]
        warning_printed = False 

        # === 修正 1: 强制展平 ===
        # 无论输入是 (1, 4) 还是 (1, 4, 8)，都展平为一维列表
        # 这样就能逐个处理标量，避免 "Tensor with 8 elements" 报错
        indices_flat = indices.view(-1).tolist()

        for idx_val in indices_flat:
            # 确保是 int 类型
            indices_int = int(idx_val)
            
            bits = self.huffman_codec.get(indices_int)
            
            if bits is None:
                if not warning_printed:
                    print(f"WARNING: Value {indices_int} unknown, replacing with fallback key {fallback_key}.")
                    warning_printed = True
                bits = fallback_bits

            string_bits = string_bits + bits
            len_bits = len_bits + len(bits)
            
        return string_bits, len_bits


    def indices_huffman_decode(self, string_bits, length=None):
        """
        Huffman 解码
        string_bits: 二进制字符串
        length: 需要解码出的符号数量 (对应 RVQ 的总长度，例如 4*8=32)
        """
        # === 修正 2: 动态长度 ===
        # 如果传入了 length，就用传入的；否则回退到默认的 self.indices_shape
        if length is not None:
            target_len = length
        else:
            # 兼容旧逻辑
            target_len = np.prod(self.indices_shape) if isinstance(self.indices_shape, tuple) else self.indices_shape

        # 初始化为一维 Tensor，长度为 target_len
        decoded_indices = torch.zeros(target_len).type(torch.int32).to(self.device)
        
        flipped_dict = {value: key for key, value in self.huffman_codec.items()}
        index = 0
        current_code = ""
        
        for bit in string_bits:
            # === 修正 3: 使用动态长度做边界检查 ===
            if index >= target_len:
                break 

            current_code += bit
            char = flipped_dict.get(current_code)
            
            if char is not None:
                decoded_indices[index] = char
                index = index + 1
                current_code = ""
        
        # 返回形状 (1, N)，方便外部后续 Reshape
        decoded_indices = torch.unsqueeze(decoded_indices, dim=0)
        return decoded_indices