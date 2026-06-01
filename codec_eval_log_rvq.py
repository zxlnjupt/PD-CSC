import torch
import os
import numpy as np
import torch.nn.functional as F
import scipy.io as scio
import sys
import glob
from pathlib import Path
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.codec_processing_APoT_rvq import Codec_processing, mean_float_to_12bit, std_float_to_12bit ,mean_bit_to_float, std_bit_to_float
from src.residual_huffman import indices_huffman_encode, indices_huffman_decode
import src.residual_huffman as residual_huffman_module
import openpyxl
import matplotlib.pyplot as plt
import huffman
from scipy.signal import filtfilt, butter
from test_codec_APoT.find_new_res_quanti import log_quantization_fixed, log_dequantization_fixed, quantize_scale_factor, decode_scale_factor, adaptive_uniform_quantization, adaptive_dequantization
from vibromaf.metrics.stsim import st_sim
from vibromaf.metrics.snr import snr, psnr
BIT_CONFIGS = {
    3: {'WEIGHT_FLOOR': 0.50, 'DEAD_ZONE_RATIO': 0.10, 'ABSOLUTE_NOISE_FLOOR': 0.05},
    4: {'WEIGHT_FLOOR': 0.50, 'DEAD_ZONE_RATIO': 0.10, 'ABSOLUTE_NOISE_FLOOR': 0.05},
    5: {'WEIGHT_FLOOR': 0.70, 'DEAD_ZONE_RATIO': 0.05, 'ABSOLUTE_NOISE_FLOOR': 0.02},
    6: {'WEIGHT_FLOOR': 0.70, 'DEAD_ZONE_RATIO': 0.01, 'ABSOLUTE_NOISE_FLOOR': 0.01},
              }

class DFT321Processor:
    """
    DFT321变换处理器类
    用于批量处理三轴触觉数据
    """

    fs = 2800.0


    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, Wn=normal_cutoff, btype='low', analog=False)
        return b, a


    def butter_lowpass_filtfilt(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order)
        y = filtfilt(b, a, data)
        return y

    def smtdA(self, txt_data):

        N = len(txt_data)
        txt_data = txt_data / 1.6
        Ax = np.fft.fft(txt_data)
        fs = 2800.0
        Ax_abs = np.abs(Ax)
        theta = np.angle(Ax)
        cutoff = 600
        condA = self.butter_lowpass_filtfilt(Ax_abs, cutoff, fs)
        condA_real = condA * np.cos(theta)
        condA_imag = condA * np.sin(theta)

        return condA, condA_real, condA_imag

    def __init__(self, frame_length=64, overlap_ratio=0.0, fs=2800.0, cutoff=600):
        """
        初始化处理器

        参数:
            frame_length: 帧长度，默认64
            overlap_ratio: 重叠比例，默认0.0
            fs: 采样频率，默认2800.0 Hz
            cutoff: 低通滤波截止频率，默认600 Hz
        """
        self.frame_length = frame_length
        self.overlap_ratio = overlap_ratio
        self.fs = fs
        self.cutoff = cutoff
        self.length_buffer = int(frame_length * (1 - overlap_ratio))

    def process_single_axis(self, data):
        """
        对单轴数据进行DFT321变换
        """
        Ax, real_part, imag_part = self.smtdA(data)
        return Ax, real_part, imag_part

    def process_three_axis(self, data_X, data_Y, data_Z):
        """
        对三轴数据进行DFT321变换并合成
        """
        # 分别处理三个轴
        Ax, real_x, imag_x = self.process_single_axis(data_X)
        Ay, real_y, imag_y = self.process_single_axis(data_Y)
        Az, real_z, imag_z = self.process_single_axis(data_Z)

        # 三轴合成
        As = np.sqrt(np.abs(Ax)**2 + np.abs(Ay)**2 + np.abs(Az)**2)
        real_sum = real_x + real_y + real_z
        imag_sum = imag_x + imag_y + imag_z
        theta = np.angle(real_sum + 1j * imag_sum)

        # 逆傅里叶变换得到最终信号
        processed_data = np.fft.ifft(As * np.exp(1j * theta)).real

        return processed_data

    def process_file_set(self, x_file, y_file, z_file):
        """
        处理一组三轴文件并返回分帧后的数据
        """
        try:
            # 加载数据
            data_X = np.loadtxt(x_file)
            data_Y = np.loadtxt(y_file)
            data_Z = np.loadtxt(z_file)

            # 确保数据长度一致
            min_length = min(len(data_X), len(data_Y), len(data_Z))
            data_X = data_X[:min_length]
            data_Y = data_Y[:min_length]
            data_Z = data_Z[:min_length]

            # 计算帧数
            num_windows = (len(data_X) - self.frame_length) // self.length_buffer + 1

            if num_windows == 0:
                print(f"警告: 数据长度不足，无法生成帧: {x_file}")
                return [], (data_X, data_Y, data_Z)

            frames = []

            # 处理每一帧
            for i in range(0, num_windows * self.length_buffer, self.length_buffer):
                frame_X = data_X[i:i + self.frame_length]
                frame_Y = data_Y[i:i + self.frame_length]
                frame_Z = data_Z[i:i + self.frame_length]

                # DFT321变换
                processed_frame = self.process_three_axis(frame_X, frame_Y, frame_Z)
                frames.append(processed_frame)

            return frames, (data_X, data_Y, data_Z)

        except Exception as e:
            print(f"处理文件时出错 {x_file}: {str(e)}")
            return [], None

    def reconstruct_signal(self, frames, overlap_ratio=0.0):
        """
        从分帧数据重构完整信号
        """
        if not frames:
            return np.array([])

        frame_length = len(frames[0])
        length_buffer = int(frame_length * (1 - overlap_ratio))

        # 计算重构信号的总长度
        total_length = length_buffer * (len(frames) - 1) + frame_length
        reconstructed = np.zeros(total_length)

        # 重构信号
        for i, frame in enumerate(frames):
            start_idx = i * length_buffer
            end_idx = start_idx + frame_length
            reconstructed[start_idx:end_idx] = frame

        return reconstructed


fold_name = 'fold_4'
input_dir = f'./cross_validation_data/{fold_name}/' 
excel_out_folder = f'./outdataset_rvq_128_3_520/'
frame_length = 64
overlap_ratio = 0.0
pattern = "*_X.txt"
sample_rate = 2800.0
cutoff_freq = 600.0
model_path = r'E:\DNA code\chujue\checkpoints_new\fold4_128_3_517\best_model_recon.pt'
FSQ_huffman_path = r'E:\DNA code\chujue\codebook\codebook_rvq_128_3_517\RVQ_model_huffman_codes_128size.npy'


processor = DFT321Processor(
    frame_length=frame_length,
    overlap_ratio=overlap_ratio,
    fs=sample_rate,
    cutoff=cutoff_freq
)

processing = Codec_processing(model_path, FSQ_huffman_path)
create_huffman = False


def get_perceptual_curve(seq_len=64, fs=2800.0):
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

def soften_weights(weights, floor=0.85):
    return weights * (1.0 - floor) + floor



def apply_frequency_weighting(time_signal, freq_weights, mode='mul'):
    if time_signal.device != freq_weights.device:
        freq_weights = freq_weights.to(time_signal.device)
    signal_fft = torch.fft.rfft(time_signal)
    if mode == 'mul':
        signal_fft_processed = signal_fft * freq_weights
    elif mode == 'div':
        signal_fft_processed = signal_fft / (freq_weights + 1e-8)
    else:
        raise ValueError("Mode must be 'mul' or 'div'")
    signal_processed = torch.fft.irfft(signal_fft_processed, n=len(time_signal))
    return signal_processed


def run_single_bit(target_bit, x_files, cfg):
    """对指定bit级别运行完整评估，返回填好数据的workbook。"""
    WEIGHT_FLOOR = cfg['WEIGHT_FLOOR']
    DEAD_ZONE_RATIO = cfg['DEAD_ZONE_RATIO']
    ABSOLUTE_NOISE_FLOOR = cfg['ABSOLUTE_NOISE_FLOOR']

    workbook = openpyxl.Workbook()
    sheet = workbook.active

    titles = [
        'Path',
        'total_kbps',
        'model_kbps',
        'residual_kbps',
        'mse_with_residual', 'mse_without_residual',
        'psnr_with_residual', 'psnr_without_residual',
        'stsim_with_residual', 'stsim_without_residual',
        'snr_with_residual', 'snr_without_residual',
        'avg_kbps'
    ]
    for col_num, title in enumerate(titles, 1):
        sheet.cell(row=1, column=col_num).value = title

    bit_width_cost_map = {0: 1, 3: 2, 4: 3, 5: 4, 6: 5}

    raw_perceptual_curve = get_perceptual_curve(seq_len=64, fs=2800.0)
    soft_perceptual_curve = soften_weights(raw_perceptual_curve, floor=WEIGHT_FLOOR)

    total_bits_sum = 0
    total_framed_data_count = 0

    for file_idx, x_file in enumerate(tqdm(x_files, desc=f"bit={target_bit} 处理文件")):
        y_file = x_file.replace('_X', '_Y')
        z_file = x_file.replace('_X', '_Z')

        if not os.path.exists(y_file) or not os.path.exists(z_file):
            print(f"警告: 找不到对应的Y或Z轴文件: {x_file}")
            continue

        base_name = Path(x_file).stem.replace('_X', '')

        frames, original_data = processor.process_file_set(x_file, y_file, z_file)

        if not frames:
            continue

        reconstructed_frames = []
        residual_frames = []
        data_recon_frames = []
        raw_residual_frames = []

        file_frame_total_bits = []
        file_frame_model_bits = []
        file_frame_residual_bits = []

        for frame_idx, frame in enumerate(frames):
            data_orig = torch.from_numpy(frame).reshape((1, 1, 64)).float()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data_orig = data_orig.to(device)

            indices, mean, std, x_stationary = processing.encode(data_orig)
            indices_flat = indices.view(indices.shape[0], -1)

            indices_bits, len_indices_bits = processing.indices_huffman_encode(indices_flat)
            decoded_indices_flat = processing.indices_huffman_decode(indices_bits, length=indices_flat.shape[1])
            decoded_indices = decoded_indices_flat.view(indices.shape)

            mean_bit_string = mean_float_to_12bit(mean)
            if std >= 0.5: std = 0.4999
            elif std <= -0.5: std = -0.4999
            std_bit_string = std_float_to_12bit(std)
            mean_rec = mean_bit_to_float(mean_bit_string)
            std_rec = std_bit_to_float(std_bit_string)

            current_frame_model_bits = len_indices_bits + len(mean_bit_string) + len(std_bit_string)
            xhat_stationary = processing.decode(decoded_indices, mean_rec, std_rec)

            residual_signal = ((x_stationary - xhat_stationary)[0][0])
            raw_scale = torch.max(torch.abs(residual_signal))
            decoded_scale, scale_bits = quantize_scale_factor(raw_scale)

            residual_weighted = apply_frequency_weighting(
                residual_signal,
                soft_perceptual_curve,
                mode='mul'
            )

            weighted_max = torch.max(torch.abs(residual_weighted))
            threshold = max(DEAD_ZONE_RATIO * weighted_max.item(), ABSOLUTE_NOISE_FLOOR)

            residual_to_quantize = residual_weighted.clone()
            residual_to_quantize[torch.abs(residual_to_quantize) < threshold] = 0.0

            is_all_zero = torch.sum(torch.abs(residual_to_quantize)) == 0

            if is_all_zero:
                bit = 0
            else:
                bit = target_bit

            len_bit_width_flag = bit_width_cost_map.get(bit, 0)
            current_frame_residual_data_bits = 0
            residual_signal_hat_weighted = torch.zeros_like(residual_signal)

            if bit != 0:
                x_indices_tensor, sign, _ = adaptive_uniform_quantization(
                    residual_to_quantize,
                    decoded_scale,
                    bits=bit
                )
                residual_indices = x_indices_tensor.view(-1).tolist()
                residual_indices_bits, residual_len_bits = indices_huffman_encode(residual_indices, bit)

                current_frame_residual_data_bits = residual_len_bits + scale_bits

                decoded_residual_indices = indices_huffman_decode(residual_indices_bits, bit)
                decoded_indices_tensor = torch.tensor(decoded_residual_indices, dtype=torch.long, device=residual_signal.device).view(residual_signal.shape)

                residual_signal_hat_weighted = adaptive_dequantization(
                    indices=decoded_indices_tensor,
                    sign=sign,
                    scale=decoded_scale,
                    bits=bit
                )
            else:
                residual_indices = []

            residual_signal_hat = apply_frequency_weighting(
                residual_signal_hat_weighted,
                soft_perceptual_curve,
                mode='div'
            )

            current_frame_residual_total_bits = current_frame_residual_data_bits + len_bit_width_flag
            current_frame_total_bits = current_frame_model_bits + current_frame_residual_total_bits

            file_frame_total_bits.append(current_frame_total_bits)
            file_frame_model_bits.append(current_frame_model_bits)
            file_frame_residual_bits.append(current_frame_residual_total_bits)

            data_recon = xhat_stationary * std_rec + mean_rec
            residual_signal_hat_denorm = residual_signal_hat * std_rec
            raw_residual_denorm = residual_signal * std_rec
            data_recon_total = data_recon + residual_signal_hat_denorm

            reconstructed_frames.append(data_recon_total.cpu().detach().numpy().reshape(-1))
            residual_frames.append(residual_signal_hat_denorm.cpu().detach().numpy().reshape(-1))
            data_recon_frames.append(data_recon.cpu().detach().numpy().reshape(-1))
            raw_residual_frames.append(raw_residual_denorm.cpu().detach().numpy().reshape(-1))

        reconstructed_signal = processor.reconstruct_signal(reconstructed_frames, overlap_ratio)
        recon_without_residual_signal = processor.reconstruct_signal(data_recon_frames, overlap_ratio)
        residual_signal = processor.reconstruct_signal(residual_frames, overlap_ratio)

        original_frames = []
        for i in range(0, len(frames) * processor.length_buffer, processor.length_buffer):
            original_frames.append(frames[i // processor.length_buffer])
        original_signal = processor.reconstruct_signal(original_frames, overlap_ratio)

        min_length = min(len(original_signal), len(reconstructed_signal), len(recon_without_residual_signal))
        original_signal = original_signal[:min_length]
        reconstructed_signal = reconstructed_signal[:min_length]
        recon_without_residual_signal = recon_without_residual_signal[:min_length]

        mse_with_residual = np.mean((original_signal - reconstructed_signal) ** 2)
        mse_without_residual = np.mean((original_signal - recon_without_residual_signal) ** 2)

        psnr_with_residual = psnr(reconstructed_signal, original_signal)
        psnr_without_residual = psnr(recon_without_residual_signal, original_signal)

        stsim_with_residual = st_sim(reconstructed_signal, original_signal)
        stsim_without_residual = st_sim(recon_without_residual_signal, original_signal)

        snr_with_residual = snr(reconstructed_signal, original_signal)
        snr_without_residual = snr(recon_without_residual_signal, original_signal)

        total_bit_sum_file = sum(file_frame_total_bits)
        model_bit_sum_file = sum(file_frame_model_bits)
        residual_bit_sum_file = sum(file_frame_residual_bits)

        file_time_duration = (len(frames) * frame_length) / sample_rate

        if file_time_duration > 0:
            file_total_kbps = (total_bit_sum_file / 1024.0) / file_time_duration
            file_model_kbps = (model_bit_sum_file / 1024.0) / file_time_duration
            file_residual_kbps = (residual_bit_sum_file / 1024.0) / file_time_duration
        else:
            file_total_kbps = 0
            file_model_kbps = 0
            file_residual_kbps = 0

        total_bits_sum += total_bit_sum_file
        total_framed_data_count += len(frames)

        row = file_idx + 2
        sheet.cell(row, 1, base_name)
        sheet.cell(row, 2, file_total_kbps)
        sheet.cell(row, 3, file_model_kbps)
        sheet.cell(row, 4, file_residual_kbps)
        sheet.cell(row, 5, mse_with_residual)
        sheet.cell(row, 6, mse_without_residual)
        sheet.cell(row, 7, psnr_with_residual)
        sheet.cell(row, 8, psnr_without_residual)
        sheet.cell(row, 9, stsim_with_residual)
        sheet.cell(row, 10, stsim_without_residual)
        sheet.cell(row, 11, snr_with_residual)
        sheet.cell(row, 12, snr_without_residual)

    if total_framed_data_count > 0:
        total_time_duration = (total_framed_data_count * 64) / 2800.0
        avg_kbps = (total_bits_sum / 1024.0) / total_time_duration
        sheet.cell(2, 13, avg_kbps)

    return workbook


if __name__ == '__main__':
    search_pattern = os.path.join(input_dir, pattern)
    x_files = glob.glob(search_pattern)

    if not x_files:
        print(f"在目录 {input_dir} 中没有找到匹配模式 {pattern} 的文件")
        exit(1)

    print(f"找到 {len(x_files)} 个文件进行处理")

    os.makedirs(excel_out_folder, exist_ok=True)

    for target_bit in [3, 4, 5, 6]:
        cfg = BIT_CONFIGS[target_bit]
        wb = run_single_bit(target_bit, x_files, cfg)
        excel_path = os.path.join(excel_out_folder, f'{fold_name}.xlsx')
        # 文件名与原来相同，按bit级别区分存放在同一目录下
        # 为避免四个bit结果互相覆盖，在文件名中加入bit标识
        excel_path = os.path.join(excel_out_folder, f'{fold_name}_bit{target_bit}.xlsx')
        wb.save(excel_path)
        wb.close()
        print(f"bit={target_bit} 处理完成，结果已保存至 {excel_path}")
