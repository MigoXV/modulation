import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 常量定义
CARRIER_FREQ = 2e6  # 载波频率，单位为Hz
PEAK_TO_PEAK = 1    # 峰-峰值幅度
SAMPLE_RATE = 32e6  # 采样率，单位为Hz
NUM_SAMPLES = 160000  # 每个波形的总样本数
SELECTED_SAMPLES = 128000  # 从起始的32000个样本中选择的样本数量

# 调制用的频率
AM_FM_FREQS = np.array([1e3, 2e3, 3e3, 4e3, 5e3])  # 单位为Hz
DIGITAL_MOD_FREQS = np.array([3e3, 4e3, 5e3])      # 单位为Hz

# 波形计数
NUM_WAVEFORMS = 6000  # 总波形数
AM_FM_PER_TYPE = 200  # 每种频率的AM/FM波形数
DIGITAL_MOD_COUNTS = np.array([330, 340, 330])  # 数字调制的波形数，每种频率的波形数

# 辅助函数，生成载波信号
def generate_carrier(num_samples):
    """生成载波信号。

    Args:
        num_samples (int): 生成的样本数。

    Returns:
        torch.Tensor: 生成的载波信号。
    """
    t = torch.arange(0, num_samples).float() / SAMPLE_RATE
    carrier = PEAK_TO_PEAK / 2 * torch.sin(2 * np.pi * CARRIER_FREQ * t).to(device)
    return carrier

# 调幅(AM)调制
def generate_am_waveform(mod_freq, mod_index, num_samples):
    """生成调幅(AM)波形。

    Args:
        mod_freq (float): 调制频率，单位为Hz。
        mod_index (float): 调制指数。
        num_samples (int): 生成的样本数。

    Returns:
        torch.Tensor: 生成的AM调制波形。
    """
    t = torch.arange(0, num_samples).float() / SAMPLE_RATE
    modulation = (1 + mod_index * torch.sin(2 * np.pi * mod_freq * t)).to(device)
    carrier = generate_carrier(num_samples)
    return modulation * carrier

# 调频(FM)调制
def generate_fm_waveform(mod_freq, mod_index, num_samples):
    """生成调频(FM)波形。

    Args:
        mod_freq (float): 调制频率，单位为Hz。
        mod_index (float): 调制指数。
        num_samples (int): 生成的样本数。

    Returns:
        torch.Tensor: 生成的FM调制波形。
    """
    t = torch.arange(0, num_samples).float() / SAMPLE_RATE
    carrier_phase = torch.cumsum(2 * np.pi * (CARRIER_FREQ + mod_index * torch.sin(2 * np.pi * mod_freq * t)), dim=0)
    fm_wave = torch.sin(carrier_phase).to(device)
    return fm_wave

# 数字调制辅助函数
def generate_digital_signal(freq, num_samples):
    """生成数字信号，用于ASK、FSK和PSK调制。

    Args:
        freq (float): 信号频率，单位为Hz。
        num_samples (int): 生成的样本数。

    Returns:
        torch.Tensor: 生成的数字信号。
    """
    t = torch.arange(0, num_samples).float() / SAMPLE_RATE
    period = int(SAMPLE_RATE / freq)
    digital_signal = torch.sign(torch.sin(2 * np.pi * freq * t)).to(device)
    return digital_signal

# ASK 调制
def generate_ask_waveform(freq, num_samples):
    """生成幅度键控（ASK）波形。

    Args:
        freq (float): 数字信号的频率，单位为Hz。
        num_samples (int): 生成的样本数。

    Returns:
        torch.Tensor: 生成的ASK调制波形。
    """
    digital_signal = generate_digital_signal(freq, num_samples)
    carrier = generate_carrier(num_samples)
    return digital_signal * carrier

# FSK 调制
def generate_fsk_waveform(freq, num_samples):
    """生成频率键控（FSK）波形。

    Args:
        freq (float): 基频差，单位为Hz。
        num_samples (int): 生成的样本数。

    Returns:
        torch.Tensor: 生成的FSK调制波形。
    """
    t = torch.arange(0, num_samples).float() / SAMPLE_RATE
    freq1 = CARRIER_FREQ - freq
    freq2 = CARRIER_FREQ + freq
    digital_signal = generate_digital_signal(freq, num_samples)
    fsk_wave = torch.where(digital_signal > 0, torch.sin(2 * np.pi * freq1 * t), torch.sin(2 * np.pi * freq2 * t)).to(device)
    return fsk_wave

# PSK 调制
def generate_psk_waveform(freq, num_samples):
    """生成相位键控（PSK）波形。

    Args:
        freq (float): 数字信号的频率，单位为Hz。
        num_samples (int): 生成的样本数。

    Returns:
        torch.Tensor: 生成的PSK调制波形。
    """
    digital_signal = generate_digital_signal(freq, num_samples)
    carrier = generate_carrier(num_samples)
    psk_wave = torch.where(digital_signal > 0, carrier, -carrier).to(device)
    return psk_wave

def generate_all_waveforms():
    """
    生成并存储所有波形。

    Returns:
        torch.Tensor: 存储所有波形的张量。
    """
    # 初始化一个张量，用于存储所有波形，维度为(NUM_WAVEFORMS, SELECTED_SAMPLES)
    waveforms = torch.zeros((NUM_WAVEFORMS, SELECTED_SAMPLES), device=device)
    
    # 索引变量，用于记录当前存储到张量中的波形位置
    idx = 0
    
    # 遍历所有AM和FM频率，为每个频率生成调制波形
    for freq in AM_FM_FREQS:
        # 为AM调制生成随机调制指数，范围从0.3到1.0
        mod_index_am = torch.rand(AM_FM_PER_TYPE).uniform_(0.3, 1.0).to(device)
        # 为FM调制生成随机调制指数，范围从1到5
        mod_index_fm = torch.rand(AM_FM_PER_TYPE).uniform_(1, 5).to(device)
        # 为当前频率生成AM和FM波形
        for i in tqdm(range(AM_FM_PER_TYPE), desc=f"Generating AM and FM for {freq} Hz"):
            # 生成AM波形并截取所需样本
            am_wave = generate_am_waveform(freq, mod_index_am[i], NUM_SAMPLES)[32000:32000+SELECTED_SAMPLES]
            # 生成FM波形并截取所需样本
            fm_wave = generate_fm_waveform(freq, mod_index_fm[i], NUM_SAMPLES)[32000:32000+SELECTED_SAMPLES]
            # 将生成的AM和FM波形存储到waveforms张量中
            waveforms[idx] = am_wave
            waveforms[idx+1] = fm_wave
            idx += 2  # 更新索引位置，每次增加2

    # 遍历所有数字调制频率，生成ASK, FSK和PSK波形
    for i, freq in enumerate(DIGITAL_MOD_FREQS):
        # 为每个频率生成对应的数字调制波形
        for j in tqdm(range(DIGITAL_MOD_COUNTS[i]), desc=f"Generating digital modulations for {freq} Hz"):
            # 生成ASK波形并截取所需样本
            ask_wave = generate_ask_waveform(freq, NUM_SAMPLES)[32000:32000+SELECTED_SAMPLES]
            # 生成FSK波形并截取所需样本
            fsk_wave = generate_fsk_waveform(freq, NUM_SAMPLES)[32000:32000+SELECTED_SAMPLES]
            # 生成PSK波形并截取所需样本
            psk_wave = generate_psk_waveform(freq, NUM_SAMPLES)[32000:32000+SELECTED_SAMPLES]
            # 将生成的ASK, FSK和PSK波形存储到waveforms张量中
            waveforms[idx] = ask_wave
            waveforms[idx+1] = fsk_wave
            waveforms[idx+2] = psk_wave
            idx += 3  # 更新索引位置，每次增加3

    # 返回存储所有波形的张量
    return waveforms


# 示例：生成并绘制波形
def plot_waveform(img_dir):
    """从6种波形种中，每种波形随机选取5个样本并绘图。

    Args:
        img_path (str): 文件的保存路径。
    """
    
    # 从6种波形中，每种波形随机选取5个样本
    selected_idx = torch.randint(0, NUM_WAVEFORMS, (6, 5))
    
    # 绘制波形，每个波形都单独保存为png文件
    for i in range(6):
        plt.figure(figsize=(10, 6))
        for j in range(5):
            plt.subplot(2, 3, j+1)
            plt.plot(waveforms[selected_idx[i, j]].cpu().numpy())
            plt.title(f"Waveform {selected_idx[i, j]}")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{img_dir}/waveform_{i}.png")
        plt.close()

    

if __name__ == "__main__":

    # 生成所有的波形，返回的是一个二维张量
    waveforms = generate_all_waveforms()
    
    # 保存波形数据
    torch.save(waveforms, "data/waveforms.pt")
    
    # 从6种波形中，每种波形随机选取5个样本并绘图
    