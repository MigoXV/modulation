import os
import json
import torch
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# 定义信号生成器类
class wave_generator:

    def __init__(self, config):
        """初始化信号生成器"""
        self.sample_rate = config.sample_rate
        self.total_samples = config.total_samples
        self.selected_samples = config.selected_samples
        self.carrier_freq = config.carrier_freq
        self.carrier_peak_to_peak = config.carrier_peak_to_peak

        self.device = config.device

        self.ma_bottom = config.ma_bottom
        self.ma_top = config.ma_top

        self.mf_bottom = config.mf_bottom
        self.mf_top = config.mf_top
        self.F_max = config.F_max

        self.h_bottom = config.h_bottom
        self.h_top = config.h_top

        self.noise_factor = config.noise_factor

        # 生成时间序列张量供生成波形时使用
        self.t = (torch.arange(0, self.total_samples).float() / self.sample_rate).to(
            self.device
        )

    def save_dataset(self, data, labels, dataset_dir, dataset_name):

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        dataset_path = os.path.join(dataset_dir, dataset_name)
        torch.save((data,labels), dataset_path)

    def gene_data_and_labels(self, wave_list):

        labels = self.gene_labels(wave_list)
        data = self.gene_waves(wave_list)

        return data, labels

    def gene_labels(self, wave_list):
        """通过波形列表生成标签，标签为1维张量"""
        label_dict = {
            "cw": 0,
            "am": 1,
            "fm": 2,
            "2ask": 3,
            "2fsk": 4,
            "2psk": 5,
        }

        labels = torch.zeros(sum([wave["num"] for wave in wave_list]))
        start_idx = 0
        for wave in wave_list:
            labels[start_idx : start_idx + wave["num"]] = label_dict[wave["type"]]
            start_idx += wave["num"]

        return labels

    def gene_waves(self, wave_list):
        """生成所有类别的信号和标签，该函数遍历波形数组，返回信号张量和标签"""

        # 计算要生成的样本数的综合
        total_num = sum([wave["num"] for wave in wave_list])

        # 初始化波形张量
        waves = torch.zeros(total_num, self.selected_samples).to(self.device)

        start_index = 0

        # 遍历波形数组，生成信号
        for wave_gruop in tqdm(wave_list):

            # 切片
            wave_clip = waves[start_index : start_index + wave_gruop["num"]]

            # 生成波形
            self.gene_wave(wave_gruop, wave_clip)

            start_index += wave_gruop["num"]

        return waves

    def gene_wave(self, wave_group, wave_clip):

        if wave_group["type"] == "am":
            self.gene_am(wave_group, wave_clip)
        elif wave_group["type"] == "fm":
            self.gene_fm(wave_group, wave_clip)
        elif wave_group["type"] == "cw":
            self.gene_cw(wave_group, wave_clip)
        elif wave_group["type"] == "2ask":
            self.gene_2ask(wave_group, wave_clip)
        elif wave_group["type"] == "2fsk":
            self.gene_2fsk(wave_group, wave_clip)
        elif wave_group["type"] == "2psk":
            self.gene_2psk(wave_group, wave_clip)
        else:
            raise ValueError("Invalid wave type")

    def gene_am(self, wave_group, wave_clip):
        """生成am信号"""

        for index, wave in enumerate(wave_clip):

            # 随机生成调制指数
            ma = (torch.rand(1) * (self.ma_top - self.ma_bottom) + self.ma_bottom).to(
                self.device
            )

            # 生成调制信号
            modulation = (
                1 + ma * torch.sin(2 * np.pi * wave_group["freq"] * self.t)
            ).to(self.device)

            # 生成载波信号
            carrier = (
                self.carrier_peak_to_peak
                / 2
                * torch.sin(2 * np.pi * self.carrier_freq * self.t).to(self.device)
            )

            # 生成am信号
            full_wave = modulation * carrier

            # 随机化初相
            wave = self.random_phi(full_wave)

            # 添加噪声
            wave_clip[index] = self.add_noise(wave)

    def gene_fm(self, wave_group, wave_clip):
        """生成fm信号"""

        modulating_singal = torch.sin(2 * np.pi * wave_group["freq"] * self.t).to(
            self.device
        )

        for index, wave in enumerate(wave_clip):

            # 随机生成调制指数
            mf = torch.rand(1) * (self.mf_top - self.mf_bottom) + self.mf_bottom
            mf = mf.to(self.device)

            freq = self.carrier_freq + mf * self.F_max * modulating_singal

            # 生成fm信号
            full_wave = (
                self.carrier_peak_to_peak
                / 2
                * torch.sin(2 * np.pi * freq * self.t).to(self.device)
            )

            # 随机化初相
            wave = self.random_phi(full_wave)

            # 添加噪声
            wave_clip[index] = self.add_noise(wave)

    def gene_cw(self, wave_group, wave_clip):
        """生成cw信号"""

        for index, wave in enumerate(wave_clip):

            # 生成cw信号
            full_wave = (
                self.carrier_peak_to_peak
                / 2
                * torch.sin(2 * np.pi * self.carrier_freq * self.t).to(self.device)
            )

            # 随机化初相
            wave = self.random_phi(full_wave)

            # 添加噪声
            wave_clip[index] = self.add_noise(wave)

    def gene_2ask(self, wave_group, wave_clip):
        """生成2ask信号"""

        for index, wave in enumerate(wave_clip):

            # 生成调制信号，其为方波
            modulation = 0.5 * (
                1 + torch.sign(torch.sin(2 * np.pi * wave_group["freq"] * self.t))
            ).to(self.device)

            # 生成载波信号
            carrier = (
                self.carrier_peak_to_peak
                / 2
                * torch.sin(2 * np.pi * self.carrier_freq * self.t).to(self.device)
            )

            # 生成2ask信号
            full_wave = modulation * carrier

            # 随机化初相
            wave = self.random_phi(full_wave)

            # 添加噪声
            wave_clip[index] = self.add_noise(wave)

    def gene_2fsk(self, wave_group, wave_clip):
        """生成2fsk信号，发送0的载频值等于载波的频率，发送1的载频值等于载波频率呈上(1+h)"""

        # 调制信号
        modulating_singal = (
            1 - (torch.sign(torch.sin(2 * np.pi * wave_group["freq"] * self.t)) + 1) / 2
        ).to(self.device)

        for index, wave in enumerate(wave_clip):

            # 随机生成移频键控系数
            h = torch.rand(1) * (self.h_top - self.h_bottom) + self.h_bottom
            h = h.to(self.device)

            # 求出各点的频率
            freq = self.carrier_freq + h * wave_group["freq"] * modulating_singal
            freq = freq.to(self.device)
            # 生成2fsk信号
            full_wave = (
                self.carrier_peak_to_peak
                / 2
                * torch.sin(2 * np.pi * freq * self.t).to(self.device)
            )

            # 随机化初相
            wave = self.random_phi(full_wave)

            # 添加噪声
            wave_clip[index] = self.add_noise(wave)

    def gene_2psk(self, wave_group, wave_clip):
        """生成2psk信号"""

        # 调制信号
        modulating_singal = (
            1 - (torch.sign(torch.sin(2 * np.pi * wave_group["freq"] * self.t)) + 1) / 2
        ).to(self.device)

        for index, wave in enumerate(wave_clip):

            # 求出各点的相位
            phase = 2 * np.pi * self.carrier_freq * self.t + np.pi * modulating_singal
            phase.to(self.device)

            # 生成2psk信号
            full_wave = self.carrier_peak_to_peak / 2 * torch.sin(phase).to(self.device)

            # 随机化初相
            wave = self.random_phi(full_wave)

            # 添加噪声
            wave_clip[index] = self.add_noise(wave)

    def random_phi(self, wave):
        """通过在一个长的固定初相的波形中随机截取固定长度的波形来达到随机初相的效果"""

        diff = self.total_samples - self.selected_samples

        if diff:
            start_idx = torch.randint(0, diff, (1,)).item()
            wave = wave[start_idx : start_idx + self.selected_samples]

        return wave

    def add_noise(self, wave):
        """添加噪声"""

        # 生成噪声
        noise = torch.randn(wave.size()).to(self.device)

        # 标准化噪声
        noise = noise / torch.max(torch.abs(noise)) * self.noise_factor

        # 添加噪声
        wave = wave + noise

        return wave


def plot_waves(waves, wave_dir, sr):

    # 确认输出目录存在
    if not os.path.exists(wave_dir):
        os.makedirs(wave_dir)

    t = torch.arange(0, waves.size(1)).float() / sr

    # 遍历波形张量，为每个波形绘制图像
    for i, wave in enumerate(tqdm(waves)):

        plt.figure(figsize=(12, 8))
        plt.plot(t, wave.cpu().numpy())
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title(f"Wave {i}")
        plt.savefig(os.path.join(wave_dir, f"wave_{i}.png"))


def gene_waves_main(config_path, set=None, wave_dir=None):

    # 加载配置文件
    config = OmegaConf.load(config_path)

    if set:
        config = OmegaConf.merge(config, set)

    # 从config.wave_jsonl_path种加载波形字典
    wave_list = []

    with open(config.wave_jsonl_path, "r") as f:
        for line in f:
            wave_list.append(json.loads(line))

    # print(config)
    # print(wave_list)
    # 使用信号生成器生成信号
    generator = wave_generator(config)
    # waves = generator.gene_waves(wave_list)

    data, labels = generator.gene_data_and_labels(wave_list)
    generator.save_dataset(data, labels, config.output_dir, "dataset.pt")

    # plot_waves(waves, wave_dir, config.sample_rate)


if __name__ == "__main__":
    gene_waves_main("test/generator_config.yml", wave_dir="test/waves2")
