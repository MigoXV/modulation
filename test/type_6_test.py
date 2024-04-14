# 将本文件的父目录加到工作路径中
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import data_feeder.gene_waves as gene
import matplotlib.pyplot as plt

def plot_wave(img_path, waveforms):
    """绘制单个波形并保存为"""
    plt.figure(figsize=(10, 6))
    plt.plot(waveforms)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.savefig(img_path)
    
    
def test_main():
    # 测试载波生成功能
