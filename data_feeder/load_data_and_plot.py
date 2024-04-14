import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



# TODO: 适配新的数据集格式
if __name__=="__main__":
    
    # 加载数据
    data_path = "data/waveforms.pt"
    waveforms = torch.load(data_path)
    
    # 随机抽取20个波形并显示
    selected_idx = torch.randint(0, waveforms.size(0), (20,))
    plt.figure(figsize=(10, 6))
    for i in range(20):
        plt.subplot(4, 5, i+1)
        plt.plot(waveforms[selected_idx[i]].cpu().numpy())
        plt.title(f"Waveform {selected_idx[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    