from .dataset import wave_dataset
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split


class DataFeeder:

    def __init__(self, config):
        self.config = config

        # 从指定的路径加载数据和标签
        self.data, self.labels = torch.load(config.data_path)

        # 分割数据集
        train_data, val_data, train_labels, val_labels = train_test_split(
            self.data, self.labels, test_size=config.test_size
        )

        # 创建数据集
        train_dataset = wave_dataset(train_data, train_labels, config)
        val_dataset = wave_dataset(val_data, val_labels, config)
        
        # 创建数据加载器
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=config.per_device_train_batch_size, shuffle=True
        )

        self.val_dataloader = DataLoader(
            val_dataset, batch_size=config.per_device_eval_batch_size
        )
