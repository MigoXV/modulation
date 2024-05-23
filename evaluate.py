import torch

from train import get_config
from pathlib import Path
from omegaconf import OmegaConf
from typing import Optional, List

from trainer.validator import Validator
from data_feeder.data_feeder import DataFeeder
from model.wave_transformer import WaveTransformerModel

def evaluate(
    config_path: Path,
    config_set: Optional[List[str]] = None,
):
    # 加载配置
    config = get_config(config_path, config_set)

    device = config.device

    # 创建数据提供器
    datafeeder = DataFeeder(config)

    # 加载模型
    model = WaveTransformerModel(config).to(device)
    model.load_state_dict(torch.load(config.checkpoint_path))

    # 损失函数：交叉熵
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # 创建验证器
    validator = Validator(
        config=config,
        val_dataloader=datafeeder.val_dataloader,
        loss_fn=loss_fn,
        device=device,
    )

    validator.callback(model, 0)


if __name__ == "__main__":
    evaluate("config/evaluate_base.yml")