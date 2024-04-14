import wandb

from pathlib import Path
from omegaconf import OmegaConf
from typing import Optional, List

from trainer.trainer import Trainer
from trainer.validator import Validator
from data_feeder.data_feeder import DataFeeder
from model.wave_transformer import WaveTransformerModel

import torch

def get_config(config_path: Path, config_set: Optional[List[str]] = None) -> OmegaConf:

    if config_set is None:
        config_set = []

    # config = OmegaConf.structured(TrainConfig)
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(config_set))
    return config


def train_from_scatch(
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

    # 损失函数：交叉熵
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # 适配器
    def adaptor(batch, model_outputs):
        return {"losses": loss_fn(model_outputs, batch[1])}
    
    # 创建验证器
    validator = Validator(
        val_dataloader=datafeeder.val_dataloader,
        loss_fn=loss_fn,
        device=device,
    )

    # 初始化蒸馏器
    trainer = Trainer(
        config=config,
        model=model,
        adaptor=adaptor,
        scheduler_class=None,
        scheduler_args=None,
        validator=validator,
    )

    if config.report_to == "wandb":
        with wandb.init(project=config.wandb_project, config=OmegaConf.to_container(config, resolve=True)):
            trainer.train(
                train_dataloader=datafeeder.train_dataloader,
                num_epochs=config.num_epochs,
                max_grad_norm=config.max_grad_norm,
            )
    else:
        trainer.train(
            train_dataloader=datafeeder.train_dataloader,
            num_epochs=config.num_epochs,
            max_grad_norm=config.max_grad_norm,
        )


if __name__ == "__main__":
    train_from_scatch("config/train_config_test.yaml")