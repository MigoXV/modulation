import torch
import os
import wandb

from pathlib import Path
from omegaconf import OmegaConf
from typing import Optional, List
from torch.utils.data import DataLoader
from textbrewer import GeneralDistiller, TrainingConfig, BasicTrainer


class Trainer:

    def __init__(
        self,
        config,
        model,
        adaptor,
        scheduler_class,
        scheduler_args,
        validator,
        optimizer = None,
    ):

        self.config = config
        self.model = model
        self.adaptor = adaptor
        self.device = config.device
        self.validator = validator

        if optimizer is None:
            torch.optim.AdamW(self.student_model.parameters(), lr=config.learning_rate)
        else:
            self.optimizer = optimizer

        self.scheduler_class = scheduler_class
        self.scheduler_args = scheduler_args

        self.train_config = TrainingConfig(
            device=self.device,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            ckpt_epoch_frequency=config.ckpt_epoch_frequency,
            log_dir=config.log_dir,
            output_dir=config.output_dir,
            device_ids=config.device_ids,
        )  # 设置训练配置，指定设备

        self.trainer = BasicTrainer(
            train_config=self.train_config,
            model=self.model,
            adaptor=self.adaptor,
        )

    def train(
        self,
        train_dataloader,
        num_epochs,
        max_grad_norm=-1.0,
        batch_postprocessor=None,
    ):

        with self.trainer:
            # 蒸馏模型
            self.trainer.train(
                optimizer=self.optimizer,
                dataloader=train_dataloader,
                num_epochs=num_epochs,
                scheduler_class=self.scheduler_class,
                scheduler_args=self.scheduler_args,
                max_grad_norm=max_grad_norm,
                callback=self.validator.callback,
                batch_postprocessor=batch_postprocessor,
            )
