import logging
import os
import random
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any
import pickle
import wandb
import time
import json
import numpy as np

import transformers
from transformers import Trainer

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
import torch.nn as nn
from utils import Pruner, GradientLogger


class CustomTrainer(Trainer):
    def __init__(
        self,
        gradientLogger: GradientLogger,
        classifier_wd,
        classifier_lr,
        logger,
        ipt_config,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.gradientLogger = gradientLogger
        self.classifier_wd = classifier_wd
        self.classifier_lr = classifier_lr
        self.ipt_config = ipt_config
        if ipt_config is not None:
            self.ipt_config['num_train_epochs'] = kwargs['args'].num_train_epochs

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        result = super().training_step(model, inputs)
        self.gradientLogger.log(self.state)
        return result

    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            non_norm_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            classifier_parameters = [
                name
                for name in non_norm_parameters
                if "classifier" in name and "bias" not in name
            ]
            classifier_bias_parameters = [
                name
                for name in non_norm_parameters
                if "classifier" in name and "bias" in name
            ]
            decay_parameters = [
                name
                for name in non_norm_parameters
                if "classifier" not in name and "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad and 'lora_S' not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n not in decay_parameters
                            and n not in classifier_parameters
                            and n not in classifier_bias_parameters
                            and p.requires_grad
                            and 'lora_S' in n
                        )
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in classifier_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.classifier_wd,
                    "lr": self.classifier_lr,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in classifier_bias_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.classifier_lr,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        self.logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        self.logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                self.logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

class CustomTrainerWFreeze(CustomTrainer):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pruner = Pruner(self.ipt_config)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        result = super().training_step(model, inputs)
        self.pruner.total_step = self.state.max_steps
        threshold, mask_threshold = self.pruner.update_and_pruning(model, self.state.global_step)
        self.gradientLogger.log(self.state)
        return result