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
from datasets import load_dataset, load_metric


import transformers
from transformers import (
    Trainer,
)

from peft.tuners.lora import LoraLayer

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import check_min_version, is_sagemaker_mp_enabled

import torch.nn as nn
from utils import Pruner, GradientLogger

ipt_base_config = {
    "initial_threshold": 1,
    "final_threshold": 0.,
    "initial_warmup": 1,
    "final_warmup": 3,
    "warmup_steps": 120,
    "beta1": 0.85, # 0.15/0.5/0.85
    "beta2": 0.95,
    "deltaT": 10,
    "pruner_name": 'PLATON',
}
ipt_all_tasks_config = {
    'commonsense': {**ipt_base_config,
        "initial_warmup": 0.3, 
        "final_warmup": 0.3,
        "deltaT": 10, 
    }
}

class CustomTrainer(Trainer):
    def __init__(
        self,
        gradientLogger: GradientLogger,
        classifier_wd,
        classifier_lr,
        logger,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.gradientLogger = gradientLogger
        self.classifier_wd = classifier_wd
        self.classifier_lr = classifier_lr

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

class CustomTrainerWFreeze(Trainer):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        ipt_config = ipt_all_tasks_config['commonsense']
        ipt_config['num_train_epochs'] = kwargs['args'].num_train_epochs
        self.pruner = Pruner(ipt_config)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        result = super().training_step(model, inputs)
        self.pruner.total_step = self.state.max_steps
        with torch.no_grad():
            threshold, mask_threshold = self.pruner.pruning(self.model, self.state.global_step)
        return result
