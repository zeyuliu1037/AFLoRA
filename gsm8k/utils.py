import torch
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any
import pickle
import wandb
import time
import warnings

from transformers import (
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerControl, TrainerState
# from peft.tuners import InRankEfficient


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_best_metric = {
    "rte": "eval_accuracy",
    "mrpc": "eval_f1", 
    "cola": "eval_matthews_correlation", 
    "stsb": "eval_pearson", 
    "sst2": "eval_accuracy", 
    "qnli": "eval_accuracy",
    "mnli": "eval_accuracy",
    "mnli-m": "eval_accuracy",
    "mnli-mm": "eval_accuracy",
    "qqp": "eval_accuracy",
}

def whether_freeze_para(n):
    if 'lora_A' in n or 'lora_B' in n:
        return True
    return False

def get_parameters_count(model, requires_grad=False, finetune_classifier=0):
        total_params = 0
        unique_tensors = set()
        total_lora_AB = 0
        unfreezed_lora_AB = 0
        for name, module in model.named_modules():
            for attr_str in dir(module):
                if attr_str == "trainer":  # Skip the trainer attribute
                    continue
                target_attr = getattr(module, attr_str)
                if type(target_attr) in (torch.Tensor, torch.nn.Parameter):
                    if (
                        id(target_attr) not in unique_tensors
                    ):  # Check if the tensor was already counted
                        if (
                            "classifier" not in name
                            or finetune_classifier == 1
                        ) and (not requires_grad or target_attr.requires_grad):
                            # print(name, attr_str, target_attr.shape)
                            num_params = torch.numel(target_attr)
                            total_params += num_params
                            if whether_freeze_para(name):
                                total_lora_AB += num_params
                                if target_attr.requires_grad:
                                    unfreezed_lora_AB += num_params
                        unique_tensors.add(
                            id(target_attr)
                        )  # Add the tensor id to the set of counted tensors
        return total_params, total_lora_AB, unfreezed_lora_AB

class GradientLogger:
    def __init__(self, model):
        self.model = model
        self.last_step = -1

    def log(
        self,
        state: TrainerState,
    ):
        if self.last_step >= state.global_step or state.global_step % 10 != 0:
            return
        total_norm = 0.0
        total_norm_lora = 0.0
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                if "lora" in n:
                    total_norm_lora += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        total_norm_lora = total_norm_lora ** (1.0 / 2)
        wandb.log({"gradient_norm": total_norm})
        wandb.log({"gradient_norm_lora": total_norm_lora})
        self.last_step = state.global_step

class WeightLogCallback(TrainerCallback):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_step = -1

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.last_step >= state.global_step or state.global_step % 10 != 0:
            return
        total_norm = 0.0
        lora_d = []
        lora_da = []
        lora_db = []
        lora_gate = []
        for n, p in self.model.named_parameters():
            if ".lora_d." in n:
                lora_d.extend(p.detach().flatten().cpu().numpy().tolist())
            elif ".lora_da." in n:
                lora_da.extend(p.detach().flatten().cpu().numpy().tolist())
            elif ".lora_db." in n:
                lora_db.extend(p.detach().flatten().cpu().numpy().tolist())
            elif ".gate." in n:
                lora_gate.extend(p.detach().flatten().cpu().numpy().tolist())
        if torch.distributed.get_rank() == 0:
            lora_d_abs = [abs(x) for x in lora_d]
            lora_da_abs = [abs(x) for x in lora_da]
            lora_db_abs = [abs(x) for x in lora_db]
            lora_gate_abs = [abs(x) for x in lora_gate]
            to_log = {}
            if len(lora_d) > 0:
                to_log["lora_d"] = wandb.Histogram(lora_d)
                to_log["lora_d_abs"] = wandb.Histogram(lora_d_abs)
                to_log["lora_d_min"] = min(lora_d)
                to_log["lora_d_max"] = max(lora_d)
                to_log["lora_d_avg"] = sum(lora_d) / len(lora_d)
                to_log["lora_d_abs_min"] = min(lora_d_abs)
                to_log["lora_d_abs_max"] = max(lora_d_abs)
                to_log["lora_d_abs_avg"] = sum(lora_d_abs) / len(lora_d)
            if len(lora_da) > 0:
                to_log["lora_da"] = wandb.Histogram(lora_da)
                to_log["lora_da_abs"] = wandb.Histogram(lora_da_abs)
                to_log["lora_da_min"] = min(lora_da)
                to_log["lora_da_max"] = max(lora_da)
                to_log["lora_da_avg"] = sum(lora_da) / len(lora_da)
                to_log["lora_da_abs_min"] = min(lora_da_abs)
                to_log["lora_da_abs_max"] = max(lora_da_abs)
                to_log["lora_da_abs_avg"] = sum(lora_da_abs) / len(lora_da)
            if len(lora_db) > 0:
                to_log["lora_db"] = wandb.Histogram(lora_db)
                to_log["lora_db_abs"] = wandb.Histogram(lora_db_abs)
                to_log["lora_db_min"] = min(lora_db)
                to_log["lora_db_max"] = max(lora_db)
                to_log["lora_db_avg"] = sum(lora_db) / len(lora_db)
                to_log["lora_db_abs_min"] = min(lora_db_abs)
                to_log["lora_db_abs_max"] = max(lora_db_abs)
                to_log["lora_db_abs_avg"] = sum(lora_db_abs) / len(lora_db)
            if len(lora_gate) > 0:
                to_log["lora_gate"] = wandb.Histogram(lora_gate)
                to_log["lora_gate_abs"] = wandb.Histogram(lora_gate_abs)
                to_log["lora_gate_min"] = min(lora_gate)
                to_log["lora_gate_max"] = max(lora_gate)
                to_log["lora_gate_avg"] = sum(lora_gate) / len(lora_gate)
                to_log["lora_gate_abs_min"] = min(lora_gate_abs)
                to_log["lora_gate_abs_max"] = max(lora_gate_abs)
                to_log["lora_gate_abs_avg"] = sum(lora_gate_abs) / len(lora_gate)
            wandb.log(to_log)
        self.last_step = state.global_step


class AdjustTrainableParametersCallback(TrainerCallback):
    def __init__(self, model, freeze_after_epoch=10):
        self.model = model
        self.freeze_after_epoch = freeze_after_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        params_trainable, total_lora_AB, unfreezed_lora_AB = get_parameters_count(self.model, requires_grad=True)
        wandb.log({"Trainable_params_epoch": params_trainable, "unfreezed_lora_AB_epoch": unfreezed_lora_AB})
        if state.epoch >= self.freeze_after_epoch:
            for name, param in self.model.named_parameters():
                if "lora_A" in name or "lora_B" in name:  
                    param.requires_grad = False

class Pruner(object):
    def __init__(self, config=None):
        self.config = config
        self.pruner_name = config['pruner_name']
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.first_time = True
        self.stop_pruning = False
        self.avg_train_params = 0
        self.last_step = 0
        self.purning_dict = {}
        self.purning_index = 0
    def schedule_threshold_comb(self, step: int):
        # Schedule the remaining ratio
        total_step = self.total_step
        initial_threshold = self.config['initial_threshold']
        final_threshold = self.config['final_threshold']
        initial_warmup = self.config['initial_warmup']
        final_warmup = self.config['final_warmup']
        warmup_steps = total_step
        mask_ind = False
        if step <= initial_warmup * warmup_steps:
            threshold = initial_threshold
            mask_ind = False
        elif step > (total_step - final_warmup * warmup_steps):
            threshold = final_threshold
            mask_ind = True
        else:
            spars_warmup_steps = initial_warmup * warmup_steps
            spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps
            # mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
            # threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 2)
            mul_coeff = (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
            threshold = 1 - (mul_coeff ** 2)
            mask_ind = True if step % self.config['deltaT'] == 0 else False
            # mask_ind = True
        return threshold, mask_ind
    
    def update_ipt_with_local_window(self, model, global_step):
        # Calculate the sensitivity and uncertainty
        for n, p in model.named_parameters():
            if p.requires_grad and whether_freeze_para(n):
                if n not in self.exp_avg_ipt:
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.ipt[n] = torch.zeros_like(p)
                    if self.config['beta2'] > 0 and self.config['beta2'] != 1:
                        self.exp_avg_unc[n] = torch.zeros_like(p)
                if self.pruner_name == 'Magnitude':
                    # Calculate the score of magnitude pruning
                    self.ipt[n] = p.abs().detach()
                elif self.pruner_name == 'PLATON':
                    local_step = global_step % self.config['deltaT']
                    update_step = global_step // self.config['deltaT']
                    if local_step == 0:
                        self.exp_avg_ipt[n] = self.config['beta1'] * self.exp_avg_ipt[n] + (1 - self.config['beta1']) * self.ipt[n]
                        if 0 < self.config['beta2'] < 1:
                            self.exp_avg_unc[n] = self.config['beta2'] * self.exp_avg_unc[n] + \
                                                  (1 - self.config['beta2']) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                        elif self.config['beta2'] == 2.:
                            self.exp_avg_unc[n] = (update_step * self.exp_avg_unc[n] +
                                                   (self.ipt[n] - self.exp_avg_ipt[n]) ** 2) / (update_step + 1)
                        if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                            warnings.warn(f"Inf or NaN in gradient of {n}")
                            self.ipt[n] = 0.0
                        else:
                            self.ipt[n] = (torch.sign(p) * p.grad).abs().detach()
                    else:
                        if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                            warnings.warn(f"Inf or NaN in gradient of {n}")
                        else:
                            self.ipt[n] = (self.ipt[n] * local_step + (torch.sign(p) * p.grad).abs().detach()) / (local_step + 1)
                else:
                    raise ValueError("Incorrect Pruner Name.")
    def share_ipt_AB(self, is_dict):
        updated_dict = {}
        sorted_keys = sorted(is_dict.keys())
        assert len(sorted_keys) % 2 == 0
        for i in range(0, len(sorted_keys), 2):
            lora_A = torch.mean(is_dict[sorted_keys[i]])
            lora_B = torch.mean(is_dict[sorted_keys[i + 1]])
            avg = (lora_A+lora_B) / 2
            updated_dict[sorted_keys[i]] = avg
            updated_dict[sorted_keys[i + 1]] = avg
        return updated_dict
                
    def mask_with_threshold(self, model, threshold, global_step):
        # Calculate the final importance score
        is_dict = {}
        sum_exp_avg_ipt = 0
        sum_exp_avg_unc = 0
        sum_is_dict = 0
        for n, p in model.named_parameters():
            if p.requires_grad and whether_freeze_para(n):
                if self.pruner_name == 'Magnitude':
                    is_dict[n] = self.ipt[n]
                elif self.pruner_name == 'PLATON':
                    if 0 < self.config['beta2'] < 1:
                        is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
                    elif self.config['beta2'] == 1.:
                        is_dict[n] = self.exp_avg_ipt[n]
                    elif self.config['beta2'] == 2.:
                        is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc.sqrt()
                    else:
                        # Handling the unaccepted beta2 to default setting
                        is_dict[n] = self.exp_avg_ipt[n] * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                else:
                    raise ValueError("Incorrect Pruner Name.")
                sum_exp_avg_ipt += torch.mean(self.exp_avg_ipt[n])
                sum_exp_avg_unc += torch.mean(self.exp_avg_unc[n])
                sum_is_dict += torch.mean(is_dict[n])
        wandb.log({"sum_exp_avg_ipt": sum_exp_avg_ipt})
        wandb.log({"sum_exp_avg_unc": sum_exp_avg_unc})
        wandb.log({"sum_is_dict": sum_is_dict})

        # Calculate the mask threshold
        # updated_share_dict = self.share_ipt_AB(is_dict)
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
        mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0] * (1 - threshold)))[0].item()
        # Mask weights whose importance lower than threshold
        for n, p in model.named_parameters():
            if p.requires_grad and whether_freeze_para(n) and torch.mean(is_dict[n]) < mask_threshold:
            # if p.requires_grad and whether_freeze_para(n) and updated_share_dict[n] < mask_threshold:
                p.requires_grad = False
                p.grad = None
                self.purning_index = global_step
                if self.purning_index in self.purning_dict:
                    self.purning_dict[self.purning_index].append(n)
                else:
                    self.purning_dict[self.purning_index] = [n]
                # pass
        # self.purning_index += 1
        return mask_threshold
                
    def pruning(self, model, global_step):
        if self.stop_pruning:
            return None, None
        # Update importance score after optimizer stepping
        self.update_ipt_with_local_window(model, global_step)
        # Get the remaining ratio
        threshold, mask_ind = self.schedule_threshold_comb(global_step)
        wandb.log({"threshold": threshold})
        if mask_ind:
            params_trainable, total_lora_AB, unfreezed_lora_AB = get_parameters_count(model, requires_grad=True)
            wandb.log({"trainable_params": params_trainable})
            wandb.log({"unfreezed_lora_AB": unfreezed_lora_AB})
            if unfreezed_lora_AB == 0:
                self.avg_train_params += params_trainable * (self.total_step - self.last_step)
                wandb.log({"avg_train_params": self.avg_train_params/self.total_step})
                wandb.config.update({"total_lora_AB": total_lora_AB})
                self.stop_pruning = True
                return None, None
            else:
                self.avg_train_params += params_trainable * (global_step - self.last_step)
                self.last_step = global_step
            
            mask_threshold = self.mask_with_threshold(model, threshold, global_step)            
            
        else:
            mask_threshold = None
        return threshold, mask_threshold
    
