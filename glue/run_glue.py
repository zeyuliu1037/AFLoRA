#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

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
    TrainerCallback,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint, is_main_process
from peft import LoraConfig, get_peft_model, TaskType
from transformers.utils import check_min_version
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0")

from args import ModelArguments, DataTrainingArguments
from utils import GradientLogger, task_to_keys, task_to_best_metric, ipt_all_tasks_config, \
    AdjustTrainableParametersCallback, WeightLogCallback, get_parameters_count
from trainer import CustomTrainer, CustomTrainerWFreeze

try:
    _dirs = ["weights", "output", "runs"]
    for _dir in _dirs:
        if not os.path.exists(_dir):
            os.makedirs(_dir, exist_ok=True)
except:
    pass

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.metric_for_best_model = task_to_best_metric[data_args.task_name]
    job_id = os.environ.get("SLURM_JOB_ID", "0")
    run_id = wandb.util.generate_id()

    is_first_rank = True
    try:
        print("rank", torch.distributed.get_rank())
        is_first_rank = torch.distributed.get_rank() == 0
    except:
        pass

    log_time = time.strftime("%Y%m%d%H%M%S")
    log_name = f"{model_args.mode}_r{model_args.lora_r}_lr_{training_args.learning_rate}_{log_time}"

    if is_first_rank:
        # use time as log name
        name = 'roberta' if 'roberta' in model_args.model_name_or_path else 'deberta'
        wandb.init(
            id=run_id,
            group=model_args.wandb_run_group,
            project=f"{name}-{data_args.task_name}",
            name=log_name
            if model_args.wandb_run_name is None
            else model_args.wandb_run_name,
            mode="online" if model_args.wandb_offline == 0 else "offline",
        )
        wandb.config.update(model_args.__dict__)

    # torch.use_deterministic_algorithms(training_args.use_deterministic_algorithms)
    # logger.info(
    #     "use_deterministic_algorithms: "
    #     + str(torch.are_deterministic_algorithms_enabled())
    # )

    # Detecting last checkpoint.
    training_args.output_dir = f"{training_args.output_dir}/{data_args.task_name}/{log_name}"
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    os.makedirs(training_args.output_dir, exist_ok=True)
    log_file_path = os.path.join(training_args.output_dir, 'my_logfile.log')
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {
            "train": data_args.train_file,
            "validation": data_args.validation_file,
        }

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError(
                    "Need either a GLUE task or a test file for `do_predict`."
                )

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in [
            "float32",
            "float64",
        ]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        cls_dropout=None,
        apply_lora=model_args.apply_lora,
        lora_alpha=model_args.lora_alpha,
        lora_r=model_args.lora_r,
        apply_adapter=model_args.apply_adapter,
        adapter_type=model_args.adapter_type,
        adapter_size=model_args.adapter_size,
        reg_loss_wgt=model_args.reg_loss_wgt,
        masking_prob=model_args.masking_prob,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(
            # -1181188
            f"real trainable params: {trainable_params} || trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    logger.info(model)

    target_modules = (
        ["query_proj", "value_proj", "key_proj", "output.dense", "intermediate.dense"]
        if "deberta" in model_args.model_name_or_path
        else ["query", "value", "key", "output.dense", "intermediate.dense"]
    )

    if model_args.finetune_classifier == 1:
        target_modules += ["classifier.dense", "classifier.out_proj"]

    lora_config_dict = {
        'r': model_args.lora_r,
        'lora_alpha': model_args.lora_alpha,
        'target_modules': target_modules,
        'lora_dropout': model_args.lora_dropout,
        'bias': 'none',
        'modules_to_save': [] if model_args.finetune_classifier == 1 else ["classifier"],
        'task_type': TaskType.SEQ_CLS,
    }
    lora_config_custom_dict = {
        "mode": model_args.mode,
        "submode": model_args.submode,
        "d_init": 1.0,
        "sqrt_a": 5.0,
        "identity": model_args.identity_init == 1,
        "init_type": model_args.init_type,
        "d_init_type": model_args.d_init_type,
        "custom_scaling": model_args.custom_scaling,
        "shared_dim": {"A": model_args.shared_dim, "B": model_args.shared_dim}
        if model_args.shared_uv == 1
        else None,
        "shared_matrices": None,
        "shared_d": model_args.shared_d == 1,
        "shared_d_vector": None,
        "trainable_uv": model_args.trainable_uv == 1,
        "nonlin": model_args.nonlin,
        "use_float64": model_args.use_float64 == 1,
        "norm_penalty": model_args.norm_penalty,
        "norm_alpha": model_args.norm_alpha,
    }
    config = LoraConfig(**lora_config_dict)
    config.custom = {**lora_config_custom_dict}
    model = get_peft_model(model, config)

    logger.info(lora_config_dict['target_modules'])

    if model_args.finetune_classifier == 1:
        for n, p in model.named_parameters():
            disable = [
                "dense.bias",
                "dense.weight",
                "out_proj.bias",
                "out_proj.weight",
                "dense.lora_A.",
                "dense.lora_B.",
                "out_proj.lora_A.",
                "out_proj.lora_B.",
            ]
            if any([d in n for d in disable]):
                p.requires_grad = False
                logger.info("disabled:", n)

    # model.model.classifier.modules_to_save = None
    print_trainable_parameters(model)

    params_trainable, total_lora_AB, unfreezed_lora_AB = get_parameters_count(model, requires_grad=True, finetune_classifier=model_args.finetune_classifier)
    params_total, _, _ = get_parameters_count(model, requires_grad=False, finetune_classifier=model_args.finetune_classifier)

    logger.info(f"Trainable parameters: {params_trainable}, total lora AB: {total_lora_AB}, trainable parameters : {params_trainable-total_lora_AB}")
    logger.info(f"Total number of parameters: {params_total}")
    if is_first_rank:
        wandb.config.update(
            {"params_trainable": params_trainable, "params_total": params_total}
        )
        wandb.log({"params_trainable": params_trainable, "params_total": params_total})

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in datasets["train"].column_names if name != "label"
        ]
        if (
            "sentence1" in non_label_column_names
            and "sentence2" in non_label_column_names
        ):
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [
                (label_to_id[l] if l != -1 else -1) for l in examples["label"]
            ]
        return result

    datasets = datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets[
            "validation_matched" if data_args.task_name == "mnli" else "validation"
        ]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if (
        training_args.do_predict
        or data_args.task_name is not None
        or data_args.test_file is not None
    ):
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets[
            "test_matched" if data_args.task_name == "mnli" else "test"
        ]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        try:
            preds = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            # print('check shape: p.shape: {}, preds.shape: {}'.format(p.predictions.shape, preds.shape))
            # print('labels: ', np.unique(p.label_ids))
            if data_args.task_name is not None:
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {
                    "accuracy": (preds == p.label_ids).astype(np.float32).mean().item()
                }
        except Exception as e:
            logger.info(e)
            return {}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # # todo: to delete
    # if model_args.wandb_run_group is None:
    #     training_args.evaluation_strategy = "steps"
    #     training_args.eval_steps = 100
    # training_args.save_strategy = "no"

    gradientLogger = GradientLogger(model=model)
    weightLogCallback = WeightLogCallback(model=model)
    
    tf_trainer = CustomTrainer
    callbacks=[weightLogCallback]
    ipt_config = None
    if model_args.freeze_by_ipt:
        ipt_config = ipt_all_tasks_config[data_args.task_name]
        logger.info("**ipt_config**")
        logger.info(ipt_config)
        tf_trainer = CustomTrainerWFreeze
    if model_args.freeze_by_epoch > 0:
        adjustTrainableParamsCallback = AdjustTrainableParametersCallback(model, freeze_after_epoch=model_args.freeze_by_epoch*training_args.num_train_epochs \
                                                                          if model_args.freeze_by_epoch < 1 else model_args.freeze_by_epoch)
        callbacks.append(adjustTrainableParamsCallback)

    # Initialize our Trainer
    trainer = tf_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        gradientLogger=gradientLogger,
        classifier_lr=model_args.classifier_lr,
        classifier_wd=model_args.classifier_wd,
        logger=logger,
        ipt_config=ipt_config,
    )

    trainer.create_optimizer()
    # logger.info(trainer.optimizer)

    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if (
                AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels
                == num_labels
            ):
                checkpoint = model_args.model_name_or_path

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        logger.info("*** Train result ***")
        logger.info(metrics)

        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        try:
            weights = {}
            for n, p in model.named_parameters():
                if ".lora_d." in n or ".lora_da." in n or ".lora_db." in n:
                    weights[n] = p.detach().cpu().numpy()
            if is_first_rank:
                with open(
                    os.path.join(
                        "weights",
                        f"{job_id}_{run_id}.pkl",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(weights, f)
        except:
            pass

        trainer.save_model(training_args.output_dir)  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        try:
            if is_first_rank:
                # path: "runs/[timestamp]-[run_id]"
                path = os.path.join(
                    "runs",
                    f"{time.strftime('%Y-%m-%d-%H-%M-%S')}-{run_id}",
                )
                os.makedirs(path, exist_ok=True)
                state_path = os.path.join(path, "trainer_state.json")
                args_path = os.path.join(path, "trainer_args.json")
                model_args_path = os.path.join(path, "model_args.json")
                trainer.state.save_to_json(state_path)
                with open(args_path, "w") as f:
                    json.dump(trainer.args.to_dict(), f)
                with open(model_args_path, "w") as f:
                    json.dump(model_args.__dict__, f)
        except Exception as e:
            logger.info("error", e)
    try:
        with open('my_dict.pkl', 'wb') as pickle_file:
            pickle.dump(trainer.pruner.purning_dict, pickle_file)
        print('******saved******')
    except:
        print('******not saved******')
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])
        
        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            logger.info(metrics)
            max_val_samples = (
                data_args.max_val_samples
                if data_args.max_val_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=test_dataset)
            logger.info(metrics)
            max_test_samples = (
                data_args.max_test_samples
                if data_args.max_test_samples is not None
                else len(test_dataset)
            )
            metrics["test_samples"] = min(max_test_samples, len(test_dataset))

            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

        # for test_dataset, task in zip(test_datasets, tasks):
        #     # Removing the `label` columns because it contains -1 and Trainer won't like that.
        #     # test_dataset.remove_columns_("label")
        #     test_dataset = test_dataset.remove_columns("label")
        #     predictions = trainer.predict(test_dataset=test_dataset).predictions
        #     predictions = (
        #         np.squeeze(predictions)
        #         if is_regression
        #         else np.argmax(predictions, axis=1)
        #     )

        #     output_test_file = os.path.join(
        #         training_args.output_dir, f"test_results_{task}.txt"
        #     )
        #     if trainer.is_world_process_zero():
        #         with open(output_test_file, "w") as writer:
        #             logger.info(f"***** Test results {task} *****")
        #             writer.write("index\tprediction\n")
        #             for index, item in enumerate(predictions):
        #                 if is_regression:
        #                     writer.write(f"{index}\t{item:3.3f}\n")
        #                 else:
        #                     item = label_list[item]
        #                     writer.write(f"{index}\t{item}\n")
    # # Finish the run
    # wandb.finish()
    # api = wandb.Api()
    # run = api.run(f"zeyuliu/{name}-{data_args.task_name}/{run_id}")
    # memory_metrics_before = run.history(keys=["memory before training_step"])
    # memory_metrics_after = run.history(keys=["memory after training_step"])
    # memory_metrics = run.history(keys=["memory after pruning"])
    # memory_metrics_before = memory_metrics["memory before training_step"].mean()
    # memory_metrics_after = memory_metrics["memory after training_step"].mean()
    # memory_metrics = memory_metrics["memory after pruning"].mean()
    # logger.info(f"Average memory across all steps before: {memory_metrics_before} MB")
    # logger.info(f"Average memory across all steps after: {memory_metrics_after} MB")
    # logger.info(f"Average memory across all steps pruning: {memory_metrics} MB")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
