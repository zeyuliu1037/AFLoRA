from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any

from utils import task_to_keys

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in "
                    + ",".join(task_to_keys.keys())
                )
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."




@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    job_id: Optional[str] = field(default=None)
    wandb_run_group: Optional[str] = field(
        default=None,
    )
    mode: Optional[str] = field(
        default='lora',
    )
    submode: Optional[str] = field(
        default=None,
    )
    wandb_run_name: Optional[str] = field(
        default=None,
    )
    apply_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply LoRA or not."},
    )
    d_init_type: Optional[int] = field(
        default=0,
    )
    identity_init: Optional[int] = field(
        default=0,
    )
    finetune_classifier: Optional[int] = field(
        default=0,
    )
    shared_uv: Optional[int] = field(
        default=0,
    )
    wandb_offline: Optional[int] = field(
        default=0,
    )
    norm_penalty: Optional[int] = field(
        default=0,
    )
    norm_alpha: Optional[float] = field(
        default=0.001,
    )
    use_float64: Optional[int] = field(
        default=0,
    )
    optimized_order: Optional[int] = field(
        default=0,
    )
    order: Optional[int] = field(
        default=0,
    )
    shared_d: Optional[int] = field(
        default=0,
    )
    classifier_lr: Optional[float] = field(
        default=4e-4,
    )
    classifier_wd: Optional[float] = field(
        default=0.1,
    )
    trainable_uv: Optional[int] = field(
        default=0,
    )
    nonlin: Optional[int] = field(
        default=0,
    )
    init_type: Optional[int] = field(
        default=1,
    )
    d_init_type: Optional[int] = field(
        default=92,
    )
    custom_scaling: Optional[int] = field(
        default=0,
    )
    lora_alpha: Optional[float] = field(
        default=1.0,
        metadata={"help": "LoRA alpha"},
    )
    shared_dim: Optional[int] = field(
        default=None,
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA r"},
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of LoRA parameters."},
    )
    apply_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply adapter or not."},
    )
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of adapter parameters."},
    )
    adapter_type: Optional[str] = field(
        default="houlsby",
        metadata={"help": "houlsby or pfeiffer"},
    )
    adapter_size: Optional[int] = field(
        default=64,
        metadata={"help": "8, 16, 32, 64"},
    )
    apply_bitfit: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply bitfit or not."},
    )
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )
    masking_prob: Optional[float] = field(
        default=0.0,
        metadata={"help": "Token Masking Probability"},
    )
    freeze_by_epoch: Optional[float] = field(
        default=0,
        metadata={"help": "after which epoch to freeze the model."},
    )
    freeze_by_ipt: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze by ipt or not."},
    )
    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "LoRA dropout"},
    )