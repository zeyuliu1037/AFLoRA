# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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
import os
import math
import re
import warnings
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D



class LoraLayer:
    def __init__(self, in_features: int, out_features: int, fan_in_fan_out: bool, custom, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs
        self.fan_in_fan_out = fan_in_fan_out
        self.custom = custom
        self.current_epoch = 0
        if self.custom["mode"] == "elora":
            self.lora_d = nn.ParameterDict({})
            self.lora_db = nn.ParameterDict({})
        else:
            self.min_dim = min(self.in_features, self.out_features)
        if self.custom["mode"] == "elora" and self.custom["shared_dim"] is not None:
            if self.in_features == self.out_features:
                self.share_layer_type = 0
            elif self.in_features < self.out_features:
                self.share_layer_type = 1
            else:
                self.share_layer_type = 2

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # Actual trainable parameters
        if r > 0:
            if self.custom["shared_dim"] is None:
                _lora_A = nn.Linear(
                    self.in_features,
                    r,
                    bias=False,
                    dtype=torch.float64 if self.custom["use_float64"] else torch.float32,
                )
                _lora_B = nn.Linear(
                    r,
                    self.out_features,
                    bias=False,
                    dtype=torch.float64 if self.custom["use_float64"] else torch.float32,
                )
            else:
                if self.share_layer_type == 0 and self.custom["shared_matrices"] is None:
                    _lora_A = nn.Linear(
                        self.custom["shared_dim"]["A"],
                        r,
                        bias=False,
                        dtype=torch.float64 if self.custom["use_float64"] else torch.float32,
                    )
                    _lora_B = nn.Linear(
                        r,
                        self.custom["shared_dim"]["B"],
                        bias=False,
                        dtype=torch.float64 if self.custom["use_float64"] else torch.float32,
                    )
                    self.custom["shared_matrices"] = {"A0": _lora_A, "B0": _lora_B}
                elif self.share_layer_type == 1 and "A1" not in self.custom["shared_matrices"]:
                    _lora_A = nn.Linear(
                        self.in_features,
                        r,
                        bias=False,
                        dtype=torch.float64 if self.custom["use_float64"] else torch.float32,
                    )
                    _lora_B = nn.Linear(
                        r,
                        self.out_features,
                        bias=False,
                        dtype=torch.float64 if self.custom["use_float64"] else torch.float32,
                    )
                    self.custom["shared_matrices"].update({"A1": _lora_A, "B1": _lora_B})
                elif self.share_layer_type == 2 and "A2" not in self.custom["shared_matrices"]:
                    _lora_A = nn.Linear(
                        self.in_features,
                        r,
                        bias=False,
                        dtype=torch.float64 if self.custom["use_float64"] else torch.float32,
                    )
                    _lora_B = nn.Linear(
                        r,
                        self.out_features,
                        bias=False,
                        dtype=torch.float64 if self.custom["use_float64"] else torch.float32,
                    )
                    self.custom["shared_matrices"].update({"A2": _lora_A, "B2": _lora_B})
                else:
                    _lora_A = self.custom["shared_matrices"][f"A{self.share_layer_type}"]
                    _lora_B = self.custom["shared_matrices"][f"B{self.share_layer_type}"]
                if "dynamic_uv" in self.custom and self.custom["dynamic_uv"]:
                    _lora_A = nn.Linear(
                        self.in_features,
                        r,
                        bias=False,
                        dtype=torch.float64 if self.custom["use_float64"] else torch.float32,
                    )
                    _lora_B = nn.Linear(
                        r,
                        self.out_features,
                        bias=False,
                        dtype=torch.float64 if self.custom["use_float64"] else torch.float32,
                    )
                    _lora_A.weight.data = self.custom["shared_matrices"]["A"].weight.data[:r, : self.in_features]
                    _lora_B.weight.data = self.custom["shared_matrices"]["B"].weight.data[: self.out_features, :r]
            self.lora_A.update(nn.ModuleDict({adapter_name: _lora_A}))
            self.lora_B.update(nn.ModuleDict({adapter_name: _lora_B}))
            if self.custom["mode"] != "lora":
                if not self.custom["trainable_uv"] or (self.custom["mode"]=="elora" and self.in_features == self.out_features):
                # if not self.custom["trainable_uv"] or (self.custom["mode"]=="elora" and self.in_features != self.out_features):
                # if not self.custom["trainable_uv"]:
                    self.lora_A.requires_grad_(False)
                    self.lora_B.requires_grad_(False)
                if self.custom["d_init_type"] == 0:

                    def d_init_fn(dim, component="d"):
                        return (
                            torch.randn(dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32)
                            * self.custom["d_init"]
                        )

                elif self.custom["d_init_type"] == 1:

                    def d_init_fn(dim, component="d"):
                        return torch.ones(dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32)

                elif self.custom["d_init_type"] == 2:

                    def d_init_fn(dim, component="d"):
                        return (
                            torch.randn(dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32)
                            / 10.0
                            + 1
                        )

                elif self.custom["d_init_type"] == 3:

                    def d_init_fn(dim, component="d"):
                        return (
                            torch.randn(dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32)
                            / 100.0
                            + 1
                        )

                elif self.custom["d_init_type"] == 4:

                    def d_init_fn(dim, component="d"):
                        if isinstance(dim, int):
                            dim = (dim,)
                        return torch.full(
                            dim, 1e-7, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                        )

                elif self.custom["d_init_type"] == 91:

                    def d_init_fn(dim, component="d"):
                        if component == "d":
                            return torch.zeros(
                                dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )
                        else:
                            return torch.ones(
                                dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )

                elif self.custom["d_init_type"] == 92:

                    def d_init_fn(dim, component="d"):
                        if component == "b":
                            return torch.zeros(
                                dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )
                        else:
                            return torch.ones(
                                dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )

                elif self.custom["d_init_type"] == 93:

                    def d_init_fn(dim, component="d"):
                        if isinstance(dim, int):
                            dim = (dim,)
                        if component == "b":
                            return torch.ones(
                                dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )
                        else:
                            return torch.full(
                                dim, 1e-7, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )

                elif self.custom["d_init_type"] == 94:

                    def d_init_fn(dim, component="d"):
                        if isinstance(dim, int):
                            dim = (dim,)
                        if component == "b":
                            return torch.zeros(
                                dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )
                        else:
                            return torch.full(
                                dim, 0.1, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )

                elif self.custom["d_init_type"] == 95:

                    def d_init_fn(dim, component="d"):
                        if isinstance(dim, int):
                            dim = (dim,)
                        if component == "d":
                            return torch.full(
                                dim, 0.001, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )
                        else:
                            return torch.zeros(
                                dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )

                elif self.custom["d_init_type"] == 96:

                    def d_init_fn(dim, component="d"):
                        if isinstance(dim, int):
                            dim = (dim,)
                        if component == "d":
                            return torch.full(
                                dim, 0.5, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )
                        else:
                            return torch.zeros(
                                dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )

                elif self.custom["d_init_type"] == 97:

                    def d_init_fn(dim, component="d"):
                        if isinstance(dim, int):
                            dim = (dim,)
                        if component == "d":
                            return torch.full(
                                dim, 0.01, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )
                        else:
                            return torch.zeros(
                                dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )

                elif self.custom["d_init_type"] == 98:

                    def d_init_fn(dim, component="d"):
                        if isinstance(dim, int):
                            dim = (dim,)
                        if component == "b":
                            return torch.zeros(
                                dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )
                        else:
                            return torch.full(
                                dim, 1e-7, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )

                elif self.custom["d_init_type"] == 990:

                    def d_init_fn(dim, component="d"):
                        if isinstance(dim, int):
                            dim = (dim,)
                        if component == "d":
                            return torch.randn(
                                dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )
                        else:
                            return torch.zeros(
                                dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )

                elif self.custom["d_init_type"] == 991:

                    def d_init_fn(dim, component="d"):
                        if isinstance(dim, int):
                            dim = (dim,)
                        if component == "d":
                            _d = torch.zeros(dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32)
                            torch.nn.init.trunc_normal_(_d, a=-1.0, b=1.0)
                            return _d
                        else:
                            return torch.zeros(
                                dim, dtype=torch.float64 if self.custom["use_float64"] else torch.float32
                            )

                else:
                    raise NotImplementedError()

                if self.custom["shared_d"]:
                    if self.custom["shared_d_vector"] is None:
                        _lora_d = nn.Parameter(d_init_fn(r, "d"))
                        self.custom["shared_d_vector"] = _lora_d
                    else:
                        _lora_d = self.custom["shared_d_vector"]
                else:
                    _lora_d = nn.Parameter(d_init_fn(r, "d"))

                if self.custom["mode"] == "elora":
                    self.lora_db.update(
                        nn.ParameterDict({adapter_name: nn.Parameter(d_init_fn(self.out_features, "b"))})
                    )
                    self.lora_db.requires_grad_(True)
                    self.lora_d.update(nn.ParameterDict({adapter_name: _lora_d}))
                    self.lora_d.requires_grad_(True)
                elif self.custom["mode"] == "only_b":
                    self.lora_db.update(
                        nn.ParameterDict({adapter_name: nn.Parameter(d_init_fn(self.out_features, "b"))})
                    )
                    self.lora_db.requires_grad_(True)
                elif self.custom["mode"] == "only_d":
                    self.lora_d.update(nn.ParameterDict({adapter_name: _lora_d}))
                    self.lora_d.requires_grad_(True)
            if self.custom["submode"] == "lora_half" or self.custom["submode"] == "lora_half_svd":
                self.lora_A.requires_grad_(False)
        else:
            raise NotImplementedError()
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            sqrt_a = self.custom["sqrt_a"]
            if self.custom["mode"] == "lora":
                if self.custom["submode"] == "lora_half_svd" or self.custom["submode"] == "lora_svd":
                    U, s, V = torch.svd(self.weight.T if self.fan_in_fan_out else self.weight)
                    U = torch.mm(U, torch.diag(s))
                    self.lora_A[adapter_name].weight.data = V.T[: self.r[adapter_name], :]
                    self.lora_B[adapter_name].weight.data = U[:, : self.r[adapter_name]]
                else:
                    nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(sqrt_a))
                    nn.init.zeros_(self.lora_B[adapter_name].weight)
                if self.custom["identity"]:
                    self._A = self.lora_A[adapter_name].weight.data.clone().to(get_device())
                    self._B = self.lora_B[adapter_name].weight.data.clone().to(get_device())
            else:
                if (
                    self.custom["mode"] == "only_d"
                    or self.custom["mode"] == "elora"
                    or self.custom["mode"] == "only_b"
                    or (self.hlora_flag == 1)
                ):
                    if self.custom["init_type"] == 0:
                        nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(sqrt_a))
                        nn.init.kaiming_uniform_(self.lora_B[adapter_name].weight, a=math.sqrt(sqrt_a))
                    elif self.custom["init_type"] == 1:
                        nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight)
                        nn.init.kaiming_uniform_(self.lora_B[adapter_name].weight)
                    elif self.custom["init_type"] == 2:
                        nn.init.xavier_uniform_(self.lora_A[adapter_name].weight)
                        nn.init.xavier_uniform_(self.lora_B[adapter_name].weight)
                    elif self.custom["init_type"] == 3:
                        nn.init.kaiming_normal_(self.lora_A[adapter_name].weight)
                        nn.init.kaiming_normal_(self.lora_B[adapter_name].weight)
                    elif self.custom["init_type"] == 4:
                        nn.init.xavier_normal_(self.lora_A[adapter_name].weight)
                        nn.init.xavier_normal_(self.lora_B[adapter_name].weight)
                    elif self.custom["init_type"] == 5:
                        nn.init.trunc_normal_(self.lora_A[adapter_name].weight, std=0.02, a=-1.0, b=1.0)
                        nn.init.trunc_normal_(self.lora_B[adapter_name].weight, std=0.02, a=-1.0, b=1.0)
                    elif self.custom["init_type"] == 6:
                        nn.init.uniform_(self.lora_A[adapter_name].weight, a=0.0, b=0.1)
                        nn.init.uniform_(self.lora_B[adapter_name].weight, a=0.0, b=0.1)
                    elif self.custom["init_type"] == 10:
                        nn.init.normal_(self.lora_A[adapter_name].weight)
                        nn.init.normal_(self.lora_B[adapter_name].weight)
                        self.lora_A[adapter_name].weight.data *= self.scaling[adapter_name]


                if self.custom["identity"]:
                    self._A = self.lora_A[adapter_name].weight.data.clone().to(get_device())
                    self._B = self.lora_B[adapter_name].weight.data.clone().to(get_device())
                    if self.custom["mode"] == "elora":
                        self._db = self.lora_db[adapter_name].data.clone().to(get_device())
                        self._d = self.lora_d[adapter_name].data.clone().to(get_device())
                    elif self.custom["mode"] == "only_b":
                        self._db = self.lora_db[adapter_name].data.clone().to(get_device())
                    elif self.custom["mode"] == "only_d":
                        self._d = self.lora_d[adapter_name].data.clone().to(get_device())

        if adapter_name in self.lora_embedding_A.keys():
            raise NotImplementedError()

