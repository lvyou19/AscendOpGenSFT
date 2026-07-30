"""Auto-generated benchmark file for ReverseSequence.

Format mirrors /home/l00899543/Ascend_evaluation/AscendOpGenAgent/benchmarks/NPUKernelBench/level1
- get_input_groups(): list of [args...] per case
- get_init_inputs():  list of [init_args...] per case (same length, [] when none)
The Model class is the source's torch reference (NPU-only imports stripped).
prepare_inputs logic is embedded so each case is generated with the same
semantics as the original validation harness; the JSON file holds one CSV row
per line as the parameter dict.
"""

import sys as _sys
import types as _types

# Stub NPU / framework deps so import-time `import torch_npu` etc. succeeds
# even when this file is run on a host without those libraries installed.
for _n in (
    "torch_npu", "torch_npu.contrib", "torch_npu.contrib.transfer_to_npu",
    "kernel_gen_ops", "framework", "framework.utils", "framework.kernel_gen_config",
):
    if _n not in _sys.modules:
        _m = _types.ModuleType(_n); _m.__path__ = []
        _sys.modules[_n] = _m

import torch
import torch.nn as nn
import json
import os

# Patch torch.randn/rand to tolerate integer dtypes on CPU (some upstream
# prepare_inputs.py call torch.randn(shape, dtype=torch.int*) which only
# works on NPU).
_INT_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool}
_orig_randn, _orig_rand = torch.randn, torch.rand


def _safe_make(_orig, _lo, _hi, *args, **kwargs):
    dtype = kwargs.get("dtype")
    if dtype in _INT_DTYPES:
        size = args
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            size = tuple(args[0])
        new_kwargs = dict(kwargs)
        if dtype in (torch.uint8, torch.bool):
            _lo, _hi = 0, (2 if dtype == torch.bool else 100)
        return torch.randint(_lo, _hi, tuple(int(x) for x in size), **new_kwargs)
    return _orig(*args, **kwargs)


torch.randn = lambda *a, **kw: _safe_make(_orig_randn, -100, 100, *a, **kw)
torch.rand = lambda *a, **kw: _safe_make(_orig_rand, 0, 100, *a, **kw)


# ---- Model (cleaned from source module.py) ----
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, seq_lengths, seq_dim=1, batch_dim=0):
        input_shape = x.shape
        output = torch.zeros_like(x)
        batch_size = input_shape[batch_dim]
        for i in range(batch_size):
            batch_selector = [slice(None)] * len(input_shape)
            batch_selector[batch_dim] = i
            batch_selector = tuple(batch_selector)
            seq_len = seq_lengths[i].item() if seq_lengths.ndim > 0 else seq_lengths
            reversed_indices = torch.arange(seq_len - 1, -1, -1, device=x.device)
            seq_indices = torch.arange(seq_len, device=x.device)
            selector = list(batch_selector)
            selector[seq_dim] = seq_indices
            selector = tuple(selector)
            reversed_selector = list(batch_selector)
            reversed_selector[seq_dim] = reversed_indices
            reversed_selector = tuple(reversed_selector)
            output[selector] = x[reversed_selector]
            if seq_len < input_shape[seq_dim]:
                remaining_selector = list(batch_selector)
                remaining_selector[seq_dim] = slice(seq_len, None)
                remaining_selector = tuple(remaining_selector)
                output[remaining_selector] = x[remaining_selector]
        return output

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np

def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成 ReverseSequence 算子的输入张量。

    Args:
        param (dict): 参数配置，如输入形状、序列长度、维度等
        device (torch.device): 输入张量所在设备

    Returns:
        tuple: 包含输入张量 (x, seq_lengths, seq_dim, batch_dim)
    """
    shape = eval(param.get('input_shape', '[3, 5, 7]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    seq_dim = param.get('seq_dim', 1)
    batch_dim = param.get('batch_dim', 0)
    max_len = shape[seq_dim]
    batch_size = shape[batch_dim]
    seq_lengths_np = np.random.randint(1, max_len + 1, size=batch_size)
    seq_lengths = torch.tensor(seq_lengths_np, dtype=torch.int64, device=device)
    x = torch.randn(shape, device=device, dtype=dtype)
    return (x, seq_lengths, seq_dim, batch_dim)

def get_init_inputs_per_case(param, device=None):
    """
    reverse_sequence 没有模型初始化参数，返回空列表。

    Args:
        param (dict): 参数配置

    Returns:
        list: 空列表
    """
    return []


def _coerce(v):
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s == "":
        return s
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lstrip("-+").isdigit():
        try:
            return int(s)
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        pass
    return s


def _load_cases():
    json_path = os.path.join(os.path.dirname(__file__), "ReverseSequence.json")
    with open(json_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _try_get_inputs(case):
    coerced = {k: _coerce(v) for k, v in case.items()}
    last_err = None
    for param in (coerced, case):
        try:
            return get_inputs(param, device=None), param
        except Exception as e:
            last_err = e
    raise last_err


def get_input_groups():
    groups = []
    for case in _load_cases():
        args, _ = _try_get_inputs(case)
        groups.append(list(args) if isinstance(args, (list, tuple)) else [args])
    return groups


def get_init_inputs():
    out = []
    for case in _load_cases():
        coerced = {k: _coerce(v) for k, v in case.items()}
        try:
            init = get_init_inputs_per_case(coerced, device=None)
        except Exception:
            try:
                init = get_init_inputs_per_case(case, device=None)
            except Exception:
                init = []
        out.append(list(init) if isinstance(init, (list, tuple)) else [init])
    return out
