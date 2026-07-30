"""Auto-generated benchmark file for HansDecode.

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
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional
from typing import Dict, Any
import numpy as np

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor: torch.Tensor, mantissa: torch.Tensor, fixed: torch.Tensor, var: torch.Tensor, hist: torch.Tensor, reshuff: bool=False) -> torch.Tensor:
        """
        使用 torch 实现 Hans 解码，还原 input_tensor

        Args:
            mantissa: [1, M], 尾数部分, float32/f16/bf16
            fixed: [1, F], 压缩后的固定部分, float32/f16/bf16
            var: [1, V],未压缩部分, float32/f16/bf16
            hist: [1, 256], int32,指数位统计
            reshuff: 是否启用内存重整（占位参数）

        Returns:
            input_tensor: [1, N], float32/f16/bf16, 原始输入
        """
        return input_tensor

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
from typing import Tuple, Dict, Any
import numpy as np

def get_inputs(param, device=None):
    """
    生成输入数据，用于 forward 方法。

    Args:
        param (dict): 包含 'input_shape', 'dtype', 'with_group_idx', 'num_groups' 的字典
        device (torch.device): 设备，如 'cpu' 或 'npu'

    Returns:
        tuple: (input_tensor, pdf_ref, statistic, reshuff)
    """
    input_shape = eval(param.get('input_shape', '[4, 1024, 64, 64]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    assert dtype in [torch.float32, torch.float16, torch.bfloat16], 'dtype must be float32/float16/bfloat16'
    d_type_dict = {'float32': np.float32, 'float16': np.float16}
    np_type = d_type_dict[dtype_str]
    size = np.prod(input_shape)
    np.random.seed(1234)
    inputs = np.random.random(size).reshape(input_shape).astype(np_type)
    if dtype_str == 'float32':
        exp_array = inputs.view(np.uint8).reshape(-1, 4)[:, 3]
        mantissa_uint8 = inputs.view(np.uint8).reshape(-1, 4)[:, :3].flatten()
    else:
        exp_array = inputs.view(np.uint8).reshape(-1, 2)[:, 1]
        mantissa_uint8 = inputs.view(np.uint8).reshape(-1, 2)[:, 0]
    print('Element mantissa data type:', mantissa_uint8.dtype)
    hist = np.bincount(exp_array, minlength=256)
    print('Element hist data type:', hist.dtype)
    fixed_uint16 = gen_encode_golden_np(exp_array, hist)
    print('Element fixed data type:', fixed_uint16.dtype)
    print('hist shape:', hist.shape)
    print('fixed shape:', fixed_uint16.shape)
    print('mantissa shape:', mantissa_uint8.shape)
    if dtype_str == 'float32':
        mantissa = mantissa_uint8.view(np_type)
        fixed = fixed_uint16.view(np_type)
    else:
        mantissa_uint8 = np.ascontiguousarray(mantissa_uint8)
        mantissa = mantissa_uint8.view(np_type)
        fixed = fixed_uint16.view(np_type)
    print('fixed shape:', fixed.shape)
    print('Element fixed data type:', fixed.dtype)
    print('mantissa shape:', mantissa.shape)
    print('Element mantissa data type:', mantissa.dtype)
    reshuff = True if param.get('reshuff', False) == 1 else False
    hist = torch.from_numpy(hist).to(device).to(torch.int32)
    fixed = torch.from_numpy(fixed).to(device).to(dtype)
    mantissa = torch.from_numpy(mantissa).to(device).to(dtype)
    inputs_tensor = torch.from_numpy(inputs).to(device).to(dtype)
    var = inputs_tensor.flatten()[:fixed.size(0)].to(dtype)
    print('Element var data type:', var.dtype)
    inputs_tensor = inputs_tensor.view(1, -1)
    return [inputs_tensor, mantissa, fixed, var, hist, reshuff]

def gen_encode_golden_pytorch(exp_array: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
    """
    使用 PyTorch 实现 gen_encode_golden，输出为 torch.Tensor
    """
    exp_array_uint8 = exp_array.to(torch.uint8)
    if exp_array_uint8.size(-1) == 4:
        exp_array_view = exp_array_uint8.view(-1, 4)
    else:
        exp_array_view = exp_array_uint8.view(-1, 2)
    numel = exp_array.size(0)
    big_loop = numel // 4096
    max_bit = torch.max(exp_array_view[:, 0].view(-1, 64), dim=1)[0]
    max_bit = max_bit.unsqueeze(1).expand(-1, 64).reshape(-1, 64)
    max_bit = max_bit.flatten()
    buffer = torch.zeros(4096, dtype=torch.int32, device=exp_array.device)
    block_bit_num = torch.zeros(64, dtype=torch.int32, device=exp_array.device)
    meta_info = torch.zeros(128, dtype=torch.int32, device=exp_array.device)
    meta_info[0] = 12138
    meta_info[1] = 1
    meta_info[2] = numel // 64
    meta_info[3] = numel // 64
    meta_info[4] = 0
    outputs = [meta_info.to(torch.uint16)]
    for i in range(big_loop):
        max_bit_this_loop = max_bit[i * 64:(i + 1) * 64]
        shr_scale = (2 ** max_bit_this_loop).reshape(64, -1)
        block_bit_num += max_bit_this_loop
        buffer = (buffer.view(-1, 64) * shr_scale).flatten()
        buffer += exp_array[i * 4096:(i + 1) * 4096]
        cmp_mask = block_bit_num > 16
        sum_cmp = torch.sum(cmp_mask)
        if sum_cmp > 0:
            reduce_mask = cmp_mask.repeat(64).reshape(-1)
            outputs.append((buffer[reduce_mask] & 65535).to(torch.uint16))
            buffer[reduce_mask] = buffer[reduce_mask] >> 16
            outputs.append(max_bit_this_loop.to(torch.uint16))
            block_bit_num[cmp_mask] = block_bit_num[cmp_mask] - 16
        else:
            outputs.append(max_bit_this_loop.to(torch.uint16))
    outputs.append((buffer & 65535).to(torch.uint16))
    outputs.append(block_bit_num.to(torch.uint16))
    meta_info[8] = torch.sum(torch.tensor([out.numel() for out in outputs])) * 2
    compress = torch.cat(outputs, dim=0)
    fixed = torch.zeros_like(compress)
    fixed[:compress.size(0)] = compress
    return fixed

def rank_elements_with_index(arr):
    sorted_indices = np.argsort(-arr)
    rank = np.empty_like(sorted_indices)
    last_value = None
    last_rank = -1
    for idx, sorted_idx in enumerate(sorted_indices):
        if arr[sorted_idx] != last_value:
            last_rank = idx
        rank[sorted_idx] = last_rank
        last_value = arr[sorted_idx]
    sorted_with_original_indices = [(value, idx) for idx, value in enumerate(arr)]
    sorted_with_original_indices.sort(key=lambda x: (-x[0], x[1]))
    final_rank = np.empty(len(arr), dtype=int)
    for i, (_, original_idx) in enumerate(sorted_with_original_indices):
        final_rank[original_idx] = i
    return final_rank

def gen_encode_golden_np(exp_array, hist):
    fixed = np.zeros_like(exp_array, dtype=np.uint8).view(np.uint16)
    ranking = rank_elements_with_index(hist)
    exp_sort_idx = ranking[exp_array]
    exp_bit_num = np.array([max(1, int(x).bit_length()) for x in exp_sort_idx])
    buffer = np.zeros(4096, dtype=np.int32)
    block_bit_num = np.zeros(64, dtype=np.int32)
    meta_info = np.zeros(128, dtype=np.int32)
    meta_info[0] = 12138
    meta_info[1] = 1
    meta_info[2] = exp_array.size // 64
    meta_info[3] = exp_array.size // 64
    meta_info[4] = 0
    max_bit = np.max(np.array([exp_bit_num]).reshape(-1, 64), axis=1)
    big_loop = exp_array.size // 4096
    outputs = [meta_info.view(np.uint16)]
    output_acculmulate_size = 0
    for i in range(big_loop):
        max_bit_this_loop = max_bit[i * 64:(i + 1) * 64]
        shr_scale = (2 ** max_bit_this_loop).reshape(64, -1)
        block_bit_num += max_bit_this_loop
        buffer = (buffer.reshape(-1, 64) * shr_scale).flatten()
        buffer += exp_sort_idx[i * 4096:(i + 1) * 4096]
        cmp_mask = block_bit_num > 16
        sum_cmp = np.sum(cmp_mask)
        if sum_cmp > 0:
            reduce_mask = np.tile(cmp_mask.reshape(-1, 1), 64).flatten()
            outputs.append((buffer[reduce_mask] & 65535).astype(np.uint16))
            buffer[reduce_mask] = buffer[reduce_mask] >> 16
            outputs.append(max_bit_this_loop.astype(np.uint16))
            block_bit_num[cmp_mask] = block_bit_num[cmp_mask] - 16
            output_acculmulate_size += np.sum(reduce_mask) * 2 + 64 * 2
        else:
            outputs.append(max_bit_this_loop.astype(np.uint16))
            output_acculmulate_size += 64 * 2
    outputs.append((buffer & 65535).astype(np.uint16))
    output_acculmulate_size += 4096 * 2
    outputs.append(block_bit_num.view(np.uint16))
    output_acculmulate_size += 64 * 4
    meta_info[8] = output_acculmulate_size
    compress = np.concatenate(outputs, axis=0)
    fixed[:compress.size] = compress
    return fixed

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for sinh.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
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
    json_path = os.path.join(os.path.dirname(__file__), "HansDecode.json")
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
