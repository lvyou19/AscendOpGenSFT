"""Auto-generated benchmark file for DequantSwigluQuant.

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
import torch.nn.functional as F
from typing import List
from typing import Optional, Tuple

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x_tensor: torch.Tensor, weight_scale: Optional[torch.Tensor]=None, activate_scale: Optional[torch.Tensor]=None, bias: Optional[torch.Tensor]=None, quant_scale: Optional[torch.Tensor]=None, quant_offset: Optional[torch.Tensor]=None, group_index: Optional[torch.Tensor]=None, activate_left: bool=False, quant_mode: str='static') -> List[torch.Tensor]:
        if group_index is None:
            group_index = torch.tensor([x_tensor.shape[0]])
        x_shape = list(x_tensor.shape)
        x_shape[-1] //= 2
        res_y = torch.zeros(x_shape, dtype=torch.float32, device=x_tensor.device)
        input_dtype = x_tensor.dtype
        offset = 0
        for g_idx in range(group_index.shape[0]):
            groupIdx = group_index[g_idx]
            x = x_tensor[offset:offset + groupIdx].float()
            if input_dtype == torch.int32:
                if bias is not None:
                    x = x + bias
                x = x * weight_scale[g_idx]
                if activate_scale is not None:
                    x = x * activate_scale[offset:offset + groupIdx]
            gate, up = torch.chunk(x, 2, dim=-1)
            if activate_left:
                output = torch.nn.functional.silu(gate) * up
            else:
                output = torch.nn.functional.silu(up) * gate
            if quant_mode == 'static':
                output = output / quant_scale[g_idx] + quant_offset[g_idx]
            elif quant_mode == 'dynamic':
                output = output * quant_scale[g_idx]
                abs = torch.abs(output)
                max_values = torch.amax(abs, dim=-1)
                scale_out = max_values / 127
                max_values = 127 / max_values
                output = output * max_values.unsqueeze(1)
            output = torch.clamp(output, -128, 127)
            output = torch.round(output)
            res_y[offset:offset + groupIdx] = output
            offset = offset + groupIdx
        return res_y.to(torch.int8)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
from typing import Optional, Tuple
import numpy as np

def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成 DequantSwigluQuant 算子的输入张量。

    Args:
        param (dict): 参数配置，如输入形状和数据类型
        device (torch.device): 输入张量所在设备

    Returns:
        tuple: 包含 DequantSwigluQuant 算子的所有输入张量和非张量参数
               (x, weight_scale, activate_scale, bias, quant_scale, quant_offset, group_index, activate_left, quant_mode)
    """
    shape = eval(param.get('input_shape', '[1, 2]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    x = torch.rand(shape, device=device, dtype=dtype)
    activate_left = param.get('activate_left', True)
    quant_mode = param.get('quant_mode', 'static')
    group_index = None
    if param.get('group_index_present', False):
        group_index_val = x.shape[0]
        count = int(param.get('group', 1))
        group_list = torch.randint(0, group_index_val + 1, (count,), dtype=torch.int32, device=device)
        group_list[-1] = group_index_val
        group_list, _ = torch.sort(group_list)
        group_list = [group_list[0]] + [group_list[i] - group_list[i - 1] for i in range(1, len(group_list))]
        group_index = torch.tensor(group_list, device=device, dtype=torch.int32)
    else:
        count = 1
    weight_scale = None
    if param.get('weight_scale_present', False):
        weight_scale_shape = [count, shape[-1]]
        weight_scale = torch.rand(weight_scale_shape, device=device, dtype=torch.float32)
    activate_scale = None
    if param.get('activate_scale_present', False):
        activate_scale_shape = list(shape[:-1]) + [1]
        activate_scale = torch.rand(activate_scale_shape, device=device, dtype=torch.float32)
    bias = None
    if param.get('bias_present', False):
        bias_shape = [shape[-1]]
        bias_dtype_str = param.get('dtype', 'float16')
        bias_dtype = getattr(torch, bias_dtype_str)
        bias = torch.rand(bias_shape, device=device, dtype=bias_dtype)
    quant_scale = None
    if param.get('quant_scale_present', True):
        quant_scale_shape = [count, x.shape[-1] // 2]
        quant_scale = torch.rand(quant_scale_shape, device=device, dtype=torch.float32)
    quant_offset = None
    if param.get('quant_offset_present', True):
        if quant_mode == 'static':
            quant_offset = torch.rand([count, x.shape[-1] // 2], device=device, dtype=torch.float32)
    return (x, weight_scale, activate_scale, bias, quant_scale, quant_offset, group_index, activate_left, quant_mode)

def get_init_inputs_per_case(param, device=None):
    """
    DequantSwigluQuant 没有模型初始化参数，返回空列表。

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
    json_path = os.path.join(os.path.dirname(__file__), "DequantSwigluQuant.json")
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
