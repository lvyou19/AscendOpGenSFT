"""Auto-generated benchmark file for HansEncode.

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

    def forward(self, input_tensor: Tensor, pdf_ref: Tensor, statistic: bool=True, reshuff: bool=False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        使用torch实现压缩做对比标杆。
        
        参数:
            input_tensor (Tensor): shape支持多维，要求数据数量是32768的倍数，dtype=float16/bfloat16/f32
            pdf_ref (Tensor): shape [1, 256]，dtype=int32
            statistic (bool): 是否进行 PDF 统计，默认为True
            reshuff (bool): 是否对各核编码后的结果进行内存重整
            
        返回:
            pdf_out（Tensor）: [1, 256], int32 指数位统计结果
            mantissa_out（Tensor）: 表示输出的尾数部分，dtype与input一致
            fixed_out（Tensor）: 表示压缩的第一段输出，dtype与input一致
            var_out（Tensor）: 表示压缩超过fixedOut后的输出，dtype与input一致
        """
        input_cpu = input_tensor.cpu().contiguous()
        dtype = input_cpu.dtype
        if dtype == torch.float32:
            exp_bytes = input_cpu.view(torch.uint8).view(-1, 4)[:, 3]
        elif dtype in [torch.float16, torch.bfloat16]:
            exp_bytes = input_cpu.view(torch.uint8).view(-1, 2)[:, 1]
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')
        pdf_out = torch.bincount(exp_bytes, minlength=256).to(torch.int32)
        pdf_out_tensor = pdf_out.to(input_tensor.device)
        return pdf_out_tensor

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
from typing import Tuple, Dict, Any

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
    input_tensor = torch.randn(input_shape, dtype=dtype, device=device)
    assert input_tensor.numel() % 64 == 0, 'input_tensor numel must be multiple of 64'
    assert input_tensor.numel() >= 32768, 'input_tensor numel must >= 32768'
    input_tensor = input_tensor.contiguous()
    pdf_ref = torch.zeros(1, 256, dtype=torch.int32, device=device)
    statistic = True if param.get('statistic', True) == 1 else False
    reshuff = True if param.get('reshuff', False) == 1 else False
    return (input_tensor, pdf_ref, statistic, reshuff)

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
    json_path = os.path.join(os.path.dirname(__file__), "HansEncode.json")
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
