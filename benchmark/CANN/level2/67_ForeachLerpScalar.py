"""Auto-generated benchmark file for ForeachLerpScalar.

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
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], scalar_weight: float) -> List[torch.Tensor]:
        """
        Native PyTorch implementation of ForeachLerpScalar.
        Performs y_i = x1_i + weight * (x2_i - x1_i) for each tensor in the lists.
        """
        if not (isinstance(x1, list) and isinstance(x2, list)):
            raise TypeError('Inputs x1 and x2 must be lists of tensors.')
        if len(x1) != len(x2):
            raise ValueError('Input tensor lists x1 and x2 must have the same length.')
        output_list = []
        for i in range(len(x1)):
            result_tensor = torch.lerp(x1[i], x2[i], scalar_weight)
            output_list.append(result_tensor)
        return output_list

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
from typing import List, Tuple

def get_inputs(param, device=None) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    根据 DataFrame 行中的参数生成 ForeachLerpScalar 算子的输入张量。

    Args:
        param (dict): 参数配置，如输入形状列表和数据类型。
                      Expected keys: 'input_shapes_str' (e.g., '[[10, 20], [5, 5]]'),
                      'weight_scalar', 'dtype'.
        device (torch.device): 输入张量所在设备。

    Returns:
        tuple: 包含输入张量列表 (x1_list, x2_list, weight_scalar_tensor)。
    """
    shape_list = eval(param.get('input_shape', '[[1]]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    weight_scalar = param.get('weight_scalar', 0.5)
    x1_list = []
    x2_list = []
    for shape in shape_list:
        x1_list.append(torch.rand(shape, device=device, dtype=dtype))
        x2_list.append(torch.rand(shape, device=device, dtype=dtype))
    return (x1_list, x2_list, weight_scalar)

def get_init_inputs_per_case(param, device=None) -> List:
    """
    ForeachLerpScalar 没有模型初始化参数，返回空列表。

    Args:
        param (dict): 参数配置。

    Returns:
        list: 空列表。
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
    json_path = os.path.join(os.path.dirname(__file__), "ForeachLerpScalar.json")
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
