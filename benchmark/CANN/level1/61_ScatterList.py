"""Auto-generated benchmark file for ScatterList.

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
from typing import List, Optional
import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    实现ScatterList算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, varRef: list[torch.Tensor], indice: torch.Tensor, updates: torch.Tensor, mask: Optional[torch.Tensor], reduce: str, axis: int) -> torch.Tensor:
        """
        实现ScatterList算子功能。

        Args:
            varRef: 第一个输入张量
            indice: 索引张量
            updates: 更新张量
            mask: 可选的掩码张量
            reduce: 规约操作类型
            axis: 指定的轴

        Returns:
            经过ScatterList操作后的结果张量
        """
        for i in range(len(varRef)):
            if mask[i] == False:
                continue
            dest_block_slice = slice(indice[i][0], indice[i][0] + indice[i][1])
            source_block_slice = slice(0, indice[i][1])
            num_dims = varRef[i].ndim
            dest_slicer = [slice(None)] * num_dims
            src_slicer = [slice(None)] * num_dims
            dest_slicer[axis] = dest_block_slice
            src_slicer[axis] = source_block_slice
            dest_slicer = tuple(dest_slicer)
            src_slicer = tuple(src_slicer)
            source_block = updates[i][src_slicer]
            varRef[i][dest_slicer] = source_block
        return varRef

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    varRefShape = eval(param.get('varRefShape', '[1]'))
    indiceShape = eval(param.get('indiceShape', '[1]'))
    updatesShape = eval(param.get('updatesShape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    idx_dtype = torch.int64
    varRef = [torch.randn(x, device=device, dtype=dtype) for x in varRefShape]
    axis = int(param.get('axis', -2))
    indice = torch.zeros(indiceShape, device=device, dtype=idx_dtype)
    for i in range(updatesShape[0]):
        length = torch.randint(1, updatesShape[axis] + 1, (1,)).item()
        max_start_index = updatesShape[axis] - length
        start_index = torch.randint(0, max_start_index + 1, (1,)).item()
        indice[i][0], indice[i][1] = (start_index, length)
    updates = torch.randn(updatesShape, device=device, dtype=dtype)
    mask = torch.tensor(eval(param.get('mask', '[False]')), device=device, dtype=torch.bool)
    reduce = param.get('reduce', 'update')
    return (varRef, indice, updates, mask, reduce, axis)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for ScatterList.

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
    json_path = os.path.join(os.path.dirname(__file__), "ScatterList.json")
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
