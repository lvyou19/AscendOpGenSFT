"""Auto-generated benchmark file for LinearIndex.

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

class Model(nn.Module):
    """
    实现 LinearIndex 算子功能的标杆模型。
    """

    def __init__(self, axis=-1, combine=False):
        super(Model, self).__init__()
        self.axis = axis
        self.combine = combine

    def forward(self, indices: torch.Tensor, shape_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: 输入索引张量
            shape_tensor: 表示维度的 1D Int32 张量
        Returns:
            计算后的 int32 索引张量
        """
        shape_list = shape_tensor.tolist()
        rank = len(shape_list)
        axis = self.axis
        if axis < 0:
            axis += rank
        target_dim = shape_list[axis]
        out = torch.where(indices < 0, indices + target_dim, indices)
        if self.combine and rank == 3:
            stride = shape_list[1]
            if indices.dim() >= 2:
                if axis == 0:
                    cols = indices.shape[1]
                    col_indices = torch.arange(cols, device=indices.device, dtype=indices.dtype).unsqueeze(0)
                    col_indices = col_indices.expand_as(indices)
                    out = out * stride + col_indices
                elif axis == 1:
                    rows = indices.shape[0]
                    row_indices = torch.arange(rows, device=indices.device, dtype=indices.dtype).unsqueeze(1)
                    row_indices = row_indices.expand_as(indices)
                    out = out + stride * row_indices
        return out.to(torch.int32)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """
    构造输入数据。
    """
    input_shape = eval(param.get('input_shape', '[32, 64]'))
    dtype_str = param.get('dtype', 'int32')
    dtype = getattr(torch, dtype_str)
    shape_val = eval(param.get('shape_tensor', '[100, 100, 100]'))
    shape_tensor = torch.tensor(shape_val, dtype=torch.int32, device=device)
    axis = int(param.get('axis', -1))
    rank = len(shape_val)
    if axis < 0:
        axis += rank
    if axis >= rank:
        axis = rank - 1
    dim_size = shape_val[axis]
    indices = torch.randint(-dim_size, dim_size, input_shape, dtype=dtype, device=device)
    return (indices, shape_tensor)

def get_init_inputs_per_case(param, device=None):
    """
    获取模型初始化参数 (axis, combine)。
    """
    axis = int(param.get('axis', -1))
    combine = param.get('combine', 'False') == 'True'
    return [axis, combine]


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
    json_path = os.path.join(os.path.dirname(__file__), "LinearIndex.json")
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
