"""Auto-generated benchmark file for GroupQuant.

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
"""
CPU 金标准对齐 kernel group_quant_base.h::VecCompute：
x*scale(+offset) 在 fp32 上先 CAST_RINT→int32，再经 fp16 回读，最后 CAST_RINT→int8；
与 ops-nn-dev executor_aclnnGroupQuant 的分组边界语义一致。
"""
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
DTYPE_INT8 = 2
DTYPE_INT4 = 29

def _ascend_int8_from_float(y_fp32: np.ndarray) -> np.ndarray:
    i32 = np.rint(y_fp32).astype(np.int32)
    h = i32.astype(np.float16).astype(np.float32)
    out = np.rint(h)
    return np.clip(out, -128, 127).astype(np.int8)

def _ascend_int4_from_float(y_fp32: np.ndarray) -> np.ndarray:
    i32 = np.rint(y_fp32).astype(np.int32)
    h = i32.astype(np.float16).astype(np.float32)
    out = np.rint(h)
    return np.clip(out, -8, 7).astype(np.int8)

def golden_group_quant(x: torch.Tensor, scale: torch.Tensor, group_index: torch.Tensor, offset: Optional[torch.Tensor], dst_type: int) -> torch.Tensor:
    x_np = x.detach().float().cpu().numpy()
    scale_np = scale.detach().float().cpu().numpy()
    gi = group_index.detach().cpu().numpy().astype(np.int64)
    dim_s, dim_h = x_np.shape
    dim_e, dim_h2 = scale_np.shape
    if dim_h != dim_h2 or gi.shape[0] != dim_e:
        raise ValueError('shape mismatch x/scale/group_index')
    if int(gi[-1]) != dim_s:
        raise ValueError('group_index[-1] must equal S')
    off = 0.0
    if offset is not None:
        off = float(offset.detach().float().cpu().reshape(-1)[0])
    parts = []
    for row_scale in range(dim_e):
        r0 = 0 if row_scale == 0 else int(gi[row_scale - 1])
        r1 = int(gi[row_scale])
        if r0 < r1:
            blk = x_np[r0:r1] * scale_np[row_scale:row_scale + 1]
            blk = blk + off
            parts.append(blk)
    y_fp32 = np.concatenate(parts, axis=0)
    if int(dst_type) == DTYPE_INT8:
        y_np = _ascend_int8_from_float(y_fp32)
    elif int(dst_type) == DTYPE_INT4:
        y_np = _ascend_int4_from_float(y_fp32)
    else:
        raise ValueError('dst_type must be 2 (int8) or 29 (int4)')
    return torch.from_numpy(y_np).to(device=x.device)

class Model(nn.Module):

    def __init__(self, dst_type: int):
        super().__init__()
        self.dst_type = int(dst_type)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, group_index: torch.Tensor, offset: Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        y = golden_group_quant(x, scale, group_index, offset, self.dst_type)
        return [y]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('true', '1', 'yes')

def _make_group_index(s: int, e: int, device, dtype: torch.dtype):
    """单调非减边界：group_index[i] 为第 i 个 expert 负责行的右端点（开区间），末尾等于 S。"""
    if e < 1:
        raise ValueError('num_experts must be >= 1')
    base = s // e
    rem = s % e
    g = []
    cur = 0
    for i in range(e):
        cur += base + (1 if i < rem else 0)
        g.append(cur)
    assert g[-1] == s
    return torch.tensor(g, device=device, dtype=dtype)

def get_init_inputs_per_case(param, device=None):
    dst_type = int(param.get('dst_type', 2))
    return (dst_type,)

def get_inputs(param, device=None):
    shape = eval(param.get('input_shape', '[16, 32]'), {'__builtins__': {}})
    if len(shape) != 2:
        raise ValueError('GroupQuant x must be 2D [S, H]')
    s, h = (int(shape[0]), int(shape[1]))
    e = int(param.get('num_experts', 2))
    dst_type = int(param.get('dst_type', 2))
    if dst_type == 29 and h % 2 != 0:
        raise ValueError('int4 output requires even H')
    x_dtype_str = param.get('x_dtype', 'float16')
    scale_dtype_str = param.get('scale_dtype', x_dtype_str)
    gi_dtype_str = param.get('group_index_dtype', 'int32')
    x_dtype = getattr(torch, x_dtype_str)
    scale_dtype = getattr(torch, scale_dtype_str)
    gi_dtype = getattr(torch, gi_dtype_str)
    torch.manual_seed(int(param.get('case_id', 0)))
    x = (torch.rand((s, h), device='cpu', dtype=torch.float32) * 2.0 - 1.0).to(device=device, dtype=x_dtype)
    scale = (torch.rand((e, h), device='cpu', dtype=torch.float32) * 0.5 + 0.1).to(device=device, dtype=scale_dtype)
    group_index = _make_group_index(s, e, device, gi_dtype)
    if _parse_bool(param.get('has_offset', False)):
        off = torch.tensor([0.03], device=device, dtype=scale_dtype)
    else:
        off = None
    return (x, scale, group_index, off)


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
    json_path = os.path.join(os.path.dirname(__file__), "GroupQuant.json")
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
