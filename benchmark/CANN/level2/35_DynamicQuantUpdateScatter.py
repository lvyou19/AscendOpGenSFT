"""Auto-generated benchmark file for DynamicQuantUpdateScatter.

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
CPU golden 与 AICore 中 dynamic_quant_update_scatter_comm.h::ComputeQuant（int8）一致：
对 axis=-2 上每个长度为 D3 的切片单独做动态量化（与 tiling 中 quantReptNum * updateAxisShape 分段一致；
可选 smooth；127/abs 后 ReduceMin；乘回后 int32→fp16→int8 与内核 CAST 链对齐）。
scatter 偏移与 GetDetOffsetNeg2（1D / 2D indices）一致；axis 须为 -2（与 tiling 支持一致）。
"""
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

def _quantize_segment_fp32(u_fp32: np.ndarray, smooth_fp32: Optional[np.ndarray]) -> Tuple[np.ndarray, np.float32]:
    """u_fp32: 1D float32, length L. 返回 (q_int8[L], scale_fp)。与内核 CAST_RINT→int32→fp16→int8 对齐。"""
    x = u_fp32.astype(np.float64, copy=True)
    if smooth_fp32 is not None:
        x = x * smooth_fp32.astype(np.float64)
    ax = np.abs(x)
    ax = np.maximum(ax, 1e-38)
    inv_scale_per = 127.0 / ax
    inv_scale = float(np.min(inv_scale_per))
    scale_out = np.float32(1.0 / inv_scale)
    xs = x * inv_scale
    t = torch.from_numpy(xs.astype(np.float32))
    i32 = torch.round(t).to(torch.int32)
    h = i32.to(torch.float16)
    y = h.to(torch.int8).numpy()
    return (y, scale_out)

def golden_scatter(var: torch.Tensor, var_scale: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, smooth: Optional[torch.Tensor], axis: int, indices_rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if axis != -2:
        raise ValueError('golden only implements axis=-2 to match op_host tiling')
    d0, d1, d2, d3 = (int(var.shape[i]) for i in range(4))
    if tuple(var.shape) != tuple(updates.shape):
        raise ValueError('var and updates must share shape')
    exp_scale = (d0, d1, d2, 1)
    if tuple(var_scale.shape) != exp_scale:
        raise ValueError('var_scale must be (*var.shape[:-1], 1), same rank as var')
    dst_bs_stride = d2 * d3
    num_head = d1
    size_per_head = d3
    total = d0 * d1
    src_bs_stride = dst_bs_stride
    var_o = var.detach().clone()
    sc_o = var_scale.detach().clone()
    flat_v = var_o.view(-1)
    flat_s = sc_o.view(-1)
    upd = updates.detach().float()
    sm_np: Optional[np.ndarray] = None
    if smooth is not None:
        sm_np = smooth.detach().float().cpu().numpy().reshape(-1)
        if sm_np.size != d3:
            raise ValueError('smooth must match last dim')
    ind = indices.detach().cpu()
    if indices_rank == 1:
        ind1 = ind.numpy().astype(np.int64, copy=False).reshape(-1)
        if ind1.size < d0:
            raise ValueError('1D indices length must be >= d0')
        for g in range(total):
            u0, u1 = (g // d1, g % d1)
            index_idx = g // num_head
            valid_idx = int(ind1[index_idx])
            dst_offset = g * dst_bs_stride + valid_idx * size_per_head
            blk = np.empty((d2, d3), dtype=np.int8)
            scales_row = []
            for j in range(d2):
                uvec = upd[u0, u1, j, :].reshape(-1).cpu().numpy()
                q, sc = _quantize_segment_fp32(uvec, sm_np)
                blk[j, :] = q
                scales_row.append(sc)
            flat_v[dst_offset:dst_offset + src_bs_stride] = torch.from_numpy(blk.reshape(-1)).to(var_o.dtype)
            base_s = dst_offset // d3
            for j in range(d2):
                flat_s[base_s + j] = torch.tensor(scales_row[j], dtype=sc_o.dtype)
    elif indices_rank == 2:
        ind2 = ind.numpy().astype(np.int64, copy=False)
        if ind2.ndim != 2 or ind2.shape[1] != 2:
            raise ValueError('2D indices must be [K,2]')
        for g in range(total):
            u0, u1 = (g // d1, g % d1)
            index_idx = g // num_head
            bs_idx = int(ind2[index_idx, 0])
            valid_idx = int(ind2[index_idx, 1])
            actual_batch = bs_idx * num_head + g % num_head
            dst_offset = actual_batch * dst_bs_stride + valid_idx * size_per_head
            blk = np.empty((d2, d3), dtype=np.int8)
            scales_row = []
            for j in range(d2):
                uvec = upd[u0, u1, j, :].reshape(-1).cpu().numpy()
                q, sc = _quantize_segment_fp32(uvec, sm_np)
                blk[j, :] = q
                scales_row.append(sc)
            flat_v[dst_offset:dst_offset + src_bs_stride] = torch.from_numpy(blk.reshape(-1)).to(var_o.dtype)
            base_s = dst_offset // d3
            for j in range(d2):
                flat_s[base_s + j] = torch.tensor(scales_row[j], dtype=sc_o.dtype)
    else:
        raise ValueError('indices_rank must be 1 or 2')
    return (var_o, sc_o)

class Model(nn.Module):

    def __init__(self, reduce_str: str, axis: int, indices_rank: int):
        super().__init__()
        self.reduce_str = reduce_str
        self.axis = int(axis)
        self.indices_rank = int(indices_rank)

    def forward(self, var: torch.Tensor, var_scale: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor, smooth: Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        y, s = golden_scatter(var, var_scale, indices, updates, smooth, self.axis, self.indices_rank)
        return [y, s]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import os
import random
import torch
_DEFAULT_PREPARE_RNG_BASE = 34

def _case_id_as_int(param):
    cid = param.get('case_id', 0)
    try:
        import pandas as pd
        if pd.isna(cid):
            return 0
    except Exception:
        pass
    return int(cid)

def _seed_prepare_rng(param):
    base = int(os.environ.get('NKB_PREPARE_INPUTS_SEED', str(_DEFAULT_PREPARE_RNG_BASE)))
    seed = (base + _case_id_as_int(param)) % 2 ** 31
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.npu.manual_seed(seed)
    except Exception:
        pass

def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('true', '1', 'yes')

def _normalize_reduce_str(v, default='update'):
    """CSV 空单元会变成 NaN，勿传给 aclnn 成字符串 'nan'（tiling 仅接受 update/none/空串）。"""
    if v is None:
        return default
    try:
        import pandas as pd
        if pd.isna(v):
            return default
    except Exception:
        pass
    if isinstance(v, float) and v != v:
        return default
    s = str(v).strip()
    if not s or s.lower() == 'nan':
        return default
    return s

def get_init_inputs_per_case(param, device=None):
    reduce_str = _normalize_reduce_str(param.get('reduce', 'update'))
    axis = int(param.get('axis', -2))
    indices_rank = int(param.get('indices_rank', 1))
    return (reduce_str, axis, indices_rank)

def get_inputs(param, device=None):
    _seed_prepare_rng(param)
    var_shape = eval(param.get('var_shape', '[2,3,1,32]'), {'__builtins__': {}})
    if len(var_shape) != 4 or var_shape[2] < 1:
        raise ValueError('var_shape must be 4D [D0,D1,D2,D3] with D2>=1')
    d0, d1, d2, d3 = var_shape
    indices_rank = int(param.get('indices_rank', 1))
    idtype_str = param.get('indices_dtype', 'int32')
    indices_dtype = getattr(torch, idtype_str) if hasattr(torch, idtype_str) else torch.int32
    upd_dtype_str = param.get('updates_dtype', 'float16')
    upd_dtype = getattr(torch, upd_dtype_str)
    var = torch.randint(-64, 64, var_shape, device=device, dtype=torch.int8)
    scale_shape = tuple(var_shape[:-1]) + (1,)
    var_scale = (torch.rand(scale_shape, device=device, dtype=torch.float32) * 0.5 + 0.1).to(torch.float32)
    if indices_rank == 1:
        idx_len = d0
        indices = torch.zeros(idx_len, device=device, dtype=indices_dtype)
        for i in range(idx_len):
            max_v = max(0, d2 - 1)
            indices[i] = torch.randint(0, max_v + 1, (1,), device=device, dtype=indices_dtype).item()
    else:
        idx_len = d0
        indices = torch.zeros((idx_len, 2), device=device, dtype=indices_dtype)
        for i in range(idx_len):
            bs = torch.randint(0, d0, (1,), device=device, dtype=indices_dtype).item()
            max_v = max(0, d2 - 1)
            vv = torch.randint(0, max_v + 1, (1,), device=device, dtype=indices_dtype).item()
            indices[i, 0] = bs
            indices[i, 1] = vv
    updates = (torch.rand(var_shape, device='cpu', dtype=torch.float32) * 2.0 - 1.0).to(device=device, dtype=upd_dtype)
    if _parse_bool(param.get('has_smooth', False)):
        smooth = (torch.rand((d3,), device='cpu', dtype=torch.float32) * 0.6 + 0.2).to(device=device, dtype=upd_dtype)
    else:
        smooth = None
    return (var, var_scale, indices, updates, smooth)


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
    json_path = os.path.join(os.path.dirname(__file__), "DynamicQuantUpdateScatter.json")
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
