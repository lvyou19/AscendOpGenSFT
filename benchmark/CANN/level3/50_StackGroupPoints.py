"""Auto-generated benchmark file for StackGroupPoints.

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

def _batch_idx_for_pt(pt_idx: int, ibc: torch.Tensor) -> int:
    """与 kernel 中 indices_batch_cnt 累计逻辑一致。"""
    b = ibc.numel()
    pt_cnt = int(ibc[0].item())
    bs_idx = 0
    for k in range(1, b):
        if pt_idx >= pt_cnt:
            bs_idx = k
            pt_cnt += int(ibc[k].item())
    return bs_idx

def _feature_range(bs_idx: int, fbc: torch.Tensor):
    """
    fbc 长度 B+1，与 kernel / ops-cv golden 一致：
    batch i 点数为 fbc[i]，循环用 fbc[k+1] 更新 end。
    """
    features_batch_start_idx = 0
    features_batch_end_idx = int(fbc[0].item())
    for k in range(bs_idx):
        features_batch_start_idx += int(fbc[k].item())
        features_batch_end_idx = features_batch_start_idx + int(fbc[k + 1].item())
    return (features_batch_start_idx, features_batch_end_idx)

def _stack_group_points_reference(features: torch.Tensor, features_batch_cnt: torch.Tensor, indices: torch.Tensor, indices_batch_cnt: torch.Tensor) -> torch.Tensor:
    """
    features: (N, C)；fbc / ibc: 长度 B 的 int32；
    indices: (M, nsample)；输出 (M, C, nsample)，与 infershape / kernel 索引顺序一致。
    """
    n, c = features.shape
    m, nsample = indices.shape
    b = indices_batch_cnt.numel()
    standard = m * c * nsample
    out = features.new_zeros((m, c, nsample))
    feat_flat = features.reshape(-1).float()
    fbc = features_batch_cnt.cpu()
    ibc = indices_batch_cnt.cpu()
    ind = indices.cpu().long()
    for pt_idx in range(m):
        for c_idx in range(c):
            for sample_idx in range(nsample):
                index = pt_idx * c * nsample + c_idx * nsample + sample_idx
                if index > standard:
                    continue
                bs_idx = _batch_idx_for_pt(pt_idx, ibc)
                fs, fe = _feature_range(bs_idx, fbc)
                tmp_cin = pt_idx * nsample + sample_idx
                if tmp_cin >= m * nsample:
                    continue
                cin = int(ind[pt_idx, sample_idx].item())
                in_idx = cin * c + c_idx
                if in_idx < fe * c and in_idx < n * c - fs * c:
                    fs_idx = in_idx + fs * c
                    if 0 <= fs_idx < n * c:
                        out[pt_idx, c_idx, sample_idx] = feat_flat[fs_idx]
    return out.to(dtype=features.dtype)

class Model(nn.Module):
    """CPU 金标准：与 stack_group_points kernel 中 Gather 逻辑一致。"""

    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor, features_batch_cnt: torch.Tensor, indices: torch.Tensor, indices_batch_cnt: torch.Tensor) -> torch.Tensor:
        return _stack_group_points_reference(features, features_batch_cnt, indices, indices_batch_cnt)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import numpy as np
import torch

def _split_int(total: int, parts: int, rng: np.random.Generator) -> list:
    """将 total 拆成 parts 个正整数之和。"""
    if parts <= 0 or total < parts:
        raise ValueError('invalid split')
    base = total // parts
    rem = total - base * parts
    out = [base + (1 if i < rem else 0) for i in range(parts)]
    for i in range(parts):
        if out[i] < 1:
            d = 1 - out[i]
            out[i] = 1
            j = (i + 1) % parts
            out[j] = max(1, out[j] - d)
    return out

def get_inputs(param, device=None):
    rng = np.random.default_rng(17 + int(param.get('case_id', 0)))
    dtype_map = {'float32': torch.float32, 'float': torch.float32, 'float16': torch.float16, 'half': torch.float16}
    dtype_str = str(param.get('dtype', 'float32')).lower()
    dtype = dtype_map.get(dtype_str, torch.float32)
    b = int(param.get('batch', 2))
    c = int(param.get('channels', 8))
    m = int(param.get('m', 16))
    nsample = int(param.get('nsample', 4))
    ibc_list = _split_int(m, b, rng)
    pts_per_batch = [int(rng.integers(6, 18)) + int(param.get('case_id', 0)) % 3 for _ in range(b)]
    for i in range(b):
        pts_per_batch[i] = max(pts_per_batch[i], ibc_list[i])
    n = sum(pts_per_batch)
    features = torch.tensor(rng.standard_normal((n, c), dtype=np.float32), dtype=dtype)
    fbc_core = torch.tensor(pts_per_batch, dtype=torch.int32)
    features_batch_cnt = torch.cat([fbc_core, torch.zeros(1, dtype=torch.int32)])
    indices_batch_cnt = torch.tensor(ibc_list, dtype=torch.int32)
    indices = torch.zeros((m, nsample), dtype=torch.int32)
    row = 0
    for bi in range(b):
        hi_cin = pts_per_batch[bi] - 1
        for _ in range(ibc_list[bi]):
            indices[row] = torch.tensor(rng.integers(0, max(hi_cin, 0) + 1, size=nsample, dtype=np.int32))
            row += 1
    if device:
        features = features.to(device)
        features_batch_cnt = features_batch_cnt.to(device)
        indices = indices.to(device)
        indices_batch_cnt = indices_batch_cnt.to(device)
    return (features, features_batch_cnt, indices, indices_batch_cnt)

def get_init_inputs_per_case(param, device=None):
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
    json_path = os.path.join(os.path.dirname(__file__), "StackGroupPoints.json")
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
