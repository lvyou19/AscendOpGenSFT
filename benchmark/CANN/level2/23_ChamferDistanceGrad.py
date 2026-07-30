"""Auto-generated benchmark file for ChamferDistanceGrad.

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

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, xyz1, xyz2, idx1, idx2, gradDist1, gradDist2):
        """
        Args:
            xyz1: [B, N, 2] - 点集1坐标
            xyz2: [B, N, 2] - 点集2坐标
            idx1: [B, N] - 每个xyz1点对应xyz2中的最近点索引（int32）
            idx2: [B, N] - 每个xyz2点对应xyz1中的最近点索引（int32）
            gradDist1: [B, N] - 从xyz1到最近xyz2点的距离梯度
            gradDist2: [B, N] - 从xyz2到最近xyz1点的距离梯度
        Returns:
            gradXyz1: [B, N, 2] - xyz1的梯度
            gradXyz2: [B, N, 2] - xyz2的梯度
        """
        B, N, _ = xyz1.shape
        _, M, _ = xyz2.shape
        gradXyz1 = torch.zeros_like(xyz1)
        gradXyz2 = torch.zeros_like(xyz2)
        for b in range(B):
            for n in range(N):
                x1, y1 = (xyz1[b, n, 0].item(), xyz1[b, n, 1].item())
                idx = idx1[b, n].item()
                x2, y2 = (xyz2[b, idx, 0].item(), xyz2[b, idx, 1].item())
                g = gradDist1[b, n].item() * 2.0
                gradXyz1[b, n, 0] += (x1 - x2) * g
                gradXyz1[b, n, 1] += (y1 - y2) * g
                gradXyz2[b, idx, 0] -= (x1 - x2) * g
                gradXyz2[b, idx, 1] -= (y1 - y2) * g
            for m in range(M):
                x2, y2 = (xyz2[b, m, 0].item(), xyz2[b, m, 1].item())
                idx = idx2[b, m].item()
                x1, y1 = (xyz1[b, idx, 0].item(), xyz1[b, idx, 1].item())
                g = gradDist2[b, m].item() * 2.0
                gradXyz2[b, m, 0] += (x2 - x1) * g
                gradXyz2[b, m, 1] += (y2 - y1) * g
                gradXyz1[b, idx, 0] -= (x2 - x1) * g
                gradXyz1[b, idx, 1] -= (y2 - y1) * g
        return [gradXyz1, gradXyz2]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np

def get_inputs(param, device=None):
    xyz1_shape = eval(param.get('xyz1_shape', '[2,2,2]'))
    xyz2_shape = eval(param.get('xyz2_shape', '[2,2,2]'))
    idx1_shape = eval(param.get('idx1_shape', '[2,2]'))
    idx2_shape = eval(param.get('idx2_shape', '[2,2]'))
    gradDist1_shape = eval(param.get('gradDist1_shape', '[2,2]'))
    gradDist2_shape = eval(param.get('gradDist2_shape', '[2,2]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    assert dtype in [torch.float32], 'dtype must be float32'
    np.random.seed(1234)
    torch.manual_seed(1234)
    xyz1 = torch.rand(xyz1_shape, dtype=dtype, device=device)
    xyz2 = torch.rand(xyz2_shape, dtype=dtype, device=device)
    idx1 = np.random.randint(0, xyz2_shape[1], size=idx1_shape, dtype=np.int32)
    idx2 = np.random.randint(0, xyz1_shape[1], size=idx2_shape, dtype=np.int32)
    gradDist1 = torch.rand(gradDist1_shape, dtype=dtype, device=device)
    gradDist2 = torch.rand(gradDist2_shape, dtype=dtype, device=device)
    idx1_t = torch.from_numpy(idx1).to(device).to(torch.int32)
    idx2_t = torch.from_numpy(idx2).to(device).to(torch.int32)
    return [xyz1, xyz2, idx1_t, idx2_t, gradDist1, gradDist2]

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Initialization parameters for the MSE Loss model
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
    json_path = os.path.join(os.path.dirname(__file__), "ChamferDistanceGrad.json")
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
