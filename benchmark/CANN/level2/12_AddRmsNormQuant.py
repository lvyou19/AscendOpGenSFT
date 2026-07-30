"""Auto-generated benchmark file for AddRmsNormQuant.

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

class Model(nn.Module):

    def __init__(self, gamma: torch.Tensor, scales1: torch.Tensor, scales2: Optional[torch.Tensor]=None, zero_points1: Optional[torch.Tensor]=None, zero_points2: Optional[torch.Tensor]=None, axis: int=-1, epsilon: float=1e-06, div_mode: bool=True):
        super(Model, self).__init__()
        self.gamma = gamma.to(torch.float32).to('cpu')
        self.scales1 = scales1.to(torch.float32).to('cpu')
        self.scales2 = scales2
        if zero_points1 is not None:
            self.zero_points1 = zero_points1.to(torch.float32).to('cpu')
        else:
            self.zero_points1 = torch.zeros(self.gamma.shape, dtype=torch.float32, device='cpu')
        self.zero_points2 = zero_points2
        self.axis = axis
        self.epsilon = epsilon
        self.div_mode = div_mode

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        x = x1 + x2
        rms = torch.sqrt(x.pow(2).mean(dim=self.axis, keepdim=True) + self.epsilon)
        if self.div_mode:
            x_norm = x / rms
        else:
            x_norm = x * torch.rsqrt(rms + self.epsilon)
        y = x_norm * self.gamma
        if not self.div_mode:
            self.scales1 = 1.0 / self.scales1
        y1 = torch.quantize_per_channel(y, self.scales1, self.zero_points1, len(x1.shape) - len(self.gamma.shape), torch.qint8)
        return [y1.int_repr()]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import os
import random
import torch
_DEFAULT_PREPARE_RNG_BASE = 34
_BUNDLE_CACHE = {}

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
    cid = _case_id_as_int(param)
    seed = (base + cid * 7919) % 2 ** 31
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

def _csv_bool(param, key, default=True):
    v = param.get(key, default)
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ('false', '0', 'no'):
        return False
    return True

def _bundle_key(param, device):
    return (_case_id_as_int(param), str(device))

def _materialize_case_tensors(param, device=None):
    """
    仅此函数内进行 seed 与所有 torch.rand：生成该 case 的完整张量并缓存。
    batch_precision_eval 先调 get_init_inputs 再调 get_inputs，故首次实际在此构建；
    get_init_inputs 不再单独随机构造，避免两套 RNG / 两套数据不一致。
    """
    key = _bundle_key(param, device)
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]
    _seed_prepare_rng(param)
    input_shape = eval(param.get('input_shape', '[1]'))
    norm_shape = eval(param.get('normalized_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    scales_dtype_str = param.get('scales_dtype', 'float32')
    scales_dtype = getattr(torch, scales_dtype_str)
    zeros_dtype_str = param.get('zeros_dtype', 'int32')
    zeros_dtype = getattr(torch, zeros_dtype_str)
    x1 = torch.rand(input_shape, device=device, dtype=dtype) * 2 - 1
    x2 = torch.rand(input_shape, device=device, dtype=dtype) * 2 - 1
    gamma = torch.rand(norm_shape, device=device, dtype=dtype)
    scales1 = torch.rand(norm_shape, device=device, dtype=scales_dtype)
    scales2 = None
    has_zp1 = _csv_bool(param, 'has_zero_points1', True)
    zero_points1 = torch.rand(norm_shape, device=device, dtype=zeros_dtype) if has_zp1 else None
    zero_points2 = None
    axis = -1
    epsilon = param.get('epsilon', 1e-05)
    div_mode = _csv_bool(param, 'div_mode', True)
    bundle = {'x1': x1, 'x2': x2, 'gamma': gamma, 'scales1': scales1, 'scales2': scales2, 'zero_points1': zero_points1, 'zero_points2': zero_points2, 'axis': axis, 'epsilon': epsilon, 'div_mode': div_mode}
    _BUNDLE_CACHE[key] = bundle
    return bundle

def get_inputs(param, device=None):
    """
    返回 forward 用 (x1, x2)；与 get_init_inputs 共用 _materialize_case_tensors 的同一份缓存数据。
    """
    b = _materialize_case_tensors(param, device=device)
    return (b['x1'], b['x2'])

def get_init_inputs_per_case(param, device=None):
    """
    返回 Model / ModelNew 初始化参数列表；随机张量来自与 get_inputs 相同的 bundle，不在此函数内单独 rand。
    """
    b = _materialize_case_tensors(param, device=device)
    return [b['gamma'], b['scales1'], b['scales2'], b['zero_points1'], b['zero_points2'], b['axis'], b['epsilon'], b['div_mode']]


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
    json_path = os.path.join(os.path.dirname(__file__), "AddRmsNormQuant.json")
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
