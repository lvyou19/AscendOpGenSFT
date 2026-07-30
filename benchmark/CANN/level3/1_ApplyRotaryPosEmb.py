"""Auto-generated benchmark file for ApplyRotaryPosEmb.

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
CPU 金标准与 ascendc ApplyRotaryPosEmb（half rotary_mode，与 aclnn 默认一致）对齐：
对最后一维前半 sin 取负，将 x 后半与前半交换后与 sin' 相乘，再与 x*cos 相加（见 apply_rotary_pos_emb_small.h::ComputeTotary）。
"""
from typing import List, Tuple
import torch
import torch.nn as nn

def golden_apply_rotary_pos_emb_half(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """与内核 half 模式一致；在 float32 上计算再 cast 回输入 dtype。"""
    od = q.dtype
    qf = q.float()
    kf = k.float()
    cos_b = torch.broadcast_to(cos.float(), qf.shape)
    sin_b = torch.broadcast_to(sin.float(), qf.shape)
    d = qf.shape[-1]
    half = d // 2
    sin_neg = sin_b.clone()
    sin_neg[..., :half] = -sin_neg[..., :half]

    def one(x: torch.Tensor) -> torch.Tensor:
        x_rot = torch.cat([x[..., half:], x[..., :half]], dim=-1)
        return x * cos_b + x_rot * sin_neg
    qo = one(qf).to(od)
    ko = one(kf).to(od)
    return (qo, ko)

class Model(nn.Module):

    def __init__(self, layout: int=1):
        super().__init__()
        self.layout = layout

    def forward(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> List[torch.Tensor]:
        qo, ko = golden_apply_rotary_pos_emb_half(q, k, cos, sin)
        return [qo, ko]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import os
import random
from ast import literal_eval
from typing import Any, Dict, List, Tuple
import torch
_DEFAULT_PREPARE_RNG_BASE = 41
_BUNDLE_CACHE: Dict[Tuple[int, str], dict] = {}

def _case_id_as_int(param: Any) -> int:
    cid = param.get('case_id', 0)
    try:
        import pandas as pd
        if pd.isna(cid):
            return 0
    except Exception:
        pass
    return int(cid)

def _seed_prepare_rng(param: Any) -> None:
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

def _bundle_key(param: Any, device: Any) -> Tuple[int, str]:
    return (_case_id_as_int(param), str(device))

def _cos_sin_shape(q_shape: List[int], layout: int) -> List[int]:
    """BSND: [B,S,N,D] -> [B,S,1,D]；TND: [T,N,D] -> [T,1,D]。"""
    if layout == 1:
        b, s, _n, d = q_shape
        return [b, s, 1, d]
    if layout == 4:
        t, _n, d = q_shape
        return [t, 1, d]
    raise ValueError(f'unsupported layout {layout}')

def _materialize_case_tensors(param: Any, device: Any=None) -> dict:
    key = _bundle_key(param, device)
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]
    _seed_prepare_rng(param)
    layout = int(param.get('layout', 1))
    q_shape = list(literal_eval(str(param.get('q_shape', '[1,2,2,128]'))))
    dtype_str = str(param.get('dtype', 'float16'))
    dtype = getattr(torch, dtype_str)
    cos_shape = _cos_sin_shape(q_shape, layout)
    q = (torch.rand(q_shape, device=device, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    k = (torch.rand(q_shape, device=device, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    cos_t = (torch.rand(cos_shape, device=device, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    sin_t = (torch.rand(cos_shape, device=device, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    bundle = {'q': q, 'k': k, 'cos': cos_t, 'sin': sin_t, 'layout': layout}
    _BUNDLE_CACHE[key] = bundle
    return bundle

def get_inputs(param: Any, device: Any=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b = _materialize_case_tensors(param, device=device)
    return (b['q'], b['k'], b['cos'], b['sin'])

def get_init_inputs_per_case(param: Any, device: Any=None) -> List[int]:
    b = _materialize_case_tensors(param, device=device)
    return [b['layout']]


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
    json_path = os.path.join(os.path.dirname(__file__), "ApplyRotaryPosEmb.json")
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
