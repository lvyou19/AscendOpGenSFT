"""Auto-generated benchmark file for RotaryPositionEmbedding.

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
CPU 金标准与 ops-transformer `posembedding/rotary_position_embedding/README.md` 中四种 mode 公式一致：
0 half，1 interleave，2 quarter（D 能被 4 整除），3 interleave-half（x_part1*cos + x_part2*sin）。
在 float32 上计算再 cast 回输入 dtype；cos/sin 广播到与 x 同形。
"""
from typing import List
import torch
import torch.nn as nn

def golden_rotary_position_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mode: int) -> torch.Tensor:
    od = x.dtype
    xf = x.float()
    cos_b = torch.broadcast_to(cos.float(), xf.shape)
    sin_b = torch.broadcast_to(sin.float(), xf.shape)
    d = xf.shape[-1]
    if mode == 0:
        half = d // 2
        x1 = xf[..., :half]
        x2 = xf[..., half:]
        x_rot = torch.cat((-x2, x1), dim=-1)
        y = xf * cos_b + x_rot * sin_b
    elif mode == 1:
        x1 = xf[..., ::2]
        x2 = xf[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape(*xf.shape)
        y = xf * cos_b + x_rot * sin_b
    elif mode == 2:
        q = d // 4
        x1 = xf[..., :q]
        x2 = xf[..., q:2 * q]
        x3 = xf[..., 2 * q:3 * q]
        x4 = xf[..., 3 * q:]
        x_rot = torch.cat((-x2, x1, -x4, x3), dim=-1)
        y = xf * cos_b + x_rot * sin_b
    elif mode == 3:
        x1 = xf[..., ::2]
        x2 = xf[..., 1::2]
        x_part1 = torch.cat((x1, x2), dim=-1)
        x_part2 = torch.cat((-x2, x1), dim=-1)
        y = x_part1 * cos_b + x_part2 * sin_b
    else:
        raise ValueError(f'unsupported mode {mode}')
    return y.to(od)

class Model(nn.Module):

    def __init__(self, mode: int=0):
        super().__init__()
        self.mode = int(mode)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, _out: torch.Tensor) -> List[torch.Tensor]:
        del _out
        return [golden_rotary_position_embedding(x, cos, sin, self.mode)]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import os
import random
from typing import Any, List, Optional, Tuple
import torch
_DEFAULT_PREPARE_RNG_BASE = 41

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
    seed = (base + _case_id_as_int(param) * 7919) % 2 ** 31
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

def _dtype_from_str(name: str) -> torch.dtype:
    name = str(name).strip().lower()
    if name in ('float16', 'fp16'):
        return torch.float16
    if name in ('bfloat16', 'bf16'):
        return torch.bfloat16
    if name in ('float32', 'fp32'):
        return torch.float32
    return torch.float16

def get_init_inputs_per_case(param: Any, device: Optional[torch.device]=None) -> Tuple[int]:
    mode = int(param.get('mode', 0))
    return (mode,)

def get_inputs(param: Any, device: Optional[torch.device]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _seed_prepare_rng(param)
    dev = device or torch.device('cpu')
    b = int(param['B'])
    n = int(param['N'])
    s = int(param['S'])
    d = int(param['D'])
    dtype = _dtype_from_str(param.get('dtype', 'float16'))
    mode = int(param.get('mode', 0))
    if mode == 2 and d % 4 != 0:
        raise ValueError('mode=2 requires D divisible by 4')
    if mode in (0, 1, 3) and d % 2 != 0:
        raise ValueError('mode requires even D')
    x_shape = (b, n, s, d)
    x = (torch.rand(x_shape, device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    cos_t = (torch.rand(1, 1, s, d, device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    sin_t = (torch.rand(1, 1, s, d, device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    out = torch.empty_like(x)
    return (x, cos_t, sin_t, out)


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
    json_path = os.path.join(os.path.dirname(__file__), "RotaryPositionEmbedding.json")
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
