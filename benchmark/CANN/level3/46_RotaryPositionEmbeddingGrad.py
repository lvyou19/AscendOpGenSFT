"""Auto-generated benchmark file for RotaryPositionEmbeddingGrad.

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
CPU 金标准与 `posembedding/rotary_position_embedding_grad/README.md` / aclnn 文档中 half、interleave 的反向公式一致。
当前 Bench 目标核（如 910B）与 RotaryPositionEmbedding 一致，仅覆盖 mode 0、1；dcos/dsin 对 broadcast 轴求和。
"""
from typing import List, Tuple
import torch
import torch.nn as nn

def _sum_grad_to_broadcast_input(grad_full: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """将 [B,N,S,D] 上的梯度规约到 ref（如 cos/sin）的 shape。"""
    g = grad_full
    for d in range(grad_full.dim()):
        if ref.shape[d] == 1 and g.shape[d] != 1:
            g = g.sum(dim=d, keepdim=True)
    return g

def golden_rotary_position_embedding_grad(dy: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, x: torch.Tensor, mode: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dy_f = dy.float()
    cos_b = torch.broadcast_to(cos.float(), dy.shape)
    sin_b = torch.broadcast_to(sin.float(), dy.shape)
    xf = torch.broadcast_to(x.float(), dy.shape)
    d = dy.shape[-1]
    if mode == 0:
        half = d // 2
        dy1, dy2 = (dy_f[..., :half], dy_f[..., half:])
        cos1, cos2 = (cos_b[..., :half], cos_b[..., half:])
        sin1, sin2 = (sin_b[..., :half], sin_b[..., half:])
        x1, x2 = (xf[..., :half], xf[..., half:])
        dx = torch.cat((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1)
        g_dcos = dy_f * xf
        g_dsin = dy_f * torch.cat((-x2, x1), dim=-1)
    elif mode == 1:
        dy1, dy2 = (dy_f[..., ::2], dy_f[..., 1::2])
        cos1, cos2 = (cos_b[..., ::2], cos_b[..., 1::2])
        sin1, sin2 = (sin_b[..., ::2], sin_b[..., 1::2])
        x1, x2 = (xf[..., ::2], xf[..., 1::2])
        dx = torch.stack((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1).reshape(dy.shape)
        g_dcos = dy_f * xf
        rot_x = torch.stack((-x2, x1), dim=-1).reshape(xf.shape)
        g_dsin = dy_f * rot_x
    else:
        raise ValueError(f'unsupported mode {mode} for this validation')
    dcos = _sum_grad_to_broadcast_input(g_dcos, cos.float())
    dsin = _sum_grad_to_broadcast_input(g_dsin, sin.float())
    return (dx.to(dy.dtype), dcos.to(cos.dtype), dsin.to(sin.dtype))

class Model(nn.Module):

    def __init__(self, mode: int=0):
        super().__init__()
        self.mode = int(mode)

    def forward(self, dy: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, x: torch.Tensor, _dx: torch.Tensor, _dcos: torch.Tensor, _dsin: torch.Tensor) -> List[torch.Tensor]:
        del _dx, _dcos, _dsin
        dx, dcos, dsin = golden_rotary_position_embedding_grad(dy, cos, sin, x, self.mode)
        return [dx, dcos, dsin]

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

def get_inputs(param: Any, device: Optional[torch.device]=None) -> Tuple[torch.Tensor, ...]:
    _seed_prepare_rng(param)
    dev = device or torch.device('cpu')
    b = int(param['B'])
    n = int(param['N'])
    s = int(param['S'])
    d = int(param['D'])
    dtype = _dtype_from_str(param.get('dtype', 'float16'))
    mode = int(param.get('mode', 0))
    if mode in (0, 1) and d % 2 != 0:
        raise ValueError('mode 0/1 require even D')
    shape = (b, n, s, d)
    dy = (torch.rand(shape, device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    x = (torch.rand(shape, device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    cos_t = (torch.rand(1, 1, s, d, device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    sin_t = (torch.rand(1, 1, s, d, device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    dx = torch.empty_like(dy)
    dcos = torch.empty_like(cos_t)
    dsin = torch.empty_like(sin_t)
    return (dy, cos_t, sin_t, x, dx, dcos, dsin)


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
    json_path = os.path.join(os.path.dirname(__file__), "RotaryPositionEmbeddingGrad.json")
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
