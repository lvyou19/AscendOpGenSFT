"""Auto-generated benchmark file for RopeQuantKvcache.

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
CPU 金标准：qkv 切分 → 对 q/k 做与 AscendC 一致的 half RoPE → k/v 按 per-D 的 scale/offset 量化并写入 cache。
内核未将 RoPE 后的 k 写回 k_out GM，精度验证仅比对 q（见 prepare_inputs.custom_check_precision）。
"""
from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
QUANT_MIN = -128
QUANT_MAX = 127

def _rope_match_kernel(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    sin_adj = sin.clone()
    sin_adj[..., :half] = -sin_adj[..., :half]
    x_rot = torch.cat((x[..., half:], x[..., :half]), dim=-1)
    return x * cos + x_rot * sin_adj

def _quant_to_cache(x_f16: torch.Tensor, quant_scale: torch.Tensor, quant_offset: torch.Tensor) -> torch.Tensor:
    scale = quant_scale.to(torch.float32).reshape(1, 1, 1, -1)
    off = quant_offset.to(torch.float32).reshape(1, 1, 1, -1)
    t = x_f16.float() / scale + off
    t = torch.round(t)
    return torch.clamp(t, min=QUANT_MIN, max=QUANT_MAX).to(torch.int8)

def _scatter_cache(cache: torch.Tensor, quantized: torch.Tensor, indices: torch.Tensor) -> None:
    b = quantized.shape[0]
    s = quantized.shape[1]
    for bi in range(b):
        iv = int(indices[bi].item())
        cache[bi, iv:iv + s, :, :].copy_(quantized[bi])

def _run_rope_quant(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, indices: torch.Tensor, quant_scale: torch.Tensor, quant_offset: torch.Tensor, size_splits: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s, _ = qkv.shape
    sq, sk, sv = size_splits
    d = quant_scale.numel()
    n_q = sq // d
    n_kv = sk // d
    assert sq == n_q * d and sk == n_kv * d and (sv == n_kv * d)
    q_flat, k_flat, v_flat = qkv.split([sq, sk, sv], dim=-1)
    q_b = q_flat.reshape(b, s, n_q, d)
    k_b = k_flat.reshape(b, s, n_kv, d)
    v_b = v_flat.reshape(b, s, n_kv, d)
    cos_b = torch.broadcast_to(cos.float(), q_b.shape)
    sin_b = torch.broadcast_to(sin.float(), q_b.shape)
    cos_k = torch.broadcast_to(cos.float(), k_b.shape)
    sin_k = torch.broadcast_to(sin.float(), k_b.shape)
    rope_q = _rope_match_kernel(q_b.float(), cos_b, sin_b).to(torch.float16)
    rope_k = _rope_match_kernel(k_b.float(), cos_k, sin_k).to(torch.float16)
    k_q = _quant_to_cache(rope_k, quant_scale, quant_offset)
    v_q = _quant_to_cache(v_b, quant_scale, quant_offset)
    kc = k_cache.clone()
    vc = v_cache.clone()
    _scatter_cache(kc, k_q, indices)
    _scatter_cache(vc, v_q, indices)
    return (rope_q, rope_k, v_b, kc, vc)

class Model(nn.Module):

    def __init__(self, size_splits: Tuple[int, int, int], kv_output: bool=True, layout: str='BSND'):
        super().__init__()
        self.size_splits = tuple(size_splits)
        self.kv_output = kv_output
        self.layout = layout

    def forward(self, qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, quant_scale: torch.Tensor, quant_offset: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, indices: torch.Tensor, _q_buf: torch.Tensor, _k_buf: torch.Tensor, _v_buf: torch.Tensor) -> List[torch.Tensor]:
        del _q_buf, _k_buf, _v_buf
        return list(_run_rope_quant(qkv, cos, sin, k_cache, v_cache, indices, quant_scale, quant_offset, self.size_splits))

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

def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    try:
        import pandas as pd
        if pd.isna(v):
            return False
    except Exception:
        pass
    return str(v).strip().lower() in ('true', '1', 'yes')

def get_init_inputs_per_case(param: Any, device: Optional[torch.device]=None) -> Tuple[Any, ...]:
    row = param
    B = int(row['B'])
    S = int(row['S'])
    n_q = int(row['n_q'])
    n_kv = int(row['n_kv'])
    D = int(row['D'])
    sq = n_q * D
    sk = n_kv * D
    sv = n_kv * D
    kv_output = _parse_bool(row.get('kv_output', True))
    layout = str(row.get('layout', 'BSND')).strip() or 'BSND'
    return ((sq, sk, sv), kv_output, layout)

def get_inputs(param: Any, device: Optional[torch.device]=None) -> Tuple[Any, ...]:
    _seed_prepare_rng(param)
    dev = device or torch.device('cpu')
    B = int(param['B'])
    S = int(param['S'])
    n_q = int(param['n_q'])
    n_kv = int(param['n_kv'])
    D = int(param['D'])
    cache_sl = int(param['cache_sl'])
    use_offset = _parse_bool(param.get('use_offset', False))
    sq, sk, sv = (n_q * D, n_kv * D, n_kv * D)
    H = sq + sk + sv
    if cache_sl < S + 2:
        raise ValueError('cache_sl must be >= S + 2 for scatter slice')
    qkv = torch.randn(B, S, H, dtype=torch.float32, device=dev).clamp(-2.0, 2.0).to(torch.float16)
    cos = torch.randn(B, S, 1, D, dtype=torch.float32, device=dev).clamp(-1.0, 1.0).to(torch.float16)
    sin = torch.randn(B, S, 1, D, dtype=torch.float32, device=dev).clamp(-1.0, 1.0).to(torch.float16)
    quant_scale = torch.rand(D, dtype=torch.float32, device=dev) * 0.05 + 0.01
    quant_offset = torch.randint(-3, 3, (D,), dtype=torch.int32, device=dev) if use_offset else torch.zeros(D, dtype=torch.int32, device=dev)
    k_cache = torch.randint(-64, 64, (B, cache_sl, n_kv, D), dtype=torch.int8, device=dev)
    v_cache = torch.randint(-64, 64, (B, cache_sl, n_kv, D), dtype=torch.int8, device=dev)
    max_start = max(0, cache_sl - S)
    indices = torch.randint(0, max_start + 1, (B,), dtype=torch.int32, device=dev)
    q_shape = (B, S, n_q, D)
    k_shape = (B, S, n_kv, D)
    v_shape = (B, S, n_kv, D)
    q_buf = torch.empty(q_shape, dtype=torch.float16, device=dev)
    k_buf = torch.empty(k_shape, dtype=torch.float16, device=dev)
    v_buf = torch.empty(v_shape, dtype=torch.float16, device=dev)
    return (qkv, cos, sin, quant_scale, quant_offset, k_cache, v_cache, indices, q_buf, k_buf, v_buf)


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
    json_path = os.path.join(os.path.dirname(__file__), "RopeQuantKvcache.json")
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
