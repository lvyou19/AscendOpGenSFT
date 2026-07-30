"""Auto-generated benchmark file for DequantRopeQuantKvcache.

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
CPU 金标准与 ops-transformer `executor_aclnnDequantRopeQuantKvcache` 中 drqk 逻辑对齐：
dequant(可选) → split → RoPE（与内核一致：对 sin 前半取负后与 half-swap 组合）→ 量化并 scatter 到 k/v cache。
"""
from typing import List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
QUANT_MIN = -128
QUANT_MAX = 127

def _rope_match_kernel(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """与 AscendC 实现一致：对 sin 前半轴取负，尾维后半与前半交换后与 sin 组合，再与 x*cos 相加（见 op_kernel 中 RoPE 段）。"""
    d = x.shape[-1]
    half = d // 2
    sin_adj = sin.clone()
    sin_adj[..., :half] = -sin_adj[..., :half]
    x_rot = torch.cat((x[..., half:], x[..., :half]), dim=-1)
    return x * cos + x_rot * sin_adj

def _dequant_like_executor(inp: torch.Tensor, bias: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor], activation_scale: Optional[torch.Tensor]) -> torch.Tensor:
    if weight_scale is None:
        return inp
    ws = weight_scale.reshape(1, 1, -1)
    if activation_scale is not None:
        as_cpu = activation_scale.reshape(inp.shape[0], inp.shape[1], 1)
    else:
        as_cpu = None
    if bias is not None:
        bias_cpu = bias.reshape(1, 1, -1)
        if bias.dtype == torch.int32:
            t = torch.add(inp, bias_cpu)
            t = torch.mul(t, ws)
            if as_cpu is not None:
                t = torch.mul(t, as_cpu)
        else:
            t = torch.mul(inp, ws)
            if as_cpu is not None:
                t = torch.mul(t, as_cpu)
            t = torch.add(t, bias_cpu)
    else:
        t = torch.mul(inp, ws)
        if as_cpu is not None:
            t = torch.mul(t, as_cpu)
    return t

def _quant_update_scatter(key_cache: torch.Tensor, key: torch.Tensor, inv_scale: torch.Tensor, indice: torch.Tensor, offset: Optional[torch.Tensor], page_mode: bool) -> None:
    scale = inv_scale.reshape(-1, key.shape[-1])
    off = offset.reshape(-1, key.shape[-1]) if offset is not None else None
    if off is not None:
        quant_out = key.float() * scale + off
    else:
        quant_out = key.float() * scale
    quant_out = torch.round(quant_out)
    quant_out1 = torch.clamp(torch.round(quant_out.float()), min=QUANT_MIN, max=QUANT_MAX).to(torch.int8)
    if page_mode:
        d0, d1, d2, d3 = key_cache.shape
        key_cache_pa = key_cache.reshape(-1, key_cache.shape[-2], key_cache.shape[-1])
        quant_out2 = quant_out1.reshape(-1, quant_out1.shape[-2], quant_out1.shape[-1])
        for b in range(indice.shape[0]):
            iv = int(indice[b].item())
            key_cache_pa[iv] = quant_out2[b]
        key_cache.copy_(key_cache_pa.reshape(d0, d1, d2, d3))
    else:
        s_len = quant_out1.shape[1]
        for b in range(indice.shape[0]):
            iv = int(indice[b].item())
            key_cache[b, iv:iv + s_len, :, :].copy_(quant_out1[b])

def _run_drqk(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, indices: torch.Tensor, scale_k: torch.Tensor, scale_v: torch.Tensor, size_splits: Sequence[int], offset_k: Optional[torch.Tensor], offset_v: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor], activation_scale: Optional[torch.Tensor], bias: Optional[torch.Tensor], cache_mode: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    page_mode = str(cache_mode).lower() == 'page'
    x_work = x
    if x.dtype == torch.int32:
        x_work = _dequant_like_executor(x.float(), bias, weight_scale, activation_scale)
        x_work = x_work.to(cos.dtype)
    else:
        x_work = x
    if x_work.dim() == 2:
        x_work = x_work.unsqueeze(1)
    cos_u, sin_u = (cos, sin)
    if cos_u.dim() == 2:
        cos_u = cos_u.unsqueeze(1).unsqueeze(2)
        sin_u = sin_u.unsqueeze(1).unsqueeze(2)
    b, s, _ = x_work.shape
    h = k_cache.shape[-1]
    q, kt, vt = x_work.split(tuple(size_splits), dim=-1)
    q1 = q.reshape(b, s, -1, h)
    k1 = kt.reshape(b, s, -1, h)
    v1 = vt.reshape(b, s, -1, h)
    ropek = _rope_match_kernel(k1, cos_u, sin_u)
    ropeq = _rope_match_kernel(q1, cos_u, sin_u)
    inv_k = (1.0 / scale_k.to(torch.float32)).to(torch.float32)
    inv_v = (1.0 / scale_v.to(torch.float32)).to(torch.float32)
    kc = k_cache.clone()
    vc = v_cache.clone()
    _quant_update_scatter(kc, ropek, inv_k, indices, offset_k, page_mode)
    _quant_update_scatter(vc, v1, inv_v, indices, offset_v, page_mode)
    return (ropeq, ropek, v1, kc, vc)

class Model(nn.Module):

    def __init__(self, size_splits: Tuple[int, int, int], kv_output: bool=True, quant_mode: str='static', layout: str='BSND', cache_mode: str='contiguous'):
        super().__init__()
        self.size_splits = size_splits
        self.kv_output = kv_output
        self.quant_mode = quant_mode
        self.layout = layout
        self.cache_mode = cache_mode

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, indices: torch.Tensor, scale_k: torch.Tensor, scale_v: torch.Tensor, offset_k: Optional[torch.Tensor], offset_v: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor], activation_scale: Optional[torch.Tensor], bias: Optional[torch.Tensor], _q_buf: torch.Tensor, _k_buf: torch.Tensor, _v_buf: torch.Tensor) -> List[torch.Tensor]:
        del _q_buf, _k_buf, _v_buf
        ropeq, ropek, v1, kco, vco = _run_drqk(x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, self.size_splits, offset_k, offset_v, weight_scale, activation_scale, bias, self.cache_mode)
        return [ropeq, ropek, v1, kco, vco]

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
    quant_mode = str(row.get('quant_mode', 'static')).strip() or 'static'
    layout = str(row.get('layout', 'BSND')).strip() or 'BSND'
    cache_mode = str(row.get('cache_mode', 'contiguous')).strip() or 'contiguous'
    return ((sq, sk, sv), kv_output, quant_mode, layout, cache_mode)

def _dtype_from_str(name: str) -> torch.dtype:
    name = str(name).strip().lower()
    if name in ('float16', 'fp16'):
        return torch.float16
    if name in ('bfloat16', 'bf16'):
        return torch.bfloat16
    if name in ('float32', 'fp32'):
        return torch.float32
    if name in ('int32',):
        return torch.int32
    return torch.float16

def get_inputs(param: Any, device: Optional[torch.device]=None) -> Tuple[Any, ...]:
    _seed_prepare_rng(param)
    dev = device or torch.device('cpu')
    B = int(param['B'])
    S = int(param['S'])
    n_q = int(param['n_q'])
    n_kv = int(param['n_kv'])
    D = int(param['D'])
    cache_sl = int(param['cache_sl'])
    x_dtype = _dtype_from_str(param.get('x_dtype', 'float16'))
    cos_dtype = _dtype_from_str(param.get('cos_dtype', 'float16'))
    use_offset = _parse_bool(param.get('use_offset', False))
    use_bias = _parse_bool(param.get('use_bias', False))
    use_as = _parse_bool(param.get('use_activation_scale', False))
    dequant_i32 = _parse_bool(param.get('dequant_int32', False))
    sq, sk, sv = (n_q * D, n_kv * D, n_kv * D)
    H = sq + sk + sv
    if cache_sl < S + 2:
        raise ValueError('cache_sl must be >= S + 2 for scatter slice')
    if dequant_i32:
        x = torch.randint(-20, 20, (B, S, H), dtype=torch.int32, device=dev)
        weight_scale = torch.rand(H, dtype=torch.float32, device=dev) * 0.1 + 0.01
        bias = torch.randint(-5, 5, (H,), dtype=torch.int32, device=dev) if use_bias else None
        activation_scale = torch.rand(B, S, 1, dtype=torch.float32, device=dev) * 0.2 + 0.5 if use_as else None
    else:
        x = torch.randn(B, S, H, dtype=torch.float32, device=dev).clamp(-2.0, 2.0).to(x_dtype)
        weight_scale = None
        bias = torch.randn(H, dtype=torch.float32, device=dev) * 0.1 if use_bias else None
        if bias is not None and x_dtype == torch.float16:
            bias = bias.to(torch.float16)
        if bias is not None and x_dtype == torch.bfloat16:
            bias = bias.to(torch.bfloat16)
        activation_scale = torch.rand(B, S, 1, dtype=torch.float32, device=dev) * 0.2 + 0.5 if use_as else None
    cos = torch.randn(B, S, 1, D, dtype=torch.float32, device=dev).clamp(-1.0, 1.0).to(cos_dtype)
    sin = torch.randn(B, S, 1, D, dtype=torch.float32, device=dev).clamp(-1.0, 1.0).to(cos_dtype)
    k_cache = torch.randint(-64, 64, (B, cache_sl, n_kv, D), dtype=torch.int8, device=dev)
    v_cache = torch.randint(-64, 64, (B, cache_sl, n_kv, D), dtype=torch.int8, device=dev)
    max_start = max(0, cache_sl - S)
    indices = torch.randint(0, max_start + 1, (B,), dtype=torch.int32, device=dev)
    scale_k = torch.rand(n_kv * D, dtype=torch.float32, device=dev) * 0.05 + 0.01
    scale_v = torch.rand(n_kv * D, dtype=torch.float32, device=dev) * 0.05 + 0.01
    offset_k = torch.randn(n_kv * D, dtype=torch.float32, device=dev) * 0.1 if use_offset else None
    offset_v = torch.randn(n_kv * D, dtype=torch.float32, device=dev) * 0.1 if use_offset else None
    if offset_k is not None:
        offset_k = offset_k.to(dev)
    if offset_v is not None:
        offset_v = offset_v.to(dev)
    q_shape = (B, S, n_q, D)
    k_shape = (B, S, n_kv, D)
    v_shape = (B, S, n_kv, D)
    q_buf = torch.empty(q_shape, dtype=cos_dtype, device=dev)
    k_buf = torch.empty(k_shape, dtype=cos_dtype, device=dev)
    v_buf = torch.empty(v_shape, dtype=cos_dtype, device=dev)
    return (x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, offset_k, offset_v, weight_scale, activation_scale, bias, q_buf, k_buf, v_buf)


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
    json_path = os.path.join(os.path.dirname(__file__), "DequantRopeQuantKvcache.json")
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
