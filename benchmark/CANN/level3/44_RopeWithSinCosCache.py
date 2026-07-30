"""Auto-generated benchmark file for RopeWithSinCosCache.

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
from typing import List, Tuple
import torch
import torch.nn as nn

class Model(nn.Module):
    """PyTorch golden model for RoPE with sin/cos cache."""

    def __init__(self, num_q_heads: int, num_kv_heads: int, head_size: int, is_neox_style: bool=True, mrope_section: list=None, cache_mode: int=0):
        super(Model, self).__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.is_neox_style = is_neox_style
        self.mrope_section = mrope_section if mrope_section is not None else [0, 0, 0]
        self.cache_mode = cache_mode

    def forward(self, positions: torch.Tensor, query_in: torch.Tensor, key_in: torch.Tensor, cos_sin_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query_out = self._apply_rope(positions, query_in, cos_sin_cache, self.num_q_heads)
        key_out = self._apply_rope(positions, key_in, cos_sin_cache, self.num_kv_heads)
        return (query_out, key_out)

    def _apply_rope(self, positions: torch.Tensor, x: torch.Tensor, cos_sin_cache: torch.Tensor, num_heads: int) -> torch.Tensor:
        num_tokens = x.shape[0]
        head_size = self.head_size
        rotary_dim = cos_sin_cache.shape[-1]
        orig_dtype = x.dtype
        x_heads = x.reshape(num_tokens, num_heads, head_size).float()
        x_rot = x_heads[:, :, :rotary_dim]
        x_pass = x_heads[:, :, rotary_dim:]
        cos_all = cos_sin_cache[:, :rotary_dim // 2].float()
        sin_all = cos_sin_cache[:, rotary_dim // 2:].float()
        if len(positions.shape) == 1:
            pos = positions.long()
        else:
            pos = positions[0].long()
        cos_vals = cos_all[pos]
        sin_vals = sin_all[pos]
        cos_vals = cos_vals.unsqueeze(1)
        sin_vals = sin_vals.unsqueeze(1)
        if self.is_neox_style:
            x1 = x_rot[:, :, :rotary_dim // 2]
            x2 = x_rot[:, :, rotary_dim // 2:]
            out_rot = torch.cat([x1 * cos_vals - x2 * sin_vals, x2 * cos_vals + x1 * sin_vals], dim=-1)
        else:
            x1 = x_rot[:, :, 0::2]
            x2 = x_rot[:, :, 1::2]
            rotated = torch.cat([x1 * cos_vals - x2 * sin_vals, x2 * cos_vals + x1 * sin_vals], dim=-1)
            out_rot = torch.stack([rotated[:, :, :rotary_dim // 2], rotated[:, :, rotary_dim // 2:]], dim=-1)
            out_rot = out_rot.reshape(num_tokens, num_heads, rotary_dim)
        out_heads = torch.cat([out_rot.to(orig_dtype), x_pass.to(orig_dtype)], dim=-1)
        return out_heads.reshape(num_tokens, num_heads * head_size)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import math

def get_inputs(param, device=None):
    """Generate input tensors for RoPE with sin/cos cache."""
    num_tokens = int(param['num_tokens'])
    num_q_heads = int(param['num_q_heads'])
    num_kv_heads = int(param['num_kv_heads'])
    head_size = int(param['head_size'])
    rotary_dim = int(param['rotary_dim'])
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    max_pos = int(param.get('max_pos', 128))
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64, device=device)
    query_in = torch.randn(num_tokens, num_q_heads * head_size, dtype=dtype, device=device)
    key_in = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device=device)
    cos_sin_cache = torch.randn(max_pos, rotary_dim, dtype=dtype, device=device)
    return [positions, query_in, key_in, cos_sin_cache]

def get_init_inputs_per_case(param, device=None):
    """Extract initialization parameters."""
    num_q_heads = int(param['num_q_heads'])
    num_kv_heads = int(param['num_kv_heads'])
    head_size = int(param['head_size'])
    is_neox_style = bool(int(param.get('is_neox_style', 1)))
    mrope_section = [int(x) for x in param.get('mrope_section', '0,0,0').split(',')]
    cache_mode = int(param.get('cache_mode', 0))
    return [num_q_heads, num_kv_heads, head_size, is_neox_style, mrope_section, cache_mode]


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
    json_path = os.path.join(os.path.dirname(__file__), "RopeWithSinCosCache.json")
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
