"""Auto-generated benchmark file for TransQuantParamV2.

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
CPU golden：与 trans_quant_param_v2.h 中 round_mode==0 路径一致：
pack_scale = (float32_bits(scale) & 0xFFFFE000) | (1<<46)；
offset 按 CAST_RINT 到 int32 后 clamp 到 [-256,255]，取低 9 bit 左移 37 后与 pack_scale 按位或。
"""
import struct
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
DEQ_SCALE_MUL = 4294959104
QUANT_SCALE = 1 << 46
QUANT_MASK_0 = 511
OFFSET_DEVIATION = 37

def _pack_scale_rm0(f: float) -> np.uint64:
    u32 = struct.unpack('I', struct.pack('f', np.float32(f)))[0]
    return np.uint64(u32 & DEQ_SCALE_MUL | QUANT_SCALE)

def _offset_bits(f: float) -> np.uint64:
    v = int(np.rint(np.float32(f)))
    v = max(-256, min(255, v))
    return (np.uint64(v) & np.uint64(QUANT_MASK_0)) << np.uint64(OFFSET_DEVIATION)

def golden(scale: torch.Tensor, offset: Optional[torch.Tensor], round_mode: int) -> torch.Tensor:
    if int(round_mode) != 0:
        raise ValueError('validation 仅对齐 aclnnTransQuantParamV2（内部 roundMode 固定为 0）')
    s = scale.detach().cpu().numpy().astype(np.float32).reshape(-1)
    n = s.size
    out = np.zeros(n, dtype=np.uint64)
    if offset is None:
        for i in range(n):
            out[i] = _pack_scale_rm0(float(s[i]))
        return torch.from_numpy(out).to(device=scale.device)
    o = offset.detach().cpu().numpy().astype(np.float32).reshape(-1)
    m = o.size
    if m == 1 and n > 1:
        ob = _offset_bits(float(o[0]))
        for i in range(n):
            out[i] = _pack_scale_rm0(float(s[i])) | ob
    elif m == n:
        for i in range(n):
            out[i] = _pack_scale_rm0(float(s[i])) | _offset_bits(float(o[i]))
    else:
        raise ValueError(f'unsupported scale len {n} vs offset len {m}')
    return torch.from_numpy(out).to(device=scale.device)

class Model(nn.Module):

    def __init__(self, round_mode: int):
        super().__init__()
        self.round_mode = int(round_mode)

    def forward(self, scale: torch.Tensor, offset=None):
        return golden(scale, offset, self.round_mode)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def _parse_optional_shape(v):
    if v is None:
        return None
    try:
        import pandas as pd
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, float) and v != v:
        return None
    s = str(v).strip()
    if not s or s.lower() == 'nan':
        return None
    return eval(s, {'__builtins__': {}})

def get_init_inputs_per_case(param, device=None):
    round_mode = int(param.get('round_mode', 0))
    return (round_mode,)

def get_inputs(param, device=None):
    scale_shape = eval(param.get('scale_shape', '[8]'), {'__builtins__': {}})
    scale = (torch.rand(scale_shape, device=device, dtype=torch.float32) * 0.5 + 0.01).contiguous()
    off_shape = _parse_optional_shape(param.get('offset_shape', None))
    if off_shape is None:
        offset = None
    else:
        offset = (torch.rand(off_shape, device=device, dtype=torch.float32) * 0.2 - 0.1).contiguous()
    return (scale, offset)


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
    json_path = os.path.join(os.path.dirname(__file__), "TransQuantParamV2.json")
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
