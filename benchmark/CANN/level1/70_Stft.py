"""Auto-generated benchmark file for Stft.

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

    def forward(self, x: torch.Tensor, window: torch.Tensor, n_fft: int, hop_length: int, win_length: int, normalized: bool, onesided: bool, return_complex: bool) -> torch.Tensor:
        if x.dtype == torch.complex64 or x.dtype == torch.complex128:
            onesided = False
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
            window = window.float()
        return torch.stft(x, n_fft, hop_length=hop_length, win_length=win_length, window=window, center=False, normalized=normalized, onesided=onesided, return_complex=return_complex)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).lower() == 'true'

def get_inputs(param, device=None):
    shape = eval(param.get('shape', '[64]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    n_fft = int(param.get('n_fft', '8'))
    hop_length = int(param.get('hop_length', '4'))
    win_length = int(param.get('win_length', '8'))
    normalized = _to_bool(param.get('normalized', False))
    onesided = _to_bool(param.get('onesided', True))
    return_complex = _to_bool(param.get('return_complex', True))
    L = shape[-1] if len(shape) >= 2 else shape[0]
    if n_fft > L:
        raise ValueError(f'STFT constraint violated: n_fft({n_fft}) must be <= L({L}), shape={shape}')
    if win_length > n_fft:
        raise ValueError(f'STFT constraint violated: win_length({win_length}) must be <= n_fft({n_fft})')
    if hop_length <= 0 or win_length <= 0 or n_fft <= 0:
        raise ValueError(f'STFT requires n_fft, hop_length, win_length > 0, got n_fft={n_fft}, hop_length={hop_length}, win_length={win_length}')
    x = torch.randn(shape, device=device, dtype=dtype)
    window = torch.ones(win_length, device=device, dtype=dtype)
    return (x, window, n_fft, hop_length, win_length, normalized, onesided, return_complex)

def get_init_inputs_per_case(param, device=None):
    return []

def _to_float32(t):
    """统一转为 float32 或 complex64，避免 Half/Double 混用导致 'expected scalar type Double but found Half'。"""
    if isinstance(t, (list, tuple)):
        return type(t)((_to_float32(x) for x in t))
    if not isinstance(t, torch.Tensor):
        return t
    if t.is_complex():
        return t.to(torch.complex64)
    return t.float()


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
    json_path = os.path.join(os.path.dirname(__file__), "Stft.json")
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
