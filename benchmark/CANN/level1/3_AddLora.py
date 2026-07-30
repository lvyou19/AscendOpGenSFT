"""Auto-generated benchmark file for AddLora.

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

def _reference_add_lora(y, x, weight_b, weight_a, indices, layer_idx, scale, y_offset, y_slice_size):
    out = y.clone()
    bsz = x.shape[0]
    for b in range(bsz):
        w = int(indices[b].item())
        wa = weight_a[w, layer_idx]
        wb = weight_b[w, layer_idx]
        z1 = (wa @ x[b].unsqueeze(-1)).squeeze(-1)
        z2 = (wb @ z1.unsqueeze(-1)).squeeze(-1) * scale
        sl = slice(y_offset, y_offset + y_slice_size)
        out[b, sl] = out[b, sl] + z2
    return out

class Model(nn.Module):

    def __init__(self, layer_idx: int, scale: float, y_offset: int, y_slice_size: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.scale = float(scale)
        self.y_offset = int(y_offset)
        self.y_slice_size = int(y_slice_size)

    def forward(self, y, x, weight_b, indices, weight_a):
        return _reference_add_lora(y, x, weight_b, weight_a, indices, self.layer_idx, self.scale, self.y_offset, self.y_slice_size)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    b = int(param['B'])
    h1 = int(param['H1'])
    h3 = int(param['H3'])
    w = int(param['W'])
    ll = int(param['L'])
    h2 = int(param['H2'])
    r = int(param['R'])
    dtype = torch.float16
    y = torch.randn(b, h3, device=device, dtype=dtype)
    x = torch.randn(b, h1, device=device, dtype=dtype)
    weight_b = torch.randn(w, ll, h2, r, device=device, dtype=dtype)
    weight_a = torch.randn(w, ll, r, h1, device=device, dtype=dtype)
    indices = torch.randint(0, w, (b,), device=device, dtype=torch.int32)
    return (y, x, weight_b, indices, weight_a)

def get_init_inputs_per_case(param, device=None):
    return (int(param['layer_idx']), float(param['scale']), int(param['y_offset']), int(param['y_slice_size']))


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
    json_path = os.path.join(os.path.dirname(__file__), "AddLora.json")
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
