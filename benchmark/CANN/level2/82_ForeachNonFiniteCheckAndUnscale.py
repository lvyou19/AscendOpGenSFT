"""Auto-generated benchmark file for ForeachNonFiniteCheckAndUnscale.

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

    def __init__(self):
        super().__init__()

    def forward(self, scaled_grads: List[torch.Tensor], found_inf_tensor: torch.Tensor, in_scale_tensor: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        scale_value = in_scale_tensor.item()
        local_found_inf = torch.tensor(0.0, dtype=torch.float, device=found_inf_tensor.device)
        for grad in scaled_grads:
            non_finite = torch.isnan(grad) | torch.isinf(grad)
            if non_finite.any():
                local_found_inf.fill_(1.0)
            if scale_value == 0.0:
                grad.fill_(0.0)
            else:
                grad.mul_(scale_value)
        found_inf_tensor.copy_(local_found_inf)
        return [torch.concat([x.flatten() for x in scaled_grads]), found_inf_tensor]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np
from typing import List, Tuple

def get_inputs(param, device=None) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    返回：
        scaled_grads : List[Tensor]  ← 一层列表
        found_inf    : Tensor(0/1, float)
        in_scale     : Tensor(scale, float)
    """
    input_shapes = eval(param.get('input_shape', '[[8, 2048]]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    scale_value = float(param.get('scale_value', 1.0))
    add_non_finite = bool(param.get('add_non_finite', False))
    scaled_grads = []
    for shape in input_shapes:
        g = torch.rand(shape, device=device, dtype=dtype) * 0.01
        if add_non_finite and g.numel():
            g.view(-1)[0] = float('nan') if np.random.rand() > 0.5 else float('inf')
        scaled_grads.append(g)
    found_inf = torch.tensor(0.0, dtype=torch.float, device=device)
    in_scale = torch.tensor(scale_value, dtype=torch.float, device=device)
    return (scaled_grads, found_inf, in_scale)

def get_init_inputs_per_case(param, device=None) -> List:
    return []


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
    json_path = os.path.join(os.path.dirname(__file__), "ForeachNonFiniteCheckAndUnscale.json")
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
