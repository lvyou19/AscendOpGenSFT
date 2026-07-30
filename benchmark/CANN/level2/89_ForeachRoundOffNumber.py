"""Auto-generated benchmark file for ForeachRoundOffNumber.

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

def _flatten(lst):
    flat = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            flat.extend(_flatten(item))
        elif isinstance(item, torch.Tensor):
            flat.append(item)
        else:
            raise TypeError('Unexpected element type')
    return flat

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: List[torch.Tensor], round_mode: torch.Tensor) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        round_mode = round_mode.item()
        for x_tensor in x:
            if round_mode == 1:
                result_tensor = torch.round(x_tensor)
            elif round_mode == 2:
                result_tensor = torch.floor(x_tensor)
            elif round_mode == 3:
                result_tensor = torch.ceil(x_tensor)
            elif round_mode == 4:
                result_tensor = torch.where(x_tensor >= 0, (x_tensor + 0.5).floor(), (x_tensor - 0.5).ceil())
            elif round_mode == 5:
                result_tensor = torch.trunc(x_tensor)
            elif round_mode == 6:
                int_part = x_tensor.trunc()
                frac_part_abs = (x_tensor - int_part).abs()
                result_tensor = torch.where(frac_part_abs == 0.5, torch.where(int_part % 2 == 0, int_part + torch.sign(x_tensor), int_part), torch.round(x_tensor))
            elif round_mode == 7:
                result_tensor = torch.frac(x_tensor)
            else:
                result_tensor = x_tensor
            results.append(result_tensor.to(x_tensor.dtype))
        return results

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import random

def get_inputs(param, device=None):
    """
    Generate input tensor list for the ForeachRoundOffNumber operator.
    """
    num_tensors = param.get('num_tensors', 1)
    input_shapes = eval(param.get('input_shapes', '[[10]]'))
    dtype_str = param.get('dtype', 'float32')
    round_mode = int(param.get('round_mode', 1))
    torch_dtype = getattr(torch, dtype_str)
    scalar_dtype = torch.float32 if torch_dtype == torch.bfloat16 else torch_dtype
    round_mode_tensor = torch.tensor(round_mode, dtype=scalar_dtype, device=device)
    x_tensors: List[torch.Tensor] = []
    for i in range(num_tensors):
        shape = input_shapes[i % len(input_shapes)]
        torch_tensor = torch.empty(shape, dtype=torch_dtype, device=device).uniform_(-10.5, 10.5)
        if torch_tensor.numel() > 0:
            torch_tensor.view(-1)[torch.randint(0, torch_tensor.numel(), (1,))] = random.randint(-5, 5) + 0.5
        if torch_tensor.numel() > 1:
            torch_tensor.view(-1)[torch.randint(0, torch_tensor.numel(), (1,))] = random.randint(-5, 5) - 0.5
        if torch_tensor.numel() > 2:
            torch_tensor.view(-1)[torch.randint(0, torch_tensor.numel(), (1,))] = random.randint(-5, 5)
        x_tensors.append(torch_tensor)
    return (x_tensors, round_mode_tensor)

def get_init_inputs_per_case(param, device=None):
    """

    """
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
    json_path = os.path.join(os.path.dirname(__file__), "ForeachRoundOffNumber.json")
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
