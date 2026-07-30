"""Auto-generated benchmark file for GroupNormSwish.

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
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, num_channels: int, num_groups: int, eps: float, activate_swish: bool, swish_scale: float):
        super(Model, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.activate_swish = activate_swish
        self.swish_scale = swish_scale

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> List[torch.Tensor]:
        input_dtype = x.dtype
        N = x.shape[0]
        C = self.num_channels
        remaining_dims_product = 1
        for size in x.shape[2:]:
            remaining_dims_product *= size
        HxW = remaining_dims_product
        x_fp32 = x.to(torch.float32)
        gamma_fp32 = gamma.to(torch.float32)
        beta_fp32 = beta.to(torch.float32)
        gn_out, mean_out, rstd_out = torch.ops.aten.native_group_norm(x_fp32, gamma_fp32, beta_fp32, N, C, HxW, self.num_groups, self.eps)
        if self.activate_swish:
            sigmoid_arg = self.swish_scale * gn_out
            swish_out = gn_out * torch.sigmoid(sigmoid_arg)
            final_out = swish_out
        else:
            final_out = gn_out
        return [final_out.to(input_dtype)]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np

def get_inputs(param, device=None):
    """
    Generate input tensors for the GroupNormSwish operator's forward method.
    """
    x_shape = eval(param.get('x_shape', '[100, 32]'))
    gamma_shape = eval(param.get('gamma_shape', '[32]'))
    beta_shape = eval(param.get('beta_shape', '[32]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    x = torch.rand(x_shape, device=device, dtype=dtype) * 2.0 - 1.0
    gamma = torch.rand(gamma_shape, device=device, dtype=dtype)
    beta = torch.rand(beta_shape, device=device, dtype=dtype)
    return (x, gamma, beta)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the GroupNormSwish model from DataFrame row.
    """
    num_channels = eval(param.get('x_shape', '[100, 32]'))[1]
    num_groups = param.get('num_groups', 8)
    eps = float(param.get('eps', 1e-05))
    activate_swish = bool(param.get('activate_swish', False))
    swish_scale = float(param.get('swish_scale', 1.0))
    return [num_channels, num_groups, eps, activate_swish, swish_scale]


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
    json_path = os.path.join(os.path.dirname(__file__), "GroupNormSwish.json")
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
