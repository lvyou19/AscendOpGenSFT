"""Auto-generated benchmark file for GroupNormGrad.

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
from typing import List
import torch
import torch.nn as nn

def _group_norm_grad_ref(dy, x, mean, rstd, gamma, num_groups):
    """PyTorch reference: GroupNorm backward. Returns dx, dgamma, dbeta as list."""
    dtype_orig = x.dtype
    dy_hp = dy.to(torch.float32)
    mean_hp = mean.to(torch.float32)
    rstd_hp = rstd.to(torch.float32)
    x_hp = x.to(torch.float32)
    gamma_hp = gamma.to(torch.float32)
    batch_num, num_channels = (x_hp.size(0), x_hp.size(1))
    remaining = x_hp.size()[2:]
    hw = 1
    for s in remaining:
        hw *= s
    num_per_group_channel = num_channels // num_groups
    num_per_group_total = float(num_per_group_channel * hw)
    x_reshaped = x_hp.reshape((batch_num, num_channels, hw))
    dy_reshaped = dy_hp.reshape((batch_num, num_channels, hw))
    dgamma_sum = torch.zeros_like(gamma_hp)
    dbeta_sum = torch.zeros_like(gamma_hp)
    dx_out = torch.zeros_like(x_reshaped)
    for n_i in range(batch_num):
        for g_i in range(num_groups):
            ch_start = g_i * num_per_group_channel
            ch_end = (g_i + 1) * num_per_group_channel
            x_g = x_reshaped[n_i, ch_start:ch_end, :]
            dy_g = dy_reshaped[n_i, ch_start:ch_end, :]
            mean_x = mean_hp[n_i, g_i]
            rstd_x = rstd_hp[n_i, g_i]
            x_norm = (x_g - mean_x) * rstd_x
            gamma_g = gamma_hp[ch_start:ch_end].view(num_per_group_channel, 1)
            temp_1 = torch.sum(dy_g, dim=1)
            temp_2 = torch.sum(dy_g * x_norm, dim=1)
            dbeta_sum[ch_start:ch_end] += temp_1
            dgamma_sum[ch_start:ch_end] += temp_2
            c1 = torch.sum(temp_1 * gamma_g.squeeze(1)) / num_per_group_total
            c2 = torch.sum(temp_2 * gamma_g.squeeze(1)) / num_per_group_total
            dx_g = rstd_x * (dy_g * gamma_g - x_norm * c2 - c1)
            dx_out[n_i, ch_start:ch_end, :] = dx_g
    dx_out = dx_out.reshape(x_hp.shape)
    return [dx_out.to(dtype_orig), dgamma_sum.to(dtype_orig), dbeta_sum.to(dtype_orig)]

class Model(nn.Module):

    def __init__(self, num_groups: int, data_format: str):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.data_format = data_format

    def forward(self, dy: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, x: torch.Tensor, gamma: torch.Tensor, dx_is_require: bool, dgamma_is_require: bool, dbeta_is_require: bool) -> List[torch.Tensor]:
        dx, dgamma, dbeta = _group_norm_grad_ref(dy, x, mean, rstd, gamma, self.num_groups)
        out = []
        if dx_is_require:
            out.append(dx)
        if dgamma_is_require:
            out.append(dgamma)
        if dbeta_is_require:
            out.append(dbeta)
        return out

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np

def get_inputs(param, device=None):
    """
    Generate input tensors for the GroupNormGrad operator.
    Inputs: dy, mean, rstd, x, gamma; attrs: num_groups, data_format, dx_is_require, dgamma_is_require, dbeta_is_require.
    """
    x_shape = eval(param.get('x_shape', '[2, 32, 4, 4]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    num_channels = x_shape[1]
    num_groups = int(param.get('num_groups', 8))
    dy = (torch.rand(x_shape, device=device, dtype=dtype) * 2.0 - 1.0) * 0.1
    mean_rstd_shape = (x_shape[0], num_groups)
    mean = torch.rand(mean_rstd_shape, device=device, dtype=dtype) * 0.1
    rstd = torch.rand(mean_rstd_shape, device=device, dtype=dtype) * 0.1 + 0.001
    x = (torch.rand(x_shape, device=device, dtype=dtype) * 2.0 - 1.0) * 0.1
    gamma = torch.rand(num_channels, device=device, dtype=dtype) * 0.1 + 0.5
    dx_is_require = bool(int(param.get('dx_is_require', 1)))
    dgamma_is_require = bool(int(param.get('dgamma_is_require', 1)))
    dbeta_is_require = bool(int(param.get('dbeta_is_require', 1)))
    return (dy, mean, rstd, x, gamma, dx_is_require, dgamma_is_require, dbeta_is_require)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the GroupNormGrad model from DataFrame row.
    """
    num_groups = int(param.get('num_groups', 8))
    data_format = param.get('data_format', 'NCHW')
    return [num_groups, data_format]


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
    json_path = os.path.join(os.path.dirname(__file__), "GroupNormGrad.json")
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
