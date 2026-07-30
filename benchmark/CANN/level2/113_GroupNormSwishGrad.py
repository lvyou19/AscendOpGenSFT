"""Auto-generated benchmark file for GroupNormSwishGrad.

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

    def __init__(self, num_groups: int, swish_scale: float):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.swish_scale = swish_scale

    def forward(self, dy: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, dgamma_is_require: bool, dbeta_is_require: bool) -> torch.Tensor:
        dtype_orig = x.dtype
        dy_hp = dy.to(torch.float32)
        mean_hp = mean.to(torch.float32)
        rstd_hp = rstd.to(torch.float32)
        x_hp = x.to(torch.float32)
        gamma_hp = gamma.to(torch.float32)
        beta_hp = beta.to(torch.float32)
        batch_num = x_hp.size(0)
        num_channels = x_hp.size(1)
        remaining_dims = x_hp.size()[2:]
        hw = 1
        for size in remaining_dims:
            hw *= size
        num_per_group_channel = num_channels // self.num_groups
        num_per_group_total = float(num_per_group_channel * hw)
        x_reshaped = x_hp.reshape((batch_num, num_channels, hw))
        dy_reshaped = dy_hp.reshape((batch_num, num_channels, hw))
        dL_dgamma_sum = torch.zeros_like(gamma_hp)
        dL_dbeta_sum = torch.zeros_like(beta_hp)
        dL_dx_out = torch.zeros_like(x_reshaped)
        for n_i in range(batch_num):
            for g_i in range(self.num_groups):
                ch_start = g_i * num_per_group_channel
                ch_end = (g_i + 1) * num_per_group_channel
                x_group_slice = x_reshaped[n_i, ch_start:ch_end, :]
                dy_group_slice = dy_reshaped[n_i, ch_start:ch_end, :]
                mean_x = mean_hp[n_i, g_i]
                rstd_x = rstd_hp[n_i, g_i]
                x_norm_i = (x_group_slice - mean_x) * rstd_x
                gamma_group_slice = gamma_hp[ch_start:ch_end].view(num_per_group_channel, 1)
                beta_group_slice = beta_hp[ch_start:ch_end].view(num_per_group_channel, 1)
                gn_output_group = x_norm_i * gamma_group_slice + beta_group_slice
                dswish_dz_intermediate = gn_output_group * -self.swish_scale
                dswish_dz_intermediate = torch.exp(dswish_dz_intermediate)
                dswish_dz_intermediate = dswish_dz_intermediate + 1.0
                tmp_res_val = gn_output_group / dswish_dz_intermediate
                tmp_res_val = gn_output_group - tmp_res_val
                tmp_res_val = tmp_res_val + 1.0
                dswish_dz = tmp_res_val / dswish_dz_intermediate
                d_gn_output = dswish_dz * dy_group_slice
                temp_1 = torch.sum(d_gn_output, dim=1)
                temp_2 = torch.sum(d_gn_output * x_norm_i, dim=1)
                dL_dbeta_sum[ch_start:ch_end] += temp_1
                dL_dgamma_sum[ch_start:ch_end] += temp_2
                c1 = torch.sum(temp_1 * gamma_group_slice.squeeze(1)) / num_per_group_total
                c2 = torch.sum(temp_2 * gamma_group_slice.squeeze(1)) / num_per_group_total
                dL_dx_G_C = torch.zeros_like(x_group_slice)
                for i in range(num_per_group_channel):
                    dL_dx_G_C[i] = rstd_x * (d_gn_output[i] * gamma_group_slice[i] - x_norm_i[i] * c2 - c1)
                dL_dx_out[n_i, ch_start:ch_end, :] = dL_dx_G_C
        dL_dx_out = dL_dx_out.reshape(x_hp.shape)
        dx_result = dL_dx_out.to(dtype_orig)
        return dx_result

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np

def get_inputs(param, device=None):
    """
    Generate input tensors for the GroupNormSwishGrad operator's forward method.
    """
    x_shape = eval(param.get('x_shape', '[2, 32, 4, 4]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    num_channels = x_shape[1]
    num_groups = param.get('num_groups', 8)
    dy = (torch.rand(x_shape, device=device, dtype=dtype) * 2.0 - 1.0) * 0.1
    mean_rstd_shape = (x_shape[0], num_groups)
    mean = torch.rand(mean_rstd_shape, device=device, dtype=dtype) * 0.1
    rstd = torch.rand(mean_rstd_shape, device=device, dtype=dtype) * 0.1 + 0.001
    x = (torch.rand(x_shape, device=device, dtype=dtype) * 2.0 - 1.0) * 0.1
    gamma = torch.rand(num_channels, device=device, dtype=dtype) * 0.1 + 0.5
    beta = torch.rand(num_channels, device=device, dtype=dtype) * 0.1
    dgamma_is_require = bool(int(param.get('dgamma_is_require', True)))
    dbeta_is_require = bool(int(param.get('dbeta_is_require', True)))
    return (dy, mean, rstd, x, gamma, beta, dgamma_is_require, dbeta_is_require)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the GroupNormSwishGrad model from DataFrame row.
    """
    num_groups = param.get('num_groups', 8)
    swish_scale = float(param.get('swish_scale', 1.0))
    return [num_groups, swish_scale]


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
    json_path = os.path.join(os.path.dirname(__file__), "GroupNormSwishGrad.json")
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
