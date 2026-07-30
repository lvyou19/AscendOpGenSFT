"""Auto-generated benchmark file for DeepNormGrad.

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

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dy: torch.Tensor, x: torch.Tensor, gx: torch.Tensor, gamma: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, alpha: float) -> List[torch.Tensor]:
        dy_fp32 = dy.to(torch.float32)
        x_fp32 = x.to(torch.float32)
        gx_fp32 = gx.to(torch.float32)
        gamma_fp32 = gamma.to(torch.float32)
        mean_fp32 = mean.to(torch.float32)
        rstd_fp32 = rstd.to(torch.float32)
        D = float(torch.prod(torch.tensor(gamma_fp32.shape)))
        tmpone = dy_fp32 * gamma_fp32
        tmptwo = alpha * x_fp32 + gx_fp32 - mean_fp32
        reduction_dims = tuple(range(x_fp32.dim() - gamma_fp32.dim(), x_fp32.dim()))
        d_var = torch.sum(-0.5 * tmpone * tmptwo * rstd_fp32.pow(3), dim=reduction_dims, keepdim=True)
        d_mean = torch.sum(-1.0 * tmpone * rstd_fp32, dim=reduction_dims, keepdim=True)
        dgx = tmpone * rstd_fp32 + 2.0 / D * d_var * tmptwo + 1.0 / D * d_mean
        dx = alpha * dgx
        d_reduction_dims_for_gamma_beta = tuple(range(dy_fp32.dim() - gamma_fp32.dim()))
        dbeta = torch.sum(dy_fp32, dim=d_reduction_dims_for_gamma_beta, keepdim=False)
        dgamma = torch.sum(dy_fp32 * rstd_fp32 * tmptwo, dim=d_reduction_dims_for_gamma_beta, keepdim=False)
        dx = dx.to(x.dtype)
        dgx = dgx.to(gx.dtype)
        return [dx, dgx, dbeta, dgamma]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成模型的输入张量列表和标量。
    """
    input_shape = eval(param.get('input_shape', '[1]'))
    normalized_shape = eval(param.get('normalized_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    alpha = param.get('alpha', 0.3)
    epsilon = param.get('epsilon', 1e-06)
    dy = torch.rand(input_shape, device=device, dtype=dtype)
    x = torch.rand(input_shape, device=device, dtype=dtype)
    gx = torch.rand(input_shape, device=device, dtype=dtype)
    gamma = torch.rand(normalized_shape, device=device, dtype=dtype)
    x_add = x.to(torch.float32) * alpha + gx.to(torch.float32)
    reduction_dims = tuple(range(x_add.dim() - len(normalized_shape), x_add.dim()))
    mean = x_add.mean(dim=reduction_dims, keepdim=True)
    variance = (x_add - mean).pow(2).mean(dim=reduction_dims, keepdim=True)
    rstd = torch.rsqrt(variance + epsilon)
    mean_rstd_shape = list(input_shape)
    for i in range(len(normalized_shape)):
        mean_rstd_shape[len(input_shape) - 1 - i] = 1
    mean = mean.view(mean_rstd_shape)
    rstd = rstd.view(mean_rstd_shape)
    return (dy, x, gx, gamma, mean, rstd, alpha)

def get_init_inputs_per_case(param, device=None):
    """
    DeepNormGrad Model does not have initialization parameters.
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
    json_path = os.path.join(os.path.dirname(__file__), "DeepNormGrad.json")
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
