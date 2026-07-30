"""Auto-generated benchmark file for AddRmsNormDynamicQuantV2.

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
from typing import List, Optional
import torch
import torch.nn as nn

def golden_add_rms_norm(x1: torch.Tensor, x2: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> tuple:
    """与 ATK goldenAddRmsNorm 一致: x=x1+x2, rstd=rsqrt(mean(x^2)+eps), y=x*rstd*gamma+beta, return (y, x)."""
    ori_dtype = x1.dtype
    if ori_dtype != torch.float32:
        x1 = x1.float()
        x2 = x2.float()
        gamma = gamma.float()
        beta = beta.float()
    x = x1 + x2
    rstd = torch.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + eps)
    y = x * rstd * gamma + beta
    if ori_dtype != torch.float32:
        return (y, x.to(ori_dtype))
    return (y, x)

def golden_dynamic_quant(x: torch.Tensor, smooth: Optional[torch.Tensor]) -> tuple:
    """与 ATK goldenDynamicQuant 一致: smooth_x = x 或 x*smooth; gs_rev=127/max(|smooth_x|), gs=1/gs_rev, gq=round(smooth_x*gs_rev).int8."""
    x = x.float() if x.dtype != torch.float32 else x
    if smooth is not None:
        smooth = smooth.float() if smooth.dtype != torch.float32 else smooth
    else:
        smooth = None
    smooth_x = x if smooth is None else x * smooth
    x_max = torch.max(torch.abs(smooth_x), dim=-1, keepdim=True)[0].clamp(min=1e-08)
    gs_rev = 127.0 / x_max
    gs = 1.0 / gs_rev
    sx = smooth_x * gs_rev
    gq = torch.round(sx).to(torch.int8)
    return (gq, gs.squeeze(-1).float())

class Model(nn.Module):
    """Reference: add + RMSNorm(+beta) + dynamic quant，与 ATK FunctionRmsNormGradApi 标杆一致。"""

    def __init__(self, gamma: torch.Tensor, epsilon: float=1e-06, beta: Optional[torch.Tensor]=None):
        super(Model, self).__init__()
        self.gamma = gamma.detach().to(torch.float32)
        self.epsilon = epsilon
        self.beta = beta.detach().to(torch.float32) if beta is not None else None

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, smooth1: Optional[torch.Tensor]=None, smooth2: Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        device = x1.device
        dtype = x1.dtype
        gamma = self.gamma.to(device).to(dtype)
        beta = self.beta.to(device).to(dtype) if self.beta is not None else torch.zeros(gamma.shape, device=device, dtype=dtype)
        gy_fp32, gx = golden_add_rms_norm(x1, x2, gamma, beta, self.epsilon)
        smooth1_exist = smooth1 is not None
        smooth2_exist = smooth2 is not None
        if smooth1_exist and smooth2_exist:
            gq1, gs1 = golden_dynamic_quant(gy_fp32, smooth1)
            gq2, gs2 = golden_dynamic_quant(gy_fp32, smooth2)
        elif smooth1_exist and (not smooth2_exist):
            gq1, gs1 = golden_dynamic_quant(gy_fp32, smooth1)
            gq2 = torch.zeros_like(gq1, device=device, dtype=gq1.dtype)
            gs2 = torch.zeros_like(gs1, device=device, dtype=gs1.dtype)
        elif not smooth1_exist and (not smooth2_exist):
            gq1, gs1 = golden_dynamic_quant(gy_fp32, None)
            gq2 = torch.zeros_like(gq1, device=device, dtype=gq1.dtype)
            gs2 = torch.zeros_like(gs1, device=device, dtype=gs1.dtype)
        else:
            gq1, gs1 = golden_dynamic_quant(gy_fp32, None)
            gq2, gs2 = golden_dynamic_quant(gy_fp32, smooth2)
        y3 = gy_fp32
        y4 = gy_fp32.to(dtype) if dtype != torch.float32 else gy_fp32
        x_out = gx
        N = gq1.shape[0]
        scale1_out = gs1 if gs1.dim() >= 1 else gs1.unsqueeze(0).expand(N)
        scale2_out = gs2 if gs2.dim() >= 1 else gs2.unsqueeze(0).expand(N)
        return [gq1, gq2, y3, y4, x_out, scale1_out, scale2_out]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    shape = eval(param.get('input_shape', '[2, 4]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    x1 = torch.rand(shape, device=device, dtype=dtype) * 2 - 1
    x2 = torch.rand(shape, device=device, dtype=dtype) * 2 - 1
    return (x1, x2)

def get_init_inputs_per_case(param, device=None):
    shape = eval(param.get('normalized_shape', '[4]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    gamma = torch.rand(shape, device=device, dtype=dtype)
    epsilon = float(param.get('epsilon', 1e-06))
    return [gamma, epsilon]


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
    json_path = os.path.join(os.path.dirname(__file__), "AddRmsNormDynamicQuantV2.json")
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
