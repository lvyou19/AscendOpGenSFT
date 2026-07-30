"""Auto-generated benchmark file for AddRmsNormDynamicQuant.

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

def _dynamic_quant_per_row(y: torch.Tensor, smooth: Optional[torch.Tensor]=None, dim: int=-1) -> tuple:
    """Per-row dynamic quant. If smooth is not None: quantize y*smooth (smooth broadcast to y); else quantize y.
    scale = max(abs(smooth_x))/127, q = round(smooth_x/scale). Aligned with ATK goldenDynamicQuant."""
    y_flat = y.float()
    if smooth is not None:
        smooth_f = smooth.float().to(y_flat.device)
        smooth_x = y_flat * smooth_f
    else:
        smooth_x = y_flat
    scale = smooth_x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-08) / 127.0
    q = (smooth_x / scale).round().clamp(-128, 127).to(torch.int8)
    return (q, scale.squeeze(dim).float())

def _golden_add_rms_norm(x1: torch.Tensor, x2: torch.Tensor, gamma: torch.Tensor, eps: float):
    """x = x1+x2, rstd = rsqrt(mean(x^2,-1)+eps), y = x * rstd * gamma. Aligned with ATK goldenAddRmsNorm (no beta)."""
    ori_dtype = x1.dtype
    x1_f = x1.float()
    x2_f = x2.float()
    gamma_f = gamma.float().to(x1_f.device)
    x = x1_f + x2_f
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = x * rstd * gamma_f
    x_out = x if ori_dtype == torch.float32 else x.to(ori_dtype)
    return (y, x_out)

class Model(nn.Module):
    """Reference: add + RMSNorm + dynamic quant. Supports optional smooth1/smooth2. Outputs: y1, y2, x, scale1, scale2.
    Aligned with ATK goldenAddRmsNorm + goldenDynamicQuant."""

    def __init__(self, gamma: torch.Tensor, epsilon: float=1e-06, smooth_scale1: Optional[torch.Tensor]=None, smooth_scale2: Optional[torch.Tensor]=None):
        super(Model, self).__init__()
        self.gamma = gamma.detach().to(torch.float32)
        self.epsilon = epsilon
        self.smooth_scale1 = smooth_scale1
        self.smooth_scale2 = smooth_scale2

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        gamma = self.gamma.to(x1.device).to(x1.dtype)
        y, x_out = _golden_add_rms_norm(x1, x2, gamma, self.epsilon)
        smooth1 = self.smooth_scale1.to(y.device).to(y.dtype) if self.smooth_scale1 is not None else None
        smooth2 = self.smooth_scale2.to(y.device).to(y.dtype) if self.smooth_scale2 is not None else None
        if smooth1 is not None and smooth2 is not None:
            y1, scale1 = _dynamic_quant_per_row(y, smooth1, dim=-1)
            y2, scale2 = _dynamic_quant_per_row(y, smooth2, dim=-1)
        elif smooth1 is not None:
            y1, scale1 = _dynamic_quant_per_row(y, smooth1, dim=-1)
            y2 = torch.zeros_like(y1, device=y.device, dtype=torch.int8)
            scale2 = torch.zeros_like(scale1, device=y.device, dtype=torch.float32)
        else:
            y1, scale1 = _dynamic_quant_per_row(y, None, dim=-1)
            y2 = torch.zeros_like(y1, device=y.device, dtype=torch.int8)
            scale2 = torch.zeros_like(scale1, device=y.device, dtype=torch.float32)
        scale1_out = scale1 if scale1.dim() >= 1 else scale1.unsqueeze(0)
        return [y1, y2, x_out, scale1_out, scale2]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """x1, x2: same dtype and shape (2~8 dims). Constraint aligned with ATK generate_add_rms_norm_dynamic_quant."""
    shape = eval(param.get('input_shape', '[2, 4]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    x1 = torch.rand(shape, device=device, dtype=dtype) * 2 - 1
    x2 = torch.rand(shape, device=device, dtype=dtype) * 2 - 1
    return (x1, x2)

def get_init_inputs_per_case(param, device=None):
    """gamma.shape = [x1.shape[-1]]; smooth1/smooth2 when use_smooth: shape = gamma.shape, dtype = gamma.dtype."""
    input_shape = eval(param.get('input_shape', '[2, 4]'))
    normalized_shape = eval(param.get('normalized_shape', '[4]'))
    if isinstance(normalized_shape, list) and len(normalized_shape) >= 1 and (len(input_shape) >= 1):
        normalized_shape = [input_shape[-1]]
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    gamma = torch.rand(normalized_shape, device=device, dtype=dtype)
    epsilon = float(param.get('epsilon', 1e-06))
    use_smooth = int(param.get('use_smooth', 0)) != 0
    if use_smooth:
        smooth_scale1 = torch.rand(normalized_shape, device=device, dtype=dtype)
        smooth_scale2 = torch.rand(normalized_shape, device=device, dtype=dtype)
        return [gamma, epsilon, smooth_scale1, smooth_scale2]
    return [gamma, epsilon, None, None]


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
    json_path = os.path.join(os.path.dirname(__file__), "AddRmsNormDynamicQuant.json")
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
