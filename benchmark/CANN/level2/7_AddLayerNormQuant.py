"""Auto-generated benchmark file for AddLayerNormQuant.

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

def _calc_output(output_y: torch.Tensor) -> tuple:
    """Golden formula: out_scales = max(|y|, dim=-1, keepdim)/127, y_int8 = round(y/out_scales)."""
    max_y_0 = torch.max(torch.abs(output_y), dim=-1, keepdim=True)[0]
    out_scales = max_y_0 / 127.0
    x_quant = output_y / (out_scales + 1e-12)
    y_int8 = torch.round(x_quant).clamp(-128, 127).to(torch.int8)
    return (y_int8, out_scales.squeeze(-1))

class Model(nn.Module):
    """Reference aligned with TBE golden: add -> layernorm -> quant(y*scale) with out_scale = max(|z|)/127."""

    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, bias: Optional[torch.Tensor]=None, scales1: Optional[torch.Tensor]=None, scales2: Optional[torch.Tensor]=None, zero_points1: Optional[torch.Tensor]=None, zero_points2: Optional[torch.Tensor]=None, epsilon: float=1e-05, additional_output: bool=True, div_mode: bool=True):
        super(Model, self).__init__()
        self.gamma = gamma.to(torch.float32).to('cpu')
        self.beta = beta.to(torch.float32).to('cpu')
        self.bias = bias.to(torch.float32).to('cpu') if bias is not None else None
        self.scales1 = scales1
        self.scales2 = scales2
        self.zero_points1 = zero_points1
        self.zero_points2 = zero_points2
        self.epsilon = epsilon
        self.additional_output = additional_output
        self.div_mode = div_mode

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        dtype_hp = torch.float32
        x1 = x1.to(dtype_hp)
        x2 = x2.to(dtype_hp)
        gamma = self.gamma.to(x1.device).to(dtype_hp)
        beta = self.beta.to(x1.device).to(dtype_hp)
        x = x1 + x2
        if self.bias is not None:
            x = x + self.bias.to(x.device).to(dtype_hp)
        mean = x.mean(dim=-1, keepdim=True)
        var = torch.mean(torch.pow(x - mean, 2), dim=-1, keepdim=True)
        rstd = 1.0 / torch.sqrt(var + self.epsilon)
        y = (x - mean) * rstd * gamma + beta
        y = y.to(torch.float32)
        if self.scales1 is not None and self.scales2 is not None:
            s1 = self.scales1.to(y.device).to(torch.float32)
            s2 = self.scales2.to(y.device).to(torch.float32)
            output_y1 = y * s1
            output_y2 = y * s2
            y1_int, out_scale1 = _calc_output(output_y1)
            y2_int, out_scale2 = _calc_output(output_y2)
        else:
            y1_int, out_scale1 = _calc_output(y)
            y2_int = torch.zeros_like(y, dtype=torch.int8)
            out_scale2 = out_scale1
        x_out = x.to(x1.dtype) if x1.dtype != torch.float32 else x
        return [y1_int, y2_int, x_out, out_scale1, out_scale2]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    shape = eval(param.get('input_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    x1 = torch.rand(shape, device=device, dtype=dtype) * 2 - 1
    x2 = torch.rand(shape, device=device, dtype=dtype) * 2 - 1
    return (x1, x2)

def get_init_inputs_per_case(param, device=None):
    shape = eval(param.get('normalized_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    gamma = torch.rand(shape, device=device, dtype=dtype)
    beta = torch.rand(shape, device=device, dtype=dtype)
    quant_mode = param.get('quant_mode', 'dynamic')
    if quant_mode == 'dynamic':
        bias = None
        scales1 = None
        scales2 = None
        zero_points1 = None
        zero_points2 = None
    else:
        bias = torch.rand(shape, device=device, dtype=dtype)
        scales1 = torch.rand(shape, device=device, dtype=torch.float32) * 0.01 + 0.001
        scales2 = torch.rand(shape, device=device, dtype=torch.float32) * 0.01 + 0.001
        zero_points1 = torch.zeros(shape, device=device, dtype=torch.float32)
        zero_points2 = torch.zeros(shape, device=device, dtype=torch.float32)
    epsilon = float(param.get('epsilon', 1e-05))
    additional_output = param.get('additional_output', True)
    div_mode = param.get('div_mode', True)
    return [gamma, beta, bias, scales1, scales2, zero_points1, zero_points2, epsilon, additional_output, div_mode]


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
    json_path = os.path.join(os.path.dirname(__file__), "AddLayerNormQuant.json")
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
