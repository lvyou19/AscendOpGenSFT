"""Auto-generated benchmark file for RmsNormQuant.

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

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float=1e-06) -> torch.Tensor:
    """与 ATK npu_add_rms_norm_quant_golden 一致: factor = 1/sqrt(mean(x^2)+eps), return x * factor * weight + bias."""
    square_sum = torch.sum(torch.square(x), dim=-1, keepdim=True)
    factor = 1.0 / torch.sqrt(square_sum / x.shape[-1] + eps)
    return x * factor * weight + bias

class Model(nn.Module):
    """
    标杆与 ATK function_rms_norm_quant 完全一致:
    quant_in = rms_norm(x, gamma, beta, eps=epsilon);
    y = quantize_per_tensor(quant_in, scale, offset, qint8).int_repr()，scale/offset 取标量。
    误差来源说明：当某行 mean(x^2) 很小时，rms≈sqrt(epsilon)。若 kernel 侧 epsilon 与标杆不一致
    （如未注入 attr 时用默认 1e-12），kernel 的 factor 会远大于标杆，导致该行 quant_in 被放大，
    出现 72/127 等异常值，从而产生 max_abs_error=130 量级的误差。
    """

    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, epsilon: float):
        super(Model, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.scale = scale
        self.offset = offset
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        input_x = x.float()
        input_gamma = self.gamma.float()
        input_beta = self.beta.float()
        input_scale = self.scale.float().flatten()[0].item()
        input_offset = self.offset.flatten()[0].item()
        quant_in = _rms_norm(input_x, weight=input_gamma, bias=input_beta, eps=self.epsilon)
        output_q = torch.quantize_per_tensor(quant_in, input_scale, input_offset, torch.qint8)
        y_np = output_q.int_repr().detach().clone().cpu()
        out_int8 = y_np.to(torch.int8).reshape(x.shape)
        return [out_int8]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import os
import torch

def get_inputs(param, device=None):
    shape = eval(param.get('input_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    x = torch.rand(shape, device=device, dtype=dtype) * 2 - 1
    return (x,)

def get_init_inputs_per_case(param, device=None):
    """与 ATK generate_rms_norm_quant 约束一致：scale/offset 为 [1]，gamma/beta 为 [H] 或 [1,H] 后 flatten."""
    normalized_shape = eval(param.get('normalized_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    epsilon = float(param.get('epsilon', 1e-06))
    gamma = (torch.rand(normalized_shape, device=device, dtype=dtype) + 0.5).flatten()
    beta = torch.randn(normalized_shape, device=device, dtype=dtype).flatten() * 0.1
    scale = torch.ones(1, device=device, dtype=dtype)
    offset = torch.zeros(1, device=device, dtype=torch.int8)
    return [gamma, beta, scale, offset, epsilon]


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
    json_path = os.path.join(os.path.dirname(__file__), "RmsNormQuant.json")
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
