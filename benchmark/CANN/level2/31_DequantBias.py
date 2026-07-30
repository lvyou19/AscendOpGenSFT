"""Auto-generated benchmark file for DequantBias.

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
"""
CPU golden 与 ops-nn-dev ST 对齐：
`ops-nn-dev/quant/dequant_bias/tests/st/aclnnDequantBias/executor_aclnnDequantBias.py`
"""
from typing import List, Optional
import torch
import torch.nn as nn
GE_DT_FP16 = 1
GE_DT_BF16 = 27

def _golden_dequant_bias(x: torch.Tensor, weight_scale: torch.Tensor, activate_scale: Optional[torch.Tensor], bias: Optional[torch.Tensor], output_dtype: int) -> torch.Tensor:
    """复刻 executor_aclnnDequantBias.__call__ 分支逻辑。"""
    x_data = x
    weight_scale_data = weight_scale
    activate_scale_data = activate_scale
    bias_data = bias
    if bias_data is not None:
        if bias_data.dtype == torch.int32:
            if activate_scale_data is None:
                y = (x_data.to(torch.float32) + bias_data.to(torch.float32)) * weight_scale_data.to(torch.float32)
            else:
                a = activate_scale_data.to(torch.float32)[:, None]
                y = (x_data.to(torch.float32) + bias_data.to(torch.float32)) * weight_scale_data.to(torch.float32) * a
        elif bias_data.dtype in (torch.float16, torch.bfloat16, torch.float32):
            if activate_scale_data is None:
                y = x_data.to(torch.float32) * weight_scale_data.to(torch.float32) + bias_data.to(torch.float32)
            else:
                a = activate_scale_data.to(torch.float32)[:, None]
                y = x_data.to(torch.float32) * weight_scale_data.to(torch.float32) * a + bias_data.to(torch.float32)
        else:
            raise ValueError(f'Unsupported bias dtype: {bias_data.dtype}')
    elif activate_scale_data is None:
        y = x_data.to(torch.float32) * weight_scale_data.to(torch.float32)
    else:
        a = activate_scale_data.to(torch.float32)[:, None]
        y = x_data.to(torch.float32) * weight_scale_data.to(torch.float32) * a
    if output_dtype == GE_DT_FP16:
        return y.to(torch.float16)
    if output_dtype == GE_DT_BF16:
        return y.to(torch.bfloat16)
    raise ValueError(f'golden 仅支持 output_dtype 1(fp16) 或 27(bf16)，收到 {output_dtype}')

class Model(nn.Module):

    def __init__(self, output_dtype: int):
        super().__init__()
        self.output_dtype = int(output_dtype)

    def forward(self, x: torch.Tensor, weight_scale: torch.Tensor, activate_scale: Optional[torch.Tensor]=None, bias: Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        y = _golden_dequant_bias(x, weight_scale, activate_scale, bias, self.output_dtype)
        return [y]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
GE_DT_FP16 = 1
GE_DT_BF16 = 27

def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('true', '1', 'yes')

def _bias_from_dtype(dtype_str, shape, device):
    if dtype_str in ('none', '', 'null'):
        return None
    if dtype_str == 'int32':
        return torch.randint(-32, 32, shape, device=device, dtype=torch.int32)
    if dtype_str == 'float32':
        return (torch.rand(shape, device='cpu', dtype=torch.float32) * 0.2 - 0.1).to(device)
    if dtype_str == 'float16':
        return (torch.rand(shape, device='cpu', dtype=torch.float32) * 0.2 - 0.1).to(device=device, dtype=torch.float16)
    if dtype_str == 'bfloat16':
        return (torch.rand(shape, device='cpu', dtype=torch.float32) * 0.2 - 0.1).to(device=device, dtype=torch.bfloat16)
    raise ValueError(f'unknown bias_dtype: {dtype_str}')

def get_inputs(param, device=None):
    shape = eval(param.get('input_shape', '[4, 16]'), {'__builtins__': {}})
    m, n = (int(shape[0]), int(shape[1]))
    has_activate = _parse_bool(param.get('has_activate', True))
    has_bias = _parse_bool(param.get('has_bias', False))
    bias_dtype = str(param.get('bias_dtype', 'none')).strip().lower()
    output_dtype = int(param.get('output_dtype', GE_DT_FP16))
    x = torch.randint(-64, 64, (m, n), device=device, dtype=torch.int32)
    w_cpu = torch.rand(n, dtype=torch.float32, device='cpu') * 0.05 + 0.01
    if output_dtype == GE_DT_BF16:
        weight_scale = w_cpu.to(device=device, dtype=torch.bfloat16)
    else:
        weight_scale = w_cpu.to(device=device, dtype=torch.float32)
    activate_scale = None
    if has_activate:
        activate_scale = (torch.rand(m, device='cpu', dtype=torch.float32) * 0.1 + 0.95).to(device=device)
    bias = None
    if has_bias:
        bias = _bias_from_dtype(bias_dtype, (n,), device)
    return (x, weight_scale, activate_scale, bias)

def get_init_inputs_per_case(param, device=None):
    output_dtype = int(param.get('output_dtype', GE_DT_FP16))
    return [output_dtype]


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
    json_path = os.path.join(os.path.dirname(__file__), "DequantBias.json")
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
