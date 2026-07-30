"""Auto-generated benchmark file for LayerNormGradV3.

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

def _layer_norm_grad_ref_pytorch(dy: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], normalized_shape: List[int]) -> List[torch.Tensor]:
    """标杆：直接用 PyTorch 官方 native_layer_norm_backward，公式与舍入与 PyTorch 完全一致."""
    output_mask = (True, True, True)
    dx, dgamma, dbeta = torch.ops.aten.native_layer_norm_backward(dy, x, normalized_shape, mean, rstd, weight, bias, output_mask)
    dtype_out = weight.dtype if weight is not None else x.dtype
    dev = x.device
    return [dx, dgamma if dgamma is not None else torch.zeros(normalized_shape, dtype=dtype_out, device=dev), dbeta if dbeta is not None else torch.zeros(normalized_shape, dtype=dtype_out, device=dev)]

def _layer_norm_grad_ref_fallback(dy: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, weight: Optional[torch.Tensor], normalized_shape: List[int]) -> List[torch.Tensor]:
    """无 aten 时的回退：手写公式，与 aclnn 对齐——dgamma/dbeta 在 float32 下 sum 再 cast 到输出 dtype."""
    input_dim = x.dim()
    normalized_dim = len(normalized_shape)
    reduction_dims = tuple(range(input_dim - normalized_dim, input_dim))
    N = 1
    for i in reduction_dims:
        N *= x.shape[i]
    dtype_orig = x.dtype
    compute_dtype = dtype_orig if dtype_orig in (torch.float16, torch.bfloat16) else torch.float32
    dy_c = dy.to(compute_dtype)
    x_c = x.to(compute_dtype)
    mean_c = mean.to(compute_dtype)
    rstd_c = rstd.to(compute_dtype)
    weight_c = weight.to(compute_dtype) if weight is not None else None
    x_norm = (x_c - mean_c) * rstd_c
    dy_weighted = dy_c * weight_c if weight_c is not None else dy_c
    N_t = torch.tensor(N, dtype=compute_dtype, device=x.device)
    sum1 = dy_weighted.sum(dim=reduction_dims, keepdim=True).to(compute_dtype)
    sum2 = (dy_weighted * x_norm).sum(dim=reduction_dims, keepdim=True).to(compute_dtype)
    c1 = sum1 / N_t
    c2 = sum2 / N_t
    dx = (dy_weighted - c1 - x_norm * c2) * rstd_c
    dx = dx.to(dtype_orig)
    batch_dims = tuple(range(0, input_dim - normalized_dim))
    dgamma = (dy_weighted.float() * x_norm.float()).sum(dim=batch_dims)
    dbeta = dy_c.float().sum(dim=batch_dims)
    dtype_out = weight.dtype if weight is not None else dtype_orig
    dgamma = dgamma.to(dtype_out)
    dbeta = dbeta.to(dtype_out)
    return [dx, dgamma, dbeta]

def _layer_norm_grad_ref(dy: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], normalized_shape: List[int]) -> List[torch.Tensor]:
    """优先用 PyTorch 官方 backward，否则回退到手写公式."""
    try:
        return _layer_norm_grad_ref_pytorch(dy, x, mean, rstd, weight, bias, normalized_shape)
    except Exception:
        return _layer_norm_grad_ref_fallback(dy, x, mean, rstd, weight, normalized_shape)

class Model(nn.Module):

    def __init__(self, normalized_shape: List[int]):
        super(Model, self).__init__()
        self.normalized_shape = normalized_shape

    def forward(self, dy: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor]) -> List[torch.Tensor]:
        return _layer_norm_grad_ref(dy, x, mean, rstd, weight, bias, self.normalized_shape)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
try:
    from framework.utils import check_precision
except ImportError:
    check_precision = None

def get_inputs(param, device=None):
    """
    Generate inputs for LayerNorm backward: dy, x, mean, rstd, gamma (weight), bias.
    mean/rstd 用 PyTorch native_layer_norm 前向一次得到，与标杆 backward 完全一致。
    """
    input_shape = eval(param.get('input_shape', '[1, 1]'))
    normalized_shape = eval(param.get('normalized_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    eps = float(param.get('epsilon', 1e-05))
    x = torch.rand(input_shape, device=device, dtype=dtype)
    dy = torch.rand(input_shape, device=device, dtype=dtype)
    weight_type = param.get('weight_type', 'present')
    bias_type = param.get('bias_type', 'present')
    gamma = torch.rand(normalized_shape, device=device, dtype=dtype) if weight_type == 'present' else None
    bias = torch.rand(normalized_shape, device=device, dtype=dtype) if bias_type == 'present' else None
    try:
        out, mean, rstd = torch.ops.aten.native_layer_norm(x, normalized_shape, gamma, bias, eps)
    except Exception:
        input_dim = len(input_shape)
        normalized_dim = len(normalized_shape)
        reduction_dims = tuple(range(input_dim - normalized_dim, input_dim))
        mean = x.mean(dim=reduction_dims, keepdim=True)
        var = (x - mean).pow(2).mean(dim=reduction_dims, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        mean_shape = list(input_shape)
        for i in range(normalized_dim):
            mean_shape[input_dim - 1 - i] = 1
        mean = mean.view(mean_shape)
        rstd = rstd.view(mean_shape)
    return (dy, x, mean, rstd, gamma, bias)

def get_init_inputs_per_case(param, device=None):
    normalized_shape = eval(param.get('normalized_shape', '[1]'))
    return [normalized_shape]


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
    json_path = os.path.join(os.path.dirname(__file__), "LayerNormGradV3.json")
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
