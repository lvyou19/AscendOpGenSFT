"""Auto-generated benchmark file for ApplyAdamWV2.

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
    """Self-contained AdamW reference (does not depend on torch.optim internals)."""

    def __init__(self):
        super().__init__()

    def forward(self, var_ref: torch.Tensor, m_ref: torch.Tensor, v_ref: torch.Tensor,
                grad: torch.Tensor, step: torch.Tensor, max_grad_norm_ref: torch.Tensor,
                lr: float, beta1: float, beta2: float, weight_decay: float, eps: float,
                amsgrad: bool, maximize: bool) -> List[torch.Tensor]:
        dtype1 = var_ref.dtype
        if grad.dtype != dtype1:
            grad = grad.to(dtype1)
        if max_grad_norm_ref.dtype != dtype1:
            max_grad_norm_ref = max_grad_norm_ref.to(dtype1)

        compute_dtype = torch.float32
        p = var_ref.to(compute_dtype).clone()
        exp_avg = m_ref.to(compute_dtype).clone()
        exp_avg_sq = v_ref.to(compute_dtype).clone()
        g = grad.to(compute_dtype).clone()
        if maximize:
            g = -g
        # Source CSV stores (T-1); kernel adds 1 internally. Use T for bias.
        t = (float(step.item()) if isinstance(step, torch.Tensor) else float(step)) + 1.0

        p.mul_(1.0 - lr * weight_decay)
        exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

        bias1 = 1.0 - beta1 ** t
        bias2 = 1.0 - beta2 ** t
        step_size = lr / bias1
        bias_correction2_sqrt = bias2 ** 0.5

        if amsgrad:
            max_exp = max_grad_norm_ref.to(compute_dtype).clone()
            torch.maximum(max_exp, exp_avg_sq, out=max_exp)
            denom = (max_exp.sqrt() / bias_correction2_sqrt).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

        p.addcdiv_(exp_avg, denom, value=-step_size)

        var_ref.copy_(p.to(dtype1))
        m_ref.copy_(exp_avg.to(dtype1))
        v_ref.copy_(exp_avg_sq.to(dtype1))
        if amsgrad:
            max_grad_norm_ref.copy_(max_exp.to(dtype1))
            return [var_ref, m_ref, v_ref, max_grad_norm_ref]
        return [var_ref, m_ref, v_ref]


# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import numpy as np
import torch

def parse_bool_param(param):
    if isinstance(param, str) and param.lower() in ['true', 't', '1']:
        return True
    if param == 1:
        return True
    return False

def gen_input_data(shape, dtype_str, input_range):
    """
    生成AdamW优化器所需的输入数据。
    """
    dtype_map = {'float32': torch.float32, 'float': torch.float32, 'float16': torch.float16, 'half': torch.float16, 'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16, 'int64': torch.int64}
    dtype = dtype_map.get(dtype_str.lower(), torch.float32)
    np.random.seed(5)
    var = torch.tensor(np.random.uniform(input_range[0], input_range[1], shape), dtype=dtype)
    m = torch.tensor(np.random.uniform(input_range[0], input_range[1], shape), dtype=dtype)
    v = torch.tensor(np.random.uniform(input_range[0], input_range[1], shape), dtype=dtype)
    max_grad_norm = torch.tensor(np.random.uniform(input_range[0], input_range[1], shape), dtype=dtype)
    grad = torch.tensor(np.random.uniform(input_range[0], input_range[1], shape), dtype=dtype)
    return (var, m, v, max_grad_norm, grad)

def get_inputs(param, device=None):
    """
    根据参数生成 ApplyAdamWV2 算子的输入数据。
    """
    shape = eval(param.get('shape', '[2, 2, 2]'))
    dtype_str = str(param.get('dtype', 'float16')).lower()
    input_range = [0.1, 1]
    var_ref, m_ref, v_ref, max_grad_norm_ref, grad = gen_input_data(shape, dtype_str, input_range)
    t = int(param.get('step', 1))
    step_val = max(0, t - 1)
    step = torch.tensor([step_val], dtype=torch.int64)
    if device:
        var_ref = var_ref.to(device)
        m_ref = m_ref.to(device)
        v_ref = v_ref.to(device)
        max_grad_norm_ref = max_grad_norm_ref.to(device)
        grad = grad.to(device)
        step = step.to(device)
    lr = float(param.get('lr', '0.01'))
    beta1 = float(param.get('beta1', '0.9'))
    beta2 = float(param.get('beta2', '0.99'))
    weight_decay = float(param.get('weight_decay', '5e-3'))
    eps = float(param.get('eps', '1e-6'))
    amsgrad = parse_bool_param(param.get('amsgrad', False))
    maximize = parse_bool_param(param.get('maximize', False))
    return (var_ref, m_ref, v_ref, grad, step, max_grad_norm_ref, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize)

def get_init_inputs_per_case(param, device=None):
    """
    提取模型初始化参数。ApplyAdamWV2 不需要特殊的初始化参数。
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
    json_path = os.path.join(os.path.dirname(__file__), "ApplyAdamWV2.json")
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
