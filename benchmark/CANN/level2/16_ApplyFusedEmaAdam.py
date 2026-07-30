"""Auto-generated benchmark file for ApplyFusedEmaAdam.

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
import math

class Model(nn.Module):
    """
    实现FusedEmaAdam融合优化器功能的模型。
    """

    def __init__(self, lr=0.001, emaDecay=0.999, beta1=0.9, beta2=0.999, eps=1e-08, mode=0, biasCorrection=True, weightDecay=0.0):
        """
        初始化模型。
        """
        super(Model, self).__init__()
        self.lr = lr
        self.emaDecay = emaDecay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mode = mode
        self.biasCorrection = biasCorrection
        self.weightDecay = weightDecay

    def forward(self, grad: torch.Tensor, varRef: torch.Tensor, mRef: torch.Tensor, vRef: torch.Tensor, sRef: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """
        实现Adam+EMA优化计算逻辑

        参数:
            grad: 梯度
            varRef: 变量引用
            mRef: 一阶动量引用
            vRef: 二阶动量引用
            sRef: EMA平均值引用
            step: 当前步数

        返回:
            更新后的[变量, 一阶动量, 二阶动量, EMA平均值]
        """
        original_dtype = grad.dtype
        need_cast = original_dtype == torch.float16 or original_dtype == torch.bfloat16
        if need_cast:
            grad = grad.to(torch.float32)
            varRef = varRef.to(torch.float32)
            mRef = mRef.to(torch.float32)
            vRef = vRef.to(torch.float32)
            sRef = sRef.to(torch.float32)
        if self.biasCorrection:
            beta1_correction = 1.0 - self.beta1 ** step
            beta2_correction = 1.0 - self.beta2 ** step
        else:
            beta1_correction = 1.0
            beta2_correction = 1.0
        if self.mode == 0:
            grad_ = grad + self.weightDecay * varRef
        elif self.mode == 1:
            grad_ = grad
        m_ = self.beta1 * mRef + (1 - self.beta1) * grad_
        v_ = self.beta2 * vRef + (1 - self.beta2) * grad_ * grad_
        next_m = m_ / beta1_correction
        next_v = v_ / beta2_correction
        denom = torch.sqrt(next_v) + self.eps
        if self.mode == 0:
            update = next_m / denom
        elif self.mode == 1:
            update = next_m / denom + self.weightDecay * varRef
        var_ = varRef - self.lr * update
        s_ = self.emaDecay * sRef + (1 - self.emaDecay) * var_
        if need_cast:
            var_ = var_.to(original_dtype)
            m_ = m_.to(original_dtype)
            v_ = v_.to(original_dtype)
            s_ = s_.to(original_dtype)
        return [var_, m_, v_, s_]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    step = torch.tensor([param.get('step', 1)], device=device, dtype=torch.int64)
    grad = torch.randn(shape, device=device, dtype=dtype)
    varRef = torch.randn(shape, device=device, dtype=dtype)
    mRef = torch.zeros(shape, device=device, dtype=dtype)
    vRef = torch.zeros(shape, device=device, dtype=dtype)
    sRef = torch.zeros(shape, device=device, dtype=dtype)
    return (grad, varRef, mRef, vRef, sRef, step)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        dict: Model initialization parameters
    """
    lr = param.get('lr', 0.001)
    emaDecay = param.get('emaDecay', 0.999)
    beta1 = param.get('beta1', 0.9)
    beta2 = param.get('beta2', 0.999)
    eps = param.get('eps', 1e-08)
    mode = param.get('mode', 0)
    biasCorrection = bool(param.get('biasCorrection', 'True'))
    weightDecay = param.get('weightDecay', 0.0)
    return (lr, emaDecay, beta1, beta2, eps, mode, biasCorrection, weightDecay)


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
    json_path = os.path.join(os.path.dirname(__file__), "ApplyFusedEmaAdam.json")
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
