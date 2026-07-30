"""Auto-generated benchmark file for MoeGatingTopKSoftmaxV2.

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
import torch
import numpy as np

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, finished_optional: torch.Tensor, k: int, renorm: int):

        def softmax_func(x_np, axis=None):
            x_np = x_np.astype(np.float32)
            x_max = x_np.max(axis=axis, keepdims=True)
            x_sub = x_np - x_max
            y = np.exp(x_sub)
            x_sum = y.sum(axis=axis, keepdims=True)
            ans = y / x_sum
            return (ans, x_max, x_sum)
        gating_np = x.to(torch.float32).cpu().numpy()
        num_expert = gating_np.shape[-1]
        softmax_out, _, _ = softmax_func(gating_np, -1)
        indices = np.argsort(-softmax_out, axis=-1, kind='stable')
        indices = indices[:, :k]
        out = np.take_along_axis(softmax_out, indices, axis=-1)
        if renorm == 1:
            out_sum = out.sum(axis=-1, keepdims=True) + 1e-10
            out = out / out_sum
        if finished_optional is not None:
            finished_optional_np = finished_optional.cpu().numpy()
            finished_optional_np = finished_optional_np.reshape(finished_optional_np.shape[0], 1)
            finished_optional_np = np.tile(finished_optional_np, (1, k))
            indices = np.where(finished_optional_np, num_expert, indices)
        return [torch.from_numpy(out).to(x.device, dtype=x.dtype), torch.from_numpy(indices).to(x.device, dtype=torch.int32), torch.from_numpy(softmax_out).to(x.device, dtype=torch.float32)]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import random
torch.manual_seed(42)
random.seed(42)

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    dtype_str = param.get('dtype', 'float32')
    if dtype_str == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = getattr(torch, dtype_str)
    x_shape = eval(param.get('x_shape', '[5, 3]'))
    k = int(param.get('k', '3'))
    has_finished_optional = bool(param.get('has_finished_optional', 'False'))
    renorm = int(param.get('renorm', '0'))
    x = torch.randn(x_shape, device=device, dtype=dtype)
    if has_finished_optional:
        finished_optional = torch.randint(0, 2, (x_shape[0],), device=device, dtype=torch.bool)
    else:
        finished_optional = None
    return (x, finished_optional, k, renorm)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for MoeGatingTopKSoftmaxV2.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
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
    json_path = os.path.join(os.path.dirname(__file__), "MoeGatingTopKSoftmaxV2.json")
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
