"""Auto-generated benchmark file for PadV4Grad.

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
import torch.nn as nn
import torch.nn.functional as F

def _pytorch_pad_tuple_from_ge_eight(ge_pad):
    pt, pb, pl, pr = (int(ge_pad[4]), int(ge_pad[5]), int(ge_pad[6]), int(ge_pad[7]))
    return (pl, pr, pt, pb)

class Model(nn.Module):
    """CPU 参考：reflect pad 反向；半精度在 CPU 上可能不完整，统一 float32 反传再 cast。"""

    def __init__(self):
        super().__init__()

    def forward(self, grad_output: torch.Tensor, ge_padding):
        pad = _pytorch_pad_tuple_from_ge_eight(ge_padding)
        if grad_output.dim() != 4:
            raise ValueError('PadV4Grad validation 仅支持 4D NCHW')
        h = grad_output.shape[2] - pad[2] - pad[3]
        w = grad_output.shape[3] - pad[0] - pad[1]
        self_shape = (grad_output.shape[0], grad_output.shape[1], h, w)
        orig_dtype = grad_output.dtype
        g = grad_output.to(dtype=torch.float32)
        grad_input = torch.zeros(self_shape, dtype=torch.float32, device=g.device, requires_grad=True)
        out = F.pad(grad_input, pad, mode='reflect')
        out.backward(g)
        return grad_input.grad.to(dtype=orig_dtype)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import ast
import torch

def _padded_shape(self_shape, padding_four):
    s = list(self_shape)
    pl, pr, pt, pb = padding_four
    s[-2] += pt + pb
    s[-1] += pl + pr
    return tuple(s)

def _ge_padding_eight(padding_four):
    pl, pr, pt, pb = padding_four
    return [0, 0, 0, 0, int(pt), int(pb), int(pl), int(pr)]

def get_inputs(param, device=None):
    shape = ast.literal_eval(param.get('input_shape', '[1,1,4,4]'))
    shape = tuple((int(x) for x in shape))
    padding_four = [int(x) for x in ast.literal_eval(param.get('padding', '[1,1,1,1]'))]
    if len(padding_four) != 4:
        raise ValueError('PadV4Grad 仅支持 4D NCHW，padding 为 (left,right,top,bottom)')
    if len(shape) != 4:
        raise ValueError('PadV4Grad validation 仅支持 4D')
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    pl, pr, pt, pb = padding_four
    h, w = (shape[2], shape[3])
    if not (pt < h and pb < h and (pl < w) and (pr < w)):
        raise ValueError(f'reflect 约束要求 pad 各边 < 空间维: shape={shape}, pad={padding_four}')
    grad_shape = _padded_shape(shape, padding_four)
    grad_output = torch.randn(grad_shape, device=device, dtype=dtype)
    ge_pad = _ge_padding_eight(padding_four)
    return (grad_output, ge_pad)

def get_init_inputs_per_case(param, device=None):
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
    json_path = os.path.join(os.path.dirname(__file__), "PadV4Grad.json")
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
