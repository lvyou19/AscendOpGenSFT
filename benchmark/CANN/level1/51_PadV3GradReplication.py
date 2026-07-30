"""Auto-generated benchmark file for PadV3GradReplication.

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

def _pytorch_pad_from_ge10(ge_pad):
    """ge 10 元组 -> torch.nn.functional.pad 的 pad（自最后一维向前：W,H,D,C,N）。"""
    if len(ge_pad) != 10:
        raise ValueError('ge_pad 长度须为 10')
    return (int(ge_pad[8]), int(ge_pad[9]), int(ge_pad[6]), int(ge_pad[7]), int(ge_pad[4]), int(ge_pad[5]), int(ge_pad[2]), int(ge_pad[3]), int(ge_pad[0]), int(ge_pad[1]))

class Model(nn.Module):
    """CPU 参考：replicate 反传；4D 用 F.pad；5D 用 ReplicationPad3d（F.pad replicate 在部分 CPU 上未实现）。"""

    def __init__(self):
        super().__init__()

    def forward(self, grad_output: torch.Tensor, ge_padding):
        ge_list = list(ge_padding)
        nd = grad_output.dim()
        if nd not in (4, 5):
            raise ValueError('PadV3GradReplication validation 仅支持 4D / 5D')
        orig_dtype = grad_output.dtype
        g = grad_output.to(dtype=torch.float32)
        if nd == 4:
            pad = _pytorch_pad_from_ge10(ge_list)
            eff = pad[:4]
            out_sizes = list(g.shape)
            out_sizes[-2] -= eff[2] + eff[3]
            out_sizes[-1] -= eff[0] + eff[1]
            grad_input = torch.zeros(out_sizes, dtype=torch.float32, device=g.device, requires_grad=True)
            out = F.pad(grad_input, eff, mode='replicate')
            out.backward(g)
            return grad_input.grad.to(dtype=orig_dtype)
        dl, dr, hl, hr, wl, wr = (int(ge_list[i]) for i in range(4, 10))
        out_sizes = [g.shape[0], g.shape[1], g.shape[2] - dl - dr, g.shape[3] - hl - hr, g.shape[4] - wl - wr]
        pad3d = (wl, wr, hl, hr, dl, dr)
        m = nn.ReplicationPad3d(pad3d)
        grad_input = torch.zeros(out_sizes, dtype=torch.float32, device=g.device, requires_grad=True)
        out = m(grad_input)
        out.backward(g)
        return grad_input.grad.to(dtype=orig_dtype)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import ast
import torch

def _ge10_from_padding_4(pl, pr, pt, pb):
    return [0, 0, 0, 0, 0, 0, int(pt), int(pb), int(pl), int(pr)]

def _ge10_from_padding_6(wl, wr, hl, hr, dl, dr):
    return [0, 0, 0, 0, int(dl), int(dr), int(hl), int(hr), int(wl), int(wr)]

def _padded_shape(self_shape, padding_list):
    s = list(self_shape)
    if len(self_shape) == 4:
        pl, pr, pt, pb = padding_list
        s[-2] += pt + pb
        s[-1] += pl + pr
    elif len(self_shape) == 5:
        wl, wr, hl, hr, dl, dr = padding_list
        s[-3] += dl + dr
        s[-2] += hl + hr
        s[-1] += wl + wr
    else:
        raise ValueError('PadV3GradReplication 仅支持 4D 或 5D input_shape')
    return tuple(s)

def get_inputs(param, device=None):
    shape = ast.literal_eval(param.get('input_shape', '[1,1,2,2]'))
    shape = tuple((int(x) for x in shape))
    padding = [int(x) for x in ast.literal_eval(param.get('padding', '[1,1,1,1]'))]
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    if len(shape) == 4:
        if len(padding) != 4:
            raise ValueError('4D 用例 padding 须为 [pl,pr,pt,pb]（与 torch F.pad 一致）')
        ge_pad = _ge10_from_padding_4(*padding)
    elif len(shape) == 5:
        if len(padding) != 6:
            raise ValueError('5D 用例 padding 须为 [wl,wr,hl,hr,dl,dr]（末三维 W,H,D）')
        ge_pad = _ge10_from_padding_6(*padding)
    else:
        raise ValueError('input_shape 须为 4D 或 5D')
    grad_shape = _padded_shape(shape, padding)
    grad_output = torch.randn(grad_shape, device=device, dtype=dtype)
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
    json_path = os.path.join(os.path.dirname(__file__), "PadV3GradReplication.json")
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
