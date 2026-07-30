"""Auto-generated benchmark file for SwiGluQuant.

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
CPU 金标准对齐 kernel：SwiGLU 与文档非 MoE / MoE 动态公式；动态量化 scale = dstScale / rowmax(|Ytmp|)；
Cast 路径近似 swi_glu_quant_base.h::CastQuantOut（rint→int32→fp16→trunc）。
静态量化 scale 输出在核上写 0，golden 对 scale 全 0 比对。
"""
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
DT_INT8 = 2
DT_INT4 = 29

def swiglu_act(x_f: torch.Tensor, activate_left: bool) -> torch.Tensor:
    h = x_f.shape[-1] // 2
    a = x_f[..., :h]
    b = x_f[..., h:]
    if activate_left:
        return torch.nn.functional.silu(a) * b
    return torch.nn.functional.silu(b) * a

def cast_like_kernel(fp: torch.Tensor, dst_type: int) -> torch.Tensor:
    t = fp.detach().float()
    i32 = torch.round(t).to(torch.int32)
    h = i32.to(torch.float16).float()
    if int(dst_type) == DT_INT4:
        out = torch.round(h).clamp(-8, 7).to(torch.int8)
    else:
        out = torch.trunc(h).clamp(-128, 127).to(torch.int8)
    return out.to(device=fp.device)

def _dynamic_segment(seg: torch.Tensor, dst_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    m = seg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    s = dst_scale / m
    q = seg * s
    return (q, s.squeeze(-1))

def golden_dynamic(x: torch.Tensor, smooth: torch.Tensor, activate_left: bool, dst_type: int, group_index: Optional[torch.Tensor], group_list_type: int) -> Tuple[torch.Tensor, torch.Tensor]:
    act = swiglu_act(x.float(), activate_left)
    dst_scale = 127.0 if int(dst_type) == DT_INT8 else 7.0
    sm = smooth.float()
    lead = act.shape[:-1]
    rn = int(torch.tensor(lead, dtype=torch.int64).prod().item())
    nc = act.shape[-1]
    a2 = act.reshape(rn, nc)
    y_acc = torch.zeros(rn, nc, dtype=torch.float32, device=x.device)
    scale_1d = torch.zeros(rn, dtype=torch.float32, device=x.device)
    if group_index is None:
        sm2 = sm.unsqueeze(0) if sm.dim() == 1 else sm
        if sm2.shape[-1] != nc:
            raise ValueError('smooth last dim must match SwiGLU output width')
        y_tmp = a2 * sm2
        q, sc = _dynamic_segment(y_tmp, dst_scale)
        y_acc = q
        scale_1d = sc
    else:
        g = group_index.detach().cpu().numpy().astype(np.int64)
        if int(group_list_type) == 1:
            g = np.cumsum(g)
        G = int(sm.shape[0])
        sm2 = sm.reshape(G, -1)
        if sm2.shape[1] != nc:
            raise ValueError('MoE smooth shape mismatch')
        start = 0
        for gi in range(len(g)):
            end = int(g[gi])
            if end <= start or end > rn:
                continue
            seg = a2[start:end] * sm2[gi:gi + 1]
            q, sc = _dynamic_segment(seg, dst_scale)
            y_acc[start:end] = q
            scale_1d[start:end] = sc
            start = end
    y_q = cast_like_kernel(y_acc.reshape_as(act), dst_type)
    scale_out = scale_1d.reshape(lead)
    return (y_q, scale_out)

def golden_static(x: torch.Tensor, smooth: torch.Tensor, offset: torch.Tensor, activate_left: bool, dst_type: int, group_index: Optional[torch.Tensor], group_list_type: int, static_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    act = swiglu_act(x.float(), activate_left)
    sm = smooth.float()
    off = offset.float()
    lead = act.shape[:-1]
    rn = int(torch.tensor(lead, dtype=torch.int64).prod().item())
    nc = act.shape[-1]
    a2 = act.reshape(rn, nc)
    if group_index is None:
        if static_mode == 'per_tensor':
            y_tmp = a2 * sm.reshape(-1)[0] + off.reshape(-1)[0]
        else:
            sm2 = sm.unsqueeze(0) if sm.dim() == 1 else sm
            off2 = off.unsqueeze(0) if off.dim() == 1 else off
            y_tmp = a2 * sm2 + off2
    else:
        g = group_index.detach().cpu().numpy().astype(np.int64)
        if int(group_list_type) == 1:
            g = np.cumsum(g)
        G = int(sm.shape[0])
        sm2 = sm.reshape(G, -1) if sm.dim() > 1 else sm.reshape(G, 1)
        off2 = off.reshape(G, -1) if off.dim() > 1 else off.reshape(G, 1)
        y_acc = torch.zeros(rn, nc, dtype=torch.float32, device=x.device)
        start = 0
        for gi in range(len(g)):
            end = int(g[gi])
            if end <= start or end > rn:
                continue
            if static_mode == 'per_tensor':
                y_acc[start:end] = a2[start:end] * sm2[gi, 0] + off2[gi, 0]
            else:
                y_acc[start:end] = a2[start:end] * sm2[gi:gi + 1] + off2[gi:gi + 1]
            start = end
        y_tmp = y_acc
    y_q = cast_like_kernel(y_tmp.reshape_as(act), dst_type)
    scale_z = torch.zeros(lead, dtype=torch.float32, device=x.device)
    return (y_q, scale_z)

class Model(nn.Module):

    def __init__(self, activate_left: bool, quant_mode: str, group_list_type: int, dst_type: int, static_mode: str):
        super().__init__()
        self.activate_left = bool(activate_left)
        self.quant_mode = str(quant_mode)
        self.group_list_type = int(group_list_type)
        self.dst_type = int(dst_type)
        self.static_mode = str(static_mode)

    def forward(self, x: torch.Tensor, smooth: torch.Tensor, offset: Optional[torch.Tensor], group_index: Optional[torch.Tensor]) -> List[torch.Tensor]:
        if self.quant_mode == 'dynamic':
            y, s = golden_dynamic(x, smooth, self.activate_left, self.dst_type, group_index, self.group_list_type)
        else:
            assert offset is not None
            y, s = golden_static(x, smooth, offset, self.activate_left, self.dst_type, group_index, self.group_list_type, self.static_mode)
        return [y, s]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('true', '1', 'yes')

def get_init_inputs_per_case(param, device=None):
    static_mode = str(param.get('static_mode', 'none'))
    if str(param.get('quant_mode', 'dynamic')) == 'dynamic':
        static_mode = 'none'
    return (_parse_bool(param.get('activate_left', False)), str(param.get('quant_mode', 'dynamic')), int(param.get('group_list_type', 0)), int(param.get('dst_type', 2)), static_mode)

def _half_cols(param, shape):
    last = int(shape[-1])
    if last % 2 != 0:
        raise ValueError('x last dim must be even')
    h = last // 2
    if int(param.get('dst_type', 2)) == 29 and last % 4 != 0:
        raise ValueError('int4 requires x last dim divisible by 4')
    return h

def get_inputs(param, device=None):
    shape = eval(param.get('input_shape', '[4, 32]'), {'__builtins__': {}})
    if len(shape) < 2:
        raise ValueError('x must be at least 2D')
    x_dtype = getattr(torch, str(param.get('x_dtype', 'float16')))
    half = _half_cols(param, shape)
    qm = str(param.get('quant_mode', 'dynamic'))
    static_mode = str(param.get('static_mode', 'per_channel'))
    has_group = _parse_bool(param.get('has_group', False))
    torch.manual_seed(int(param.get('case_id', 0)))
    x = (torch.rand(shape, device='cpu', dtype=torch.float32) * 2.0 - 1.0).to(device=device, dtype=x_dtype)
    if has_group:
        g = int(param.get('num_groups', 2))
        rn = 1
        for d in shape[:-1]:
            rn *= int(d)
        part = max(1, rn // g)
        bounds = []
        c = 0
        for i in range(g - 1):
            c = min(c + part, rn - 1)
            bounds.append(c)
        bounds.append(rn)
        group_index = torch.tensor(bounds, dtype=torch.int32, device=device)
        smooth = (torch.rand((g, half), device='cpu', dtype=torch.float32) * 0.4 + 0.3).to(device=device)
    else:
        group_index = None
        if qm == 'dynamic' or static_mode == 'per_channel':
            smooth = (torch.rand((1, half), device='cpu', dtype=torch.float32) * 0.4 + 0.3).to(device=device)
        else:
            smooth = (torch.rand(1, device='cpu', dtype=torch.float32) * 0.4 + 0.3).to(device=device)
    if qm == 'static':
        if has_group:
            g = int(param.get('num_groups', 2))
            if static_mode == 'per_tensor':
                off = (torch.rand(g, device='cpu', dtype=torch.float32) * 0.02).to(device=device)
            else:
                off = (torch.rand((g, half), device='cpu', dtype=torch.float32) * 0.02).to(device=device)
        elif static_mode == 'per_tensor':
            off = (torch.rand(1, device='cpu', dtype=torch.float32) * 0.02).to(device=device)
        else:
            off = (torch.rand((1, half), device='cpu', dtype=torch.float32) * 0.02).to(device=device)
    else:
        off = None
    return (x, smooth, off, group_index)


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
    json_path = os.path.join(os.path.dirname(__file__), "SwiGluQuant.json")
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
