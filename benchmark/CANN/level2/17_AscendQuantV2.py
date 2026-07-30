"""Auto-generated benchmark file for AscendQuantV2.

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
`ops-nn-dev/quant/ascend_quant/tests/st/aclnnAscendQuant/executor_aclnnAscendQuant.py`
（.codex/TODO.md：AscendQuantV2 → quant → ops-nn-dev；L2 接口名为 aclnnAscendQuant，axis 固定 -1）。
"""
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
_GOLDEN_AXIS = -1

def _golden_aclnn_ascend_quant_int8(x: torch.Tensor, scale: torch.Tensor, offset: Optional[torch.Tensor], sqrt_mode: bool, round_mode: str, dst_type: int) -> torch.Tensor:
    """
    复刻 executor 中 __call__ 的计算顺序与舍入（float32 → numpy → np.round 等），仅实现 dst_type==2 (int8)。
    """
    x_t = x.to(torch.float32)
    scale_t = scale.to(torch.float32)
    offset_t = offset.to(torch.float32) if offset is not None else None
    if len(scale_t.shape) == 1:
        scale_new_shape = [1] * len(x_t.shape)
        scale_new_shape[_GOLDEN_AXIS] = scale_t.shape[0]
        scale_t = torch.reshape(scale_t, scale_new_shape)
        if offset_t is not None:
            offset_t = torch.reshape(offset_t, scale_new_shape)
    x_np = x_t.detach().cpu().numpy()
    scale_np = scale_t.detach().cpu().numpy()
    offset_np = offset_t.detach().cpu().numpy() if offset_t is not None else None
    if sqrt_mode:
        scale_sqrt = x_np * scale_np
        scale_rst = scale_sqrt * scale_np
    else:
        scale_rst = x_np * scale_np
    if offset_np is not None:
        add_offset = scale_rst + offset_np
    else:
        add_offset = scale_rst
    rm = str(round_mode).strip()
    if rm == 'round':
        round_data = np.round(add_offset, 0)
    elif rm == 'floor':
        round_data = np.floor(add_offset)
    elif rm == 'ceil':
        round_data = np.ceil(add_offset)
    elif rm == 'trunc':
        round_data = np.trunc(add_offset)
    else:
        raise ValueError(f'unsupported round_mode: {round_mode}')
    if dst_type == 2:
        round_data = np.clip(round_data, -128, 127)
        return torch.from_numpy(round_data.astype(np.int8)).to(x.device)
    raise ValueError(f'golden 仅支持 dst_type==2 (int8)，收到 {dst_type}')

class Model(nn.Module):

    def __init__(self, sqrt_mode: bool, round_mode: str, dst_type: int):
        super().__init__()
        self.sqrt_mode = bool(sqrt_mode)
        self.round_mode = str(round_mode)
        self.dst_type = int(dst_type)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, offset: Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        assert self.dst_type == 2, 'validation 仅校验 int8 输出 (ge::DT_INT8 == 2)'
        y = _golden_aclnn_ascend_quant_int8(x, scale, offset, self.sqrt_mode, self.round_mode, self.dst_type)
        return [y]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
GE_DT_INT8 = 2

def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('true', '1', 'yes')

def _make_scale_offset(shape, scale_mode, x_dtype, device, has_offset):
    nd = len(shape)
    if scale_mode == 'per_tensor':
        scale_shape = (1,) * nd
        scale_v = torch.tensor(0.11, device=device, dtype=x_dtype).reshape(scale_shape)
    elif scale_mode == 'per_channel':
        c = shape[-1]
        scale_shape = (1,) * (nd - 1) + (c,)
        base = torch.rand(c, device='cpu', dtype=torch.float32) * 0.14 + 0.06
        scale_v = base.to(device=device, dtype=x_dtype).reshape(scale_shape)
    else:
        raise ValueError(f'unknown scale_mode: {scale_mode}')
    if has_offset:
        offset_v = (torch.rand(scale_shape, device='cpu', dtype=torch.float32) * 0.5 - 0.25).to(device=device, dtype=x_dtype)
    else:
        offset_v = None
    return (scale_v, offset_v)

def get_inputs(param, device=None):
    shape = eval(param.get('input_shape', '[8, 32]'), {'__builtins__': {}})
    dtype_str = param.get('x_dtype', 'float16')
    x_dtype = getattr(torch, dtype_str)
    scale_mode = str(param.get('scale_mode', 'per_channel')).strip()
    has_offset = _parse_bool(param.get('has_offset', False))
    x = (torch.rand(shape, device='cpu', dtype=torch.float32) * 2.0 - 1.0).to(device=device, dtype=x_dtype)
    scale, offset = _make_scale_offset(shape, scale_mode, x_dtype, device, has_offset)
    return (x, scale, offset)

def get_init_inputs_per_case(param, device=None):
    sqrt_mode = _parse_bool(param.get('sqrt_mode', False))
    round_mode = str(param.get('round_mode', 'round')).strip()
    return [sqrt_mode, round_mode, GE_DT_INT8]


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
    json_path = os.path.join(os.path.dirname(__file__), "AscendQuantV2.json")
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
