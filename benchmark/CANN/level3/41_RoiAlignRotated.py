"""Auto-generated benchmark file for RoiAlignRotated.

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
import math
import torch
import torch.nn as nn

def _bilinear_kernel(feat: torch.Tensor, y: float, x: float, input_h: int, input_w: int) -> torch.Tensor:
    """
    与 AscendC bilinear_interpolate 一致：越界返回 0；边界与 floor/ceil 处理与 kernel 对齐。
    feat: (H, W, C)
    """
    c = feat.shape[-1]
    device, dtype = (feat.device, feat.dtype)
    if y < -1.0 or y > float(input_h) or x < -1.0 or (x > float(input_w)):
        return torch.zeros(c, device=device, dtype=dtype)
    y = max(y, 0.0)
    x = max(x, 0.0)
    x_floor = int(math.floor(x))
    y_floor = int(math.floor(y))
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1
    if x_floor >= input_w - 1:
        x_ceil = input_w - 1
        x_floor = x_ceil
        x = float(x_ceil)
    if y_floor >= input_h - 1:
        y_ceil = input_h - 1
        y_floor = y_ceil
        y = float(y_ceil)
    lx = x - float(x_floor)
    ly = y - float(y_floor)
    hx = 1.0 - lx
    hy = 1.0 - ly
    p1 = feat[y_floor, x_floor]
    p2 = feat[y_floor, x_ceil]
    p3 = feat[y_ceil, x_floor]
    p4 = feat[y_ceil, x_ceil]
    return hy * hx * p1 + hy * lx * p2 + ly * hx * p3 + ly * lx * p4

def _roi_align_rotated_reference(x: torch.Tensor, rois: torch.Tensor, pooled_h: int, pooled_w: int, spatial_scale: float, sampling_ratio: int, aligned: bool, clockwise: bool) -> torch.Tensor:
    """
    x: (B, H, W, C)，rois: (6, N) 行为 [batch_idx, cx, cy, w, h, theta]（与 kernel 平面布局一致）。
    输出: (N, pooled_h, pooled_w, C)
    """
    _, input_h, input_w, channels = x.shape
    n = rois.size(1)
    offset = -0.5 if aligned else 0.0
    out = torch.zeros((n, pooled_h, pooled_w, channels), dtype=x.dtype, device=x.device)
    xf = x.float()
    rf = rois.float()
    for j in range(n):
        b = int(rf[0, j].item())
        cx = float(rf[1, j] * spatial_scale + offset)
        cy = float(rf[2, j] * spatial_scale + offset)
        rw = float(rf[3, j] * spatial_scale)
        rh = float(rf[4, j] * spatial_scale)
        theta = rf[5, j]
        if not aligned:
            rw = max(rw, 1.0)
            rh = max(rh, 1.0)
        if clockwise:
            theta = -theta
        theta_f = float(theta)
        sin_t = math.sin(theta_f)
        cos_t = math.cos(theta_f)
        roi_start_h = -0.5 * rh
        roi_start_w = -0.5 * rw
        bin_size_h = rh / float(pooled_h)
        bin_size_w = rw / float(pooled_w)
        if sampling_ratio > 0:
            bin_grid_h = sampling_ratio
            bin_grid_w = sampling_ratio
        else:
            bin_grid_h = int(math.ceil(bin_size_h))
            bin_grid_w = int(math.ceil(bin_size_w))
            if bin_grid_h < 1:
                bin_grid_h = 1
            if bin_grid_w < 1:
                bin_grid_w = 1
        grid_h = bin_size_h / float(bin_grid_h)
        grid_w = bin_size_w / float(bin_grid_w)
        count = max(float(bin_grid_h * bin_grid_w), 1.0)
        feat_b = xf[b]
        for idx in range(pooled_h * pooled_w):
            ph = idx // pooled_w
            pw = idx - ph * pooled_w
            acc = torch.zeros(channels, dtype=torch.float32, device=x.device)
            for iy in range(bin_grid_h):
                yy = roi_start_h + ph * bin_size_h + (iy + 0.5) * grid_h
                for ix in range(bin_grid_w):
                    xx = roi_start_w + pw * bin_size_w + (ix + 0.5) * grid_w
                    y_img = yy * cos_t - xx * sin_t + float(cy)
                    x_img = yy * sin_t + xx * cos_t + float(cx)
                    acc = acc + _bilinear_kernel(feat_b, y_img, x_img, input_h, input_w).float()
            out[j, ph, pw] = (acc / count).to(dtype=x.dtype)
    return out

class Model(nn.Module):
    """CPU 金标准：与 op_kernel 中 RoiAlignRotated 采样与双线性逻辑对齐。"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, rois: torch.Tensor, pooled_h: int, pooled_w: int, spatial_scale: float, sampling_ratio: int, aligned: bool, clockwise: bool) -> torch.Tensor:
        return _roi_align_rotated_reference(x, rois, int(pooled_h), int(pooled_w), float(spatial_scale), int(sampling_ratio), bool(aligned), bool(clockwise))

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import math
import numpy as np
import torch

def parse_bool_param(param):
    if isinstance(param, str) and param.lower() in ['true', 't', '1']:
        return True
    if param == 1:
        return True
    return False

def get_inputs(param, device=None):
    rng = np.random.default_rng(11 + int(param.get('case_id', 0)))
    dtype_map = {'float32': torch.float32, 'float': torch.float32, 'float16': torch.float16, 'half': torch.float16}
    dtype_str = str(param.get('dtype', 'float32')).lower()
    dtype = dtype_map.get(dtype_str, torch.float32)
    b = int(param.get('batch', 2))
    h = int(param.get('input_h', 24))
    w = int(param.get('input_w', 32))
    c = int(param.get('channels', 8))
    n = int(param.get('num_rois', 8))
    pooled_h = int(param.get('pooled_h', 4))
    pooled_w = int(param.get('pooled_w', 4))
    spatial_scale = float(param.get('spatial_scale', 0.25))
    sampling_ratio = int(param.get('sampling_ratio', 0))
    aligned = parse_bool_param(param.get('aligned', True))
    clockwise = parse_bool_param(param.get('clockwise', False))
    x = torch.tensor(rng.standard_normal((b, h, w, c), dtype=np.float32), dtype=dtype)
    rois = torch.zeros((6, n), dtype=dtype)
    for j in range(n):
        bi = int(rng.integers(0, b))
        cx_feat = float(rng.uniform(w * 0.25, w * 0.75))
        cy_feat = float(rng.uniform(h * 0.25, h * 0.75))
        cx = cx_feat / max(spatial_scale, 1e-08)
        cy = cy_feat / max(spatial_scale, 1e-08)
        rw_feat = float(rng.uniform(2.0, min(10.0, 0.45 * w)))
        rh_feat = float(rng.uniform(2.0, min(10.0, 0.45 * h)))
        rw = rw_feat / max(spatial_scale, 1e-08)
        rh = rh_feat / max(spatial_scale, 1e-08)
        theta = float(rng.uniform(-math.pi / 3, math.pi / 3))
        rois[0, j] = float(bi)
        rois[1, j] = cx
        rois[2, j] = cy
        rois[3, j] = rw
        rois[4, j] = rh
        rois[5, j] = theta
    if device:
        x = x.to(device)
        rois = rois.to(device)
    return (x, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, aligned, clockwise)

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
    json_path = os.path.join(os.path.dirname(__file__), "RoiAlignRotated.json")
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
