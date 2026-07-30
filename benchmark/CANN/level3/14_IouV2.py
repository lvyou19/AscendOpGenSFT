"""Auto-generated benchmark file for IouV2.

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

def _overlap_reference(bboxes_nm4: torch.Tensor, gt_nm4: torch.Tensor, eps: float, mode: str) -> torch.Tensor:
    """
    与 IouV2 kernel 一致的 CPU 参考：面积在边长上加 eps 再相乘；
    交集边长为 relu(min(x2)+eps - max(x1)) 等形式。
    bboxes_nm4: (N_a, 4), gt_nm4: (N_b, 4)，xyxy。
    返回 (N_b, N_a)，与 GE inferShape 中 overlap[gt, bbox] 一致。
    """
    a = bboxes_nm4.unsqueeze(0).float()
    b = gt_nm4.unsqueeze(1).float()
    x1a, y1a, x2a, y2a = (a[..., 0], a[..., 1], a[..., 2], a[..., 3])
    x1b, y1b, x2b, y2b = (b[..., 0], b[..., 1], b[..., 2], b[..., 3])
    wa = x2a - x1a + eps
    ha = y2a - y1a + eps
    wb = x2b - x1b + eps
    hb = y2b - y1b + eps
    area_a = wa * ha
    area_b = wb * hb
    ix1 = torch.maximum(x1a, x1b)
    iy1 = torch.maximum(y1a, y1b)
    ix2 = torch.minimum(x2a, x2b)
    iy2 = torch.minimum(y2a, y2b)
    iw = torch.relu(ix2 + eps - ix1)
    ih = torch.relu(iy2 + eps - iy1)
    inter = iw * ih
    if mode == 'iof':
        denom = area_b.clamp_min(1e-45)
    else:
        denom = (area_a + area_b - inter).clamp_min(1e-45)
    return (inter / denom).to(bboxes_nm4.dtype)

class Model(nn.Module):
    """CPU 金标准：与 op_kernel 中 IOU/IOF 公式对齐。"""

    def __init__(self):
        super().__init__()

    def forward(self, bboxes: torch.Tensor, gtboxes: torch.Tensor, mode: str, eps: float, aligned: bool) -> torch.Tensor:
        mode = mode.lower() if isinstance(mode, str) else 'iou'
        if aligned:
            boxes_a = bboxes.t().contiguous()
            boxes_b = gtboxes.t().contiguous()
            o = _overlap_reference(boxes_a, boxes_b, eps, mode)
            return torch.diagonal(o).to(bboxes.dtype).unsqueeze(1)
        return _overlap_reference(bboxes, gtboxes, eps, mode)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import numpy as np
import torch

def parse_bool_param(param):
    if isinstance(param, str) and param.lower() in ['true', 't', '1']:
        return True
    if param == 1:
        return True
    return False

def _rand_boxes_xyxy(n: int, rng: np.random.Generator, lo=0.0, hi=10.0):
    """生成 n 个有效 xyxy 框（x2>x1, y2>y1）。"""
    boxes = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        x1, y1 = rng.uniform(lo, hi - 1, 2)
        x2 = rng.uniform(x1 + 0.1, hi)
        y2 = rng.uniform(y1 + 0.1, hi)
        boxes[i] = [x1, y1, x2, y2]
    return boxes

def _to_layout(boxes_nm4: torch.Tensor, aligned: bool) -> torch.Tensor:
    """[N,4] -> aligned 时 [4,N]，否则仍为 [N,4]。"""
    if aligned:
        return boxes_nm4.t().contiguous()
    return boxes_nm4

def get_inputs(param, device=None):
    rng = np.random.default_rng(7 + int(param.get('case_id', 0)))
    dtype_map = {'float32': torch.float32, 'float': torch.float32, 'float16': torch.float16, 'half': torch.float16, 'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16}
    dtype_str = str(param.get('dtype', 'float16')).lower()
    dtype = dtype_map.get(dtype_str, torch.float16)
    aligned = parse_bool_param(param.get('aligned', False))
    mode = str(param.get('mode', 'iou')).lower()
    if mode not in ('iou', 'iof'):
        mode = 'iou'
    eps = float(param.get('eps', 1.0))
    n = int(param.get('n', 32))
    n_gt = int(param.get('n_gt', 8))
    n_bbox = int(param.get('n_bbox', 12))
    if aligned:
        boxes_a = torch.tensor(_rand_boxes_xyxy(n, rng), dtype=dtype)
        boxes_b = torch.tensor(_rand_boxes_xyxy(n, rng), dtype=dtype)
        bboxes = _to_layout(boxes_a, True)
        gtboxes = _to_layout(boxes_b, True)
    else:
        boxes_a = torch.tensor(_rand_boxes_xyxy(n_bbox, rng), dtype=dtype)
        boxes_b = torch.tensor(_rand_boxes_xyxy(n_gt, rng), dtype=dtype)
        bboxes = boxes_a
        gtboxes = boxes_b
    if device:
        bboxes = bboxes.to(device)
        gtboxes = gtboxes.to(device)
    return (bboxes, gtboxes, mode, eps, aligned)

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
    json_path = os.path.join(os.path.dirname(__file__), "IouV2.json")
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
