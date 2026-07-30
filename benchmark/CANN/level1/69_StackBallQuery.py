"""Auto-generated benchmark file for StackBallQuery.

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

def stack_ball_query_cpu(xyz: torch.Tensor, center_xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor, center_xyz_batch_cnt: torch.Tensor, max_radius: float, sample_num: int) -> torch.Tensor:
    """
    xyz: [3, N] planar；center_xyz: [M, 3]。
    与两条 kernel 分支（FP32 / FP16）在「顺序扫描 + 平方距离与 max_radius^2 比较」语义上一致；
    距离在 float32 中计算（FP16 输入先提升），避免半精度边界与参考不一致。
    无命中：首元素 -1，其余填 0；有命中：输出 batch 内局部下标，不足槽位用第一个命中下标填充。
    """
    m = center_xyz.shape[0]
    max_r2 = max_radius * max_radius
    xb = xyz_batch_cnt.detach().cpu().tolist()
    cb = center_xyz_batch_cnt.detach().cpu().tolist()
    batch_size = len(xb)

    def center_batch(global_idx: int):
        cum = 0
        for b in range(batch_size):
            if global_idx < cum + cb[b]:
                return (b, global_idx - cum)
            cum += cb[b]
        return (batch_size - 1, 0)

    def xyz_off(b: int):
        return sum(xb[:b])
    out = torch.empty(m * sample_num, device=xyz.device, dtype=torch.int32)
    cx = center_xyz[:, 0].to(torch.float32)
    cy = center_xyz[:, 1].to(torch.float32)
    cz = center_xyz[:, 2].to(torch.float32)
    px = xyz[0].to(torch.float32)
    py = xyz[1].to(torch.float32)
    pz = xyz[2].to(torch.float32)
    for mi in range(m):
        b, _ = center_batch(mi)
        cxv, cyv, czv = (cx[mi].item(), cy[mi].item(), cz[mi].item())
        off = xyz_off(b)
        cnt = xb[b]
        collected = []
        for i in range(cnt):
            dx = px[off + i].item() - cxv
            dy = py[off + i].item() - cyv
            dz = pz[off + i].item() - czv
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < max_r2:
                collected.append(i)
                if len(collected) >= sample_num:
                    break
        if len(collected) == 0:
            out[mi * sample_num] = -1
            for s in range(1, sample_num):
                out[mi * sample_num + s] = 0
        else:
            fr = collected[0]
            for s in range(sample_num):
                if s < len(collected):
                    out[mi * sample_num + s] = collected[s]
                else:
                    out[mi * sample_num + s] = fr
    return out

class Model(nn.Module):

    def __init__(self, max_radius: float, sample_num: int):
        super().__init__()
        self.max_radius = max_radius
        self.sample_num = sample_num

    def forward(self, xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt):
        return stack_ball_query_cpu(xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt, self.max_radius, self.sample_num)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def _build_xyz_planar(batch_xyz_counts, dtype, device, base_sep=10.0):
    """xyz: [3, N] planar layout (all x, all y, all z), 与 kernel 一致。"""
    counts = [int(batch_xyz_counts[i]) for i in range(len(batch_xyz_counts))]
    n = sum(counts)
    xyz = torch.zeros(3, n, device=device, dtype=dtype)
    off = 0
    for b, cnt in enumerate(counts):
        for i in range(cnt):
            xyz[0, off + i] = b * base_sep + i * 0.15
            xyz[1, off + i] = 0.02 * i
            xyz[2, off + i] = 0.01 * (b + 1)
        off += cnt
    return xyz

def _build_centers_stacked(batch_center_counts, dtype, device, base_sep=10.0):
    """center_xyz: [M, 3]，与 kernel 交错存储一致。"""
    counts = [int(batch_center_counts[i]) for i in range(len(batch_center_counts))]
    m = sum(counts)
    c = torch.zeros(m, 3, device=device, dtype=dtype)
    off = 0
    for b, cnt in enumerate(counts):
        for i in range(cnt):
            c[off + i, 0] = b * base_sep + 0.05 + i * 0.1
            c[off + i, 1] = 0.0
            c[off + i, 2] = 0.01 * (b + 1)
        off += cnt
    return c

def get_inputs(param, device=None):
    dtype_str = param.get('dtype_xyz', 'float32')
    dtype = getattr(torch, dtype_str)
    bx0 = int(param['batch_xyz_0'])
    bx1 = int(param['batch_xyz_1'])
    bc0 = int(param['batch_center_0'])
    bc1 = int(param['batch_center_1'])
    xyz = _build_xyz_planar([bx0, bx1], dtype, device)
    center_xyz = _build_centers_stacked([bc0, bc1], dtype, device)
    xyz_batch_cnt = torch.tensor([bx0, bx1], device=device, dtype=torch.int32)
    center_xyz_batch_cnt = torch.tensor([bc0, bc1], device=device, dtype=torch.int32)
    return [xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt]

def get_init_inputs_per_case(param, device=None):
    max_radius = float(param.get('max_radius', 2.0))
    sample_num = int(param.get('sample_num', 4))
    return [max_radius, sample_num]


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
    json_path = os.path.join(os.path.dirname(__file__), "StackBallQuery.json")
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
