"""Auto-generated benchmark file for EmbeddingBag.

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
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    实现EmbeddingBag算子功能的模型（PyTorch标杆）。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, weight: torch.Tensor, indices: torch.Tensor, offsets: torch.Tensor, per_sample_weights: Optional[torch.Tensor], mode: str, scale_grad_by_freq: bool, sparse: bool, include_last_offset: bool, padding_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        实现EmbeddingBag算子功能。

        Args:
            weight: 嵌入权重矩阵 (num_embeddings, embedding_dim)
            indices: 索引张量
            offsets: 偏移量张量
            per_sample_weights: 可选的每样本权重
            mode: 聚合模式 ('sum', 'mean', 'max')
            scale_grad_by_freq: 是否按频率缩放梯度
            sparse: 是否使用稀疏梯度
            include_last_offset: offsets是否包含最后一个偏移
            padding_idx: 填充索引

        Returns:
            (y, offset2bag, bag_size, max_indices)
        """
        num_bags = offsets.size(0)
        if include_last_offset:
            num_bags -= 1
        y, offset2bag, bag_size, max_indices = torch.ops.aten._embedding_bag_forward_only(weight, indices, offsets, scale_grad_by_freq, 0 if mode == 'sum' else 1 if mode == 'mean' else 2, sparse, per_sample_weights, include_last_offset, padding_idx)
        return (y, offset2bag, bag_size, max_indices)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
from typing import Optional

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    num_embeddings = int(param.get('num_embeddings', 10))
    embedding_dim = int(param.get('embedding_dim', 4))
    num_indices = int(param.get('num_indices', 8))
    num_bags = int(param.get('num_bags', 2))
    dtype_str = param.get('dtype', 'float32')
    mode = param.get('mode', 'mean')
    include_last_offset = param.get('include_last_offset', 'False') == 'True'
    padding_idx = int(param.get('padding_idx', -1))
    has_per_sample_weights = param.get('has_per_sample_weights', 'False') == 'True'
    scale_grad_by_freq = param.get('scale_grad_by_freq', 'False') == 'True'
    sparse = param.get('sparse', 'False') == 'True'
    dtype = getattr(torch, dtype_str)
    weight = torch.randn(num_embeddings, embedding_dim, dtype=dtype, device=device)
    indices = torch.randint(0, num_embeddings, (num_indices,), dtype=torch.int32, device=device)
    if include_last_offset:
        num_segments = num_bags
        segment_size = max(1, num_indices // num_segments)
        offsets_list = [0]
        for i in range(1, num_segments):
            next_offset = min(offsets_list[-1] + segment_size, num_indices - (num_segments - i))
            offsets_list.append(next_offset)
        offsets_list.append(num_indices)
        offsets = torch.tensor(offsets_list, dtype=torch.int32, device=device)
    else:
        segment_size = max(1, num_indices // num_bags)
        offsets_list = [0]
        for i in range(1, num_bags):
            next_offset = min(offsets_list[-1] + segment_size, num_indices - (num_bags - i))
            offsets_list.append(next_offset)
        offsets = torch.tensor(offsets_list, dtype=torch.int32, device=device)
    if has_per_sample_weights:
        per_sample_weights = torch.randn(num_indices, dtype=dtype, device=device)
    else:
        per_sample_weights = None
    return (weight, indices, offsets, per_sample_weights, mode, scale_grad_by_freq, sparse, include_last_offset, padding_idx)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for EmbeddingBag.

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
    json_path = os.path.join(os.path.dirname(__file__), "EmbeddingBag.json")
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
