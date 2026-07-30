"""Auto-generated benchmark file for AdvanceStep.

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
from typing import List
import torch
import torch.nn as nn

class Model(nn.Module):
    """AdvanceStep 算子的 PyTorch 参考实现（golden model）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tokens: torch.Tensor, sampled_token_ids: torch.Tensor, input_positions: torch.Tensor, seq_lens: torch.Tensor, slot_mapping: torch.Tensor, block_tables: torch.Tensor, num_seqs: int, num_queries: int, block_size: int) -> List[torch.Tensor]:
        input_tokens = input_tokens.clone()
        input_positions = input_positions.clone()
        seq_lens = seq_lens.clone()
        slot_mapping = slot_mapping.clone()
        n_pad = num_seqs - num_queries
        total_core_num = 48
        if n_pad > 0:
            for i in range(0, n_pad, total_core_num):
                input_tokens[num_queries + i] = 0
                input_positions[num_queries + i] = 0
                slot_mapping[num_queries + i] = -1
        for index in range(num_queries):
            input_tokens[index] = sampled_token_ids[index]
            seq_len = seq_lens[index].item()
            next_seq_len = seq_len + 1
            next_input_pos = next_seq_len - 1
            seq_lens[index] = next_seq_len
            input_positions[index] = next_input_pos
            block_index = next_input_pos // block_size
            block_offset = next_input_pos % block_size
            block_tables_flat = block_tables.flatten()
            slot_num = (block_tables_flat[block_index].item() + block_tables.shape[1] * index) * block_size + block_offset
            slot_mapping[index] = slot_num
        return [input_tokens, input_positions, seq_lens, slot_mapping]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """根据 test_cases.csv 行生成输入张量。返回与 Model.forward() 签名匹配的元组。"""
    num_seqs = int(param['num_seqs'])
    num_queries = int(param['num_queries'])
    block_size = int(param['block_size'])
    max_blocks_per_seq = int(param.get('max_blocks_per_seq', '8'))
    input_tokens = torch.randint(0, 1000, (num_seqs,), dtype=torch.int64)
    sampled_token_ids = torch.randint(0, 1000, (num_queries, 1), dtype=torch.int64)
    input_positions = torch.randint(0, max_blocks_per_seq * block_size, (num_seqs,), dtype=torch.int64)
    seq_lens = torch.randint(1, max_blocks_per_seq * block_size, (num_seqs,), dtype=torch.int64)
    slot_mapping = torch.full((num_seqs,), -1, dtype=torch.int64)
    block_tables = torch.randint(0, 1024, (num_seqs, max_blocks_per_seq), dtype=torch.int64)
    if device:
        input_tokens = input_tokens.to(device)
        sampled_token_ids = sampled_token_ids.to(device)
        input_positions = input_positions.to(device)
        seq_lens = seq_lens.to(device)
        slot_mapping = slot_mapping.to(device)
        block_tables = block_tables.to(device)
    return (input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, num_seqs, num_queries, block_size)

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
    json_path = os.path.join(os.path.dirname(__file__), "AdvanceStep.json")
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
