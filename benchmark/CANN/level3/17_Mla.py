"""Auto-generated benchmark file for Mla.

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
from typing import List, Optional
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def _group_matmul(self, q, k_or_v):
        """Helper for Grouped-Query Attention matmul."""
        num_heads, q_seqlen, _ = q.shape
        kv_heads = k_or_v.shape[0]
        if num_heads == kv_heads:
            return torch.matmul(q, k_or_v)
        group_num = num_heads // kv_heads
        q = q.view(kv_heads, group_num, q_seqlen, -1)
        if k_or_v.dim() == 3:
            k = k_or_v.unsqueeze(1)
            score = torch.matmul(q, k)
        else:
            v = k_or_v.unsqueeze(1)
            score = torch.matmul(q, v)
        return score.view(num_heads, q_seqlen, -1)

    def _ref_masked_attention(self, query, key, value, scale, mask=None):
        """Performs a single scaled dot-product attention operation."""
        q = query.permute(1, 0, 2)
        k = key.permute(1, 2, 0)
        v = value.permute(1, 0, 2)
        scores = self._group_matmul(q, k) * scale
        if mask is not None:
            scores += mask
        attn = F.softmax(scores, dim=-1)
        output = self._group_matmul(attn, v)
        return output.permute(1, 0, 2)

    def forward(self, query_nope, query_rope, kv_nope_cache, kv_rope_cache, block_tables, q_seqlen_list, k_seqlen_list, mask=None):
        """
        Forward pass for the reference Paged Attention model.
        It computes the result in float32 for high precision.
        """
        query = torch.concat([query_nope, query_rope], dim=-1)
        key_cache = torch.concat([kv_nope_cache, kv_rope_cache], dim=-1)
        output_shape = (query.shape[0], query.shape[1], kv_nope_cache.shape[3])
        final_output = torch.empty(output_shape, dtype=torch.float32, device=query_nope.device)
        block_size = kv_nope_cache.shape[1]
        cu_q_seqlen = 0
        for i in range(len(q_seqlen_list)):
            q_len = q_seqlen_list[i]
            k_len = k_seqlen_list[i]
            q_current = query[cu_q_seqlen:cu_q_seqlen + q_len]
            k_list, v_list = ([], [])
            for j in range(k_len):
                block_idx = j // block_size
                block_offset = j % block_size
                block_number = block_tables[i, block_idx].item()
                k_list.append(key_cache[block_number, block_offset])
                v_list.append(kv_nope_cache[block_number, block_offset])
            keys = torch.stack(k_list, dim=0)
            values = torch.stack(v_list, dim=0)
            scale = 1.0 / keys.shape[-1] ** 0.5
            current_mask = mask[cu_q_seqlen:cu_q_seqlen + q_len, :k_len] if mask is not None else None
            out = self._ref_masked_attention(q_current.to(torch.float32), keys.to(torch.float32), values.to(torch.float32), scale, current_mask.to(torch.float32) if current_mask is not None else None)
            final_output[cu_q_seqlen:cu_q_seqlen + q_len] = out
            cu_q_seqlen += q_len
        return final_output

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the Paged Attention model based on parameters.

    Args:
        param (dict): Parameters from a pandas DataFrame row.
        device (str, optional): The device to place tensors on. Defaults to None.

    Returns:
        tuple: A tuple of input tensors (query, key_cache, value_cache,
               block_tables, q_seqlen_list, k_seqlen_list, mask).
    """
    batch_size = int(param.get('batch'))
    q_seqlen = int(param.get('q_seqlen'))
    kv_seqlen = int(param.get('kv_seqlen'))
    num_heads = int(param.get('num_heads'))
    kv_heads = int(param.get('kv_heads'))
    head_size = int(param.get('head_size'))
    head_size_rope = int(param.get('head_size_rope'))
    num_blocks = int(param.get('num_blocks'))
    block_size = int(param.get('block_size'))
    mask_type = 0
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    assert batch_size * kv_seqlen <= num_blocks * block_size, '[ERROR] the number of K and V tokens is too big to fit in the paged cache.'
    assert block_size == 128, '[ERROR] blockSize != 128 is not supported.'
    assert q_seqlen <= 4, '[ERROR] q_seqlen > 4 is not supported.'
    q_seqlen_list = [q_seqlen] * batch_size
    k_seqlen_list = [kv_seqlen] * batch_size
    num_tokens = sum(q_seqlen_list)
    head_size_qk = head_size + head_size_rope
    query = torch.rand(num_tokens, num_heads, head_size_qk, device=device, dtype=dtype) * 2 - 1
    query_nope = query[:, :, :head_size]
    query_rope = query[:, :, -head_size_rope:]
    key_cache = torch.rand(num_blocks, block_size, kv_heads, head_size_qk, device=device, dtype=dtype) * 2 - 1
    kv_nope_cache = key_cache[:, :, :, :head_size]
    kv_rope_cache = key_cache[:, :, :, -head_size_rope:]
    max_k_seqlen = max(k_seqlen_list)
    max_num_blocks_per_seq = (max_k_seqlen + block_size - 1) // block_size
    block_tables_list = []
    for i in range(batch_size):
        block_table = torch.arange(start=i * max_num_blocks_per_seq, end=(i + 1) * max_num_blocks_per_seq, dtype=torch.int32, device=device)
        block_tables_list.append(block_table)
    block_tables = torch.stack(block_tables_list)
    mask = None
    if mask_type == 1:
        pre_mask_factor = -10000.0
        mask = torch.zeros(num_tokens, max_k_seqlen, device=device, dtype=dtype)
        pre_qseqlen = 0
        for i in range(batch_size):
            qlen = q_seqlen_list[i]
            klen = k_seqlen_list[i]
            causal_mask = torch.triu(torch.ones(qlen, qlen, device=device, dtype=dtype), diagonal=1) * pre_mask_factor
            if klen >= qlen:
                mask[pre_qseqlen:pre_qseqlen + qlen, klen - qlen:klen] = causal_mask
            pre_qseqlen += qlen
    return (query_nope, query_rope, kv_nope_cache, kv_rope_cache, block_tables, q_seqlen_list, k_seqlen_list, mask)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for sinh.

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
    json_path = os.path.join(os.path.dirname(__file__), "Mla.json")
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
