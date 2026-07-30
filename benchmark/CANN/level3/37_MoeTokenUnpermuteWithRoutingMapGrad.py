"""Auto-generated benchmark file for MoeTokenUnpermuteWithRoutingMapGrad.

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

class Model(nn.Module):
    """PyTorch native reference implementation (golden model)."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, unpermuted_tokens_grad, out_index, permute_token_id, routing_map, permuted_tokens, probs, drop_and_pad, restore_shape):
        """
        Args:
            unpermuted_tokens_grad: [tokens_num, hidden_size] - gradient from forward output
            out_index: [total_length] - output position indices
            permute_token_id: [total_length] - token id for each position
            routing_map: optional [tokens_num, experts_num] bool/int8 mask
            permuted_tokens: optional [total_length, hidden_size] - forward permuted tokens
            probs: optional [tokens_num, experts_num] - expert probabilities
            drop_and_pad: bool - padded mode flag
            restore_shape: List[int] - shape info for padded mode
        Returns:
            permuted_tokens_grad: [out_length, hidden_size]
            probs_grad: optional [tokens_num, experts_num]
        """
        orig_dtype = unpermuted_tokens_grad.dtype
        tokens_num = unpermuted_tokens_grad.shape[0]
        hidden_size = unpermuted_tokens_grad.shape[1]
        total_length = out_index.shape[0]
        grad_f = unpermuted_tokens_grad.float()
        out_index_cpu = out_index.cpu().long()
        permute_token_id_cpu = permute_token_id.cpu().long()
        if probs is None:
            permuted_tokens_grad = torch.zeros(total_length, hidden_size, dtype=torch.float32, device=unpermuted_tokens_grad.device)
            for i in range(total_length):
                tid = permute_token_id_cpu[i].item()
                oi = out_index_cpu[i].item()
                permuted_tokens_grad[oi] = grad_f[tid]
            return (permuted_tokens_grad.to(orig_dtype), None)
        num_experts = probs.shape[1]
        probs_cpu = probs.float().cpu()
        permuted_tokens_f = permuted_tokens.float()
        permuted_tokens_cpu = permuted_tokens_f.cpu()
        permuted_tokens_grad = torch.zeros(total_length, hidden_size, dtype=torch.float32, device=unpermuted_tokens_grad.device)
        for i in range(total_length):
            tid = permute_token_id_cpu[i].item()
            oi = out_index_cpu[i].item()
            permuted_tokens_grad[oi] = grad_f[tid]
        if not drop_and_pad:
            topK = total_length // tokens_num
            permuted_probs_grad = permuted_tokens_grad * permuted_tokens_f
            probs_grad_expert_order = permuted_probs_grad.sum(dim=-1)
            probs_grad = torch.zeros(tokens_num, num_experts, dtype=torch.float32, device=unpermuted_tokens_grad.device)
            if routing_map is not None:
                routing_map_cpu = routing_map.cpu()
                idx = 0
                for t in range(tokens_num):
                    for e in range(num_experts):
                        if routing_map_cpu[t, e].item():
                            if idx < total_length:
                                probs_grad[t, e] = probs_grad_expert_order[idx].item()
                                idx += 1
            permuted_probs = []
            if routing_map is not None:
                routing_map_cpu = routing_map.cpu()
                for t in range(tokens_num):
                    for e in range(num_experts):
                        if routing_map_cpu[t, e].item():
                            permuted_probs.append(probs_cpu[t, e].item())
            if len(permuted_probs) > 0:
                permuted_probs_tensor = torch.tensor(permuted_probs, dtype=torch.float32, device=unpermuted_tokens_grad.device)
                permuted_tokens_grad = permuted_probs_tensor.unsqueeze(-1) * permuted_tokens_grad
        else:
            capacity = total_length // num_experts
            permuted_probs_grad = permuted_tokens_grad * permuted_tokens_f
            probs_grad_expert_order = permuted_probs_grad.sum(dim=-1)
            probs_grad = torch.zeros(tokens_num, num_experts, dtype=torch.float32, device=unpermuted_tokens_grad.device)
            for i in range(total_length):
                tid = permute_token_id_cpu[i].item()
                oi = out_index_cpu[i].item()
                expert_id = oi // capacity
                probs_grad[tid, expert_id] = probs_grad_expert_order[oi].item()
            for i in range(total_length):
                tid = permute_token_id_cpu[i].item()
                oi = out_index_cpu[i].item()
                expert_id = oi // capacity
                prob_val = probs_cpu[tid, expert_id].item()
                permuted_tokens_grad[oi] = prob_val * permuted_tokens_grad[oi]
        return (permuted_tokens_grad.to(orig_dtype), probs_grad.to(probs.dtype))

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np

def get_inputs(param, device=None):
    """Generate input tensors based on test case parameters."""
    num_tokens = param.get('num_tokens')
    num_experts = param.get('num_experts')
    top_k = param.get('top_k')
    hidden_size = param.get('hidden_size')
    dtype_str = param.get('dtype', 'float32')
    has_probs = param.get('has_probs', True)
    has_routing_map = param.get('has_routing_map', True)
    drop_and_pad = param.get('drop_and_pad', False)
    if dtype_str == 'float32':
        dtype = torch.float32
    elif dtype_str == 'float16':
        dtype = torch.float16
    elif dtype_str == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    if drop_and_pad:
        capacity = param.get('capacity', top_k)
        total_length = num_experts * capacity
    else:
        capacity = top_k
        total_length = num_tokens * top_k
    unpermuted_tokens_grad = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device) * 0.1
    if drop_and_pad:
        out_index_list = []
        permute_token_id_list = []
        for e in range(num_experts):
            token_ids = torch.randperm(num_tokens)[:capacity].tolist()
            for c in range(capacity):
                out_index_list.append(e * capacity + c)
                permute_token_id_list.append(token_ids[c])
        out_index = torch.tensor(out_index_list, dtype=torch.int32, device=device)
        permute_token_id = torch.tensor(permute_token_id_list, dtype=torch.int32, device=device)
    else:
        out_index_list = []
        permute_token_id_list = []
        permuted_offset = 0
        for t in range(num_tokens):
            for k in range(top_k):
                out_index_list.append(permuted_offset)
                permute_token_id_list.append(t)
                permuted_offset += 1
        out_index = torch.tensor(out_index_list, dtype=torch.int32, device=device)
        permute_token_id = torch.tensor(permute_token_id_list, dtype=torch.int32, device=device)
    if has_routing_map and has_probs:
        routing_map = torch.ones(num_tokens, num_experts, dtype=torch.bool, device=device)
    else:
        routing_map = None
    if has_probs:
        permuted_tokens = torch.randn(total_length, hidden_size, dtype=dtype, device=device) * 0.1
    else:
        permuted_tokens = None
    if has_probs:
        probs_dtype = dtype
        if dtype_str == 'bfloat16' and param.get('mixed_probs', False):
            probs_dtype = torch.float32
        probs = torch.rand(num_tokens, num_experts, dtype=probs_dtype, device=device) * 0.5 + 0.5
    else:
        probs = None
    restore_shape = [num_tokens]
    return (unpermuted_tokens_grad, out_index, permute_token_id, routing_map, permuted_tokens, probs, drop_and_pad, restore_shape)

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
    json_path = os.path.join(os.path.dirname(__file__), "MoeTokenUnpermuteWithRoutingMapGrad.json")
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
