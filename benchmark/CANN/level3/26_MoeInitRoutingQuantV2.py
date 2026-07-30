"""Auto-generated benchmark file for MoeInitRoutingQuantV2.

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
import numpy as np

def adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
    count = 0
    last = sorted_expert_idx[0]
    for i, val in enumerate(sorted_expert_idx):
        if last != val:
            count = 1
            last = val
        else:
            count += 1
            if count > capacity:
                sorted_expert_idx[i] = -1
                sorted_row_idx[i] = -1

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, expert_idx, scale_optional, offset_optional, active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag, quant_mode):
        input_x = x.to(torch.float32).numpy()
        expert_idx = expert_idx.numpy()
        scale = scale_optional.numpy()
        offset_t = offset_optional.numpy()
        num_rows = input_x.shape[0]
        hidden_size = input_x.shape[-1]
        k = expert_idx.shape[-1]
        flat_expert = expert_idx.reshape(-1)
        sorted_row_idx = np.argsort(flat_expert, kind='stable')
        sorted_expert_idx = flat_expert[sorted_row_idx].astype(np.int32, copy=True)
        if drop_pad_mode == 1 and expert_num <= 0:
            return
        expert_tokens_count_or_cumsum = None
        expert_tokens_before_capacity = None
        expert_idx_hist = np.bincount(sorted_expert_idx, minlength=expert_num).astype(np.int64)
        expert_token_idx = np.cumsum(expert_idx_hist)
        if drop_pad_mode == 1 and expert_tokens_before_capacity_flag:
            expert_tokens_before_capacity = expert_idx_hist.astype('int32')
        if drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 1:
            expert_tokens_count_or_cumsum = expert_token_idx.astype('int32')
        elif drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 2:
            expert_tokens_count_or_cumsum = expert_idx_hist.astype('int32')
        slot_filled = None
        if drop_pad_mode == 0:
            expanded_row_idx = np.zeros(sorted_row_idx.shape, dtype=np.int32)
            expanded_row_idx[sorted_row_idx] = np.arange(sorted_row_idx.shape[-1], dtype=np.int32)
            if active_num == 0:
                active_num = num_rows * k
            else:
                active_num = min(active_num, num_rows * k)
            expanded_x = input_x[sorted_row_idx[:active_num] // k, :]
        else:
            adapter_capacity(sorted_row_idx, sorted_expert_idx, expert_capacity)
            sort_row_tmp = np.full(expert_num * expert_capacity, -1, dtype=int)
            offset = 0
            lastExpertId = 0
            for i, val in enumerate(sorted_row_idx):
                if val != -1:
                    if lastExpertId != sorted_expert_idx[i]:
                        offset = 0
                        lastExpertId = sorted_expert_idx[i]
                    sort_row_tmp[sorted_expert_idx[i] * expert_capacity + offset] = sorted_row_idx[i]
                    offset += 1
            expanded_row_idx = np.full(sorted_row_idx.shape, -1)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_row_idx[val] = i
            expanded_x = np.full((expert_num * expert_capacity, hidden_size), 0, dtype=input_x.dtype)
            slot_filled_flat = np.zeros(expert_num * expert_capacity, dtype=bool)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_x[i] = input_x[val // k]
                    slot_filled_flat[i] = True
            expanded_x = expanded_x.reshape(expert_num, expert_capacity, hidden_size)
            slot_filled = slot_filled_flat.reshape(expert_num, expert_capacity)
        if expert_tokens_count_or_cumsum is None:
            expert_tokens_count_or_cumsum = torch.tensor([])
        else:
            expert_tokens_count_or_cumsum = torch.from_numpy(expert_tokens_count_or_cumsum)
        ds = torch.tensor([])
        if quant_mode == 0:
            expanded_x = np.clip(expanded_x, np.finfo(np.float16).min, np.finfo(np.float16).max)
            expanded_x = expanded_x.astype(np.float16)
            scale_v = np.clip(scale[0], np.finfo(np.float16).min, np.finfo(np.float16).max)
            offset_v = offset_t.astype('float16')
            rr = expanded_x * scale_v + offset_v
            roundd = np.rint(rr)
            roundd = np.clip(roundd, -128, 127)
            roundd = roundd.astype('int8')
            if slot_filled is not None:
                mask = slot_filled[:, :, np.newaxis]
                expanded_x = np.where(mask, roundd, np.int8(0))
            else:
                expanded_x = roundd
        else:
            xf = expanded_x.astype('float32')
            xa = np.abs(xf)
            xm = np.max(xa, axis=-1, keepdims=True)
            ds_arr = xm / 127.0
            q = np.round(np.where(xm > 0, xf / np.maximum(ds_arr, 1e-30), 0.0)).astype('int8')
            if slot_filled is not None:
                mask = slot_filled[:, :, np.newaxis]
                expanded_x = np.where(mask, q, np.int8(0))
            else:
                expanded_x = q
            ds = torch.from_numpy(ds_arr).reshape(-1)
        t_expanded_x = torch.from_numpy(expanded_x)
        if expert_tokens_before_capacity is None:
            expert_tokens_before_capacity = torch.tensor([])
        else:
            expert_tokens_before_capacity = torch.from_numpy(expert_tokens_before_capacity)
        return (t_expanded_x, torch.from_numpy(expanded_row_idx.astype('int32')), expert_tokens_count_or_cumsum, expert_tokens_before_capacity, ds)

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import numpy as np

def get_inputs(param, device=None):
    x_shape = eval(param.get('x_shape', [36, 32]))
    exprt_idx_shape = eval(param.get('exprt_idx_shape', [36, 1]))
    active_num = int(param.get('active_num', 166))
    expert_capacity = int(param.get('expert_capacity', 4))
    expert_num = int(param.get('expert_num', 361))
    drop_pad_mode = int(param.get('drop_pad_mode', 1))
    expert_tokens_count_or_cumsum_flag = int(param.get('expert_tokens_count_or_cumsum_flag', 2))
    expert_tokens_before_capacity_flag = int(param.get('expert_tokens_before_capacity_flag', 0))
    quant_mode = int(param.get('quant_mode', 0))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    x = torch.randn(x_shape, dtype=dtype).to(device)
    expert_idx = torch.randint(1, 2, size=exprt_idx_shape, dtype=torch.int32).to(device)
    scale_optional = None
    offset_optional = None
    if quant_mode == 0:
        scale_optional = torch.tensor([0.6], dtype=torch.float32).to(device)
        offset_optional = torch.tensor([0.6], dtype=torch.float32).to(device)
    elif quant_mode == 1:
        if np.random.random() > 0.5:
            scale_optional = torch.randn([1, x_shape[-1]], dtype=torch.float32).to(device)
    return (x, expert_idx, scale_optional, offset_optional, active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag, quant_mode)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for MoeInitRoutingV3.

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
    json_path = os.path.join(os.path.dirname(__file__), "MoeInitRoutingQuantV2.json")
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
