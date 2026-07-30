"""Auto-generated benchmark file for NsaCompressGrad.

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
NsaCompressGrad：CPU 金标准通过对可微的 forward（与 NsaCompress 参考同构的树形规约）做 autograd 得到 inputGrad / weightGrad。
"""
import torch
import torch.nn as nn

def _compress_out_rows(act_cumsum, compress_block_size: int, compress_stride: int) -> int:
    pre = 0
    total = 0
    for end in act_cumsum:
        cur = end - pre
        if cur >= compress_block_size:
            total += (cur - compress_block_size + compress_stride) // compress_stride
        pre = end
    return total

def _ceil_power2_u32(n: int) -> int:
    if n <= 1:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1

def reduce_block_like_torch(mul_rows: torch.Tensor) -> torch.Tensor:
    """与 NsaCompress `reduce_block_like_kernel` 同构、可反传。"""
    x = mul_rows
    l_len, dim = x.shape
    if l_len == 1:
        return x[0]
    align = _ceil_power2_u32(l_len) // 2
    add_n = l_len - align
    new = x.clone()
    new[0:add_n] = x[0:add_n] + x[align:align + add_n]
    x = new
    while align > 1:
        align = align >> 1
        new = x.clone()
        new[0:align] = x[0:align] + x[align:2 * align]
        x = new
    return x[0]

def nsa_compress_forward_torch(input_tensor: torch.Tensor, weight: torch.Tensor, act_seq_len_cumsum: torch.Tensor, compress_block_size: int, compress_stride: int) -> torch.Tensor:
    """可微 forward（float32），与 `NsaCompress` 参考语义一致。"""
    t, n, d = input_tensor.shape
    l_blk, n_w = weight.shape
    assert l_blk == compress_block_size and n_w == n
    cum = act_seq_len_cumsum.detach().cpu().tolist()
    m = _compress_out_rows(cum, compress_block_size, compress_stride)
    out_rows = []
    inp_f = input_tensor
    w_f = weight
    pre = 0
    flat_dim = n * d
    for end in cum:
        seg = inp_f[pre:end]
        t_seg = seg.shape[0]
        k = 0
        while k * compress_stride + compress_block_size <= t_seg:
            s = k * compress_stride
            block = seg[s:s + compress_block_size]
            mul = block * w_f.unsqueeze(-1)
            mul_flat = mul.reshape(compress_block_size, flat_dim)
            red = reduce_block_like_torch(mul_flat)
            out_rows.append(red.view(n, d))
            k += 1
        pre = end
    assert len(out_rows) == m
    return torch.stack(out_rows, dim=0)

def nsa_compress_grad_reference(output_grad: torch.Tensor, input_tensor: torch.Tensor, weight: torch.Tensor, act_seq_len_cumsum: torch.Tensor, compress_block_size: int, compress_stride: int, act_seq_len_type: int):
    del act_seq_len_type
    inp = input_tensor.float().detach().clone().requires_grad_(True)
    w = weight.float().detach().clone().requires_grad_(True)
    out = nsa_compress_forward_torch(inp, w, act_seq_len_cumsum, compress_block_size, compress_stride)
    loss = (out * output_grad.float()).sum()
    gi, gw = torch.autograd.grad(loss, (inp, w), retain_graph=False, create_graph=False)
    return (gi.to(input_tensor.dtype), gw.to(weight.dtype))

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output_grad: torch.Tensor, input_tensor: torch.Tensor, weight: torch.Tensor, act_seq_len_cumsum: torch.Tensor, compress_block_size: int, compress_stride: int, act_seq_len_type: int):
        ig, wg = nsa_compress_grad_reference(output_grad, input_tensor, weight, act_seq_len_cumsum, compress_block_size, compress_stride, act_seq_len_type)
        return [ig, wg]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import ast
import torch

def _compress_out_rows(act_cumsum, compress_block_size: int, compress_stride: int) -> int:
    pre = 0
    total = 0
    for end in act_cumsum:
        cur = end - pre
        if cur >= compress_block_size:
            total += (cur - compress_block_size + compress_stride) // compress_stride
        pre = end
    return total

def get_init_inputs_per_case(param, device=None):
    return []

def _parse_list(s):
    if isinstance(s, list):
        return s
    return ast.literal_eval(str(s).strip())

def get_inputs(row, device=None):
    """列与 NsaCompress 一致；额外构造与 forward 输出同形状的 output_grad。"""
    dtype = getattr(torch, str(row['dtype']).strip())
    t = int(row['T'])
    n = int(row['N'])
    d = int(row['D'])
    cbs = int(row['compress_block_size'])
    cs = int(row['compress_stride'])
    cum = [int(x) for x in _parse_list(row['act_seq_len_cumsum'])]
    cst = int(row.get('act_seq_len_type', 0))
    torch.manual_seed(int(row.get('seed', 42)))
    input_tensor = torch.randn(t, n, d, device=device, dtype=dtype)
    weight = torch.randn(cbs, n, device=device, dtype=dtype)
    act_seq = torch.tensor(cum, device=device, dtype=torch.int64)
    m = _compress_out_rows(cum, cbs, cs)
    output_grad = torch.randn(m, n, d, device=device, dtype=dtype)
    return (output_grad, input_tensor, weight, act_seq, cbs, cs, cst)


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
    json_path = os.path.join(os.path.dirname(__file__), "NsaCompressGrad.json")
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
