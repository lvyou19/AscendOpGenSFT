"""Auto-generated benchmark file for CrossEntropyLossGrad.

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

    def __init__(self, weight: Optional[torch.Tensor], ignore_index: int, label_smoothing: float, reduction: str, lse_square_scale_for_zloss: float):
        super(Model, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.lse_square_scale_for_zloss = lse_square_scale_for_zloss

    def forward(self, grad_loss: torch.Tensor, log_softmax: torch.Tensor, target: torch.Tensor, grad_zloss: Optional[torch.Tensor], lse_for_zloss: Optional[torch.Tensor]) -> List[torch.Tensor]:
        log_softmax_fp32 = log_softmax.to(torch.float32)
        grad_loss_fp32 = grad_loss.to(torch.float32)
        weight_fp32 = self.weight.to(torch.float32) if self.weight is not None else torch.ones(log_softmax.size(-1), dtype=torch.float32, device=log_softmax.device)
        target_fp32 = target.to(torch.int64)
        batch_size, num_classes = log_softmax_fp32.shape
        weight_yn = torch.gather(weight_fp32, 0, target_fp32)
        if self.ignore_index >= 0:
            ignore_mask = (target_fp32 != self.ignore_index).float()
        else:
            ignore_mask = torch.ones(batch_size, dtype=torch.float32, device=log_softmax.device)
        if self.reduction == 'mean':
            mean_out_grad = grad_loss_fp32 * (1.0 - self.label_smoothing)
            weight_sum = torch.sum(weight_yn * ignore_mask)
            loss_out_grad = mean_out_grad / (weight_sum + 1e-12)
            smooth_loss_grad = grad_loss_fp32 * self.label_smoothing / num_classes / (weight_sum + 1e-12)
            loss_out_grad = loss_out_grad.unsqueeze(-1)
            smooth_loss_grad = smooth_loss_grad.unsqueeze(-1)
        elif self.reduction == 'sum':
            sum_out_grad = grad_loss_fp32 * (1.0 - self.label_smoothing)
            loss_out_grad = sum_out_grad.unsqueeze(-1)
            smooth_loss_grad = grad_loss_fp32 * self.label_smoothing / num_classes
            smooth_loss_grad = smooth_loss_grad.unsqueeze(-1)
        else:
            none_out_grad = grad_loss_fp32 * (1.0 - self.label_smoothing)
            loss_out_grad = none_out_grad
            smooth_loss_grad = grad_loss_fp32 * self.label_smoothing / num_classes
        loss_out_grad = loss_out_grad * ignore_mask
        smooth_loss_grad = smooth_loss_grad * ignore_mask
        nll_loss_grad = loss_out_grad * weight_yn
        log_softmax_probs_grad_loss_out_sub_part = torch.exp(log_softmax_fp32) * nll_loss_grad.unsqueeze(-1)
        predictions_grad_loss_out = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=log_softmax.device)
        predictions_grad_loss_out.scatter_(1, target_fp32.unsqueeze(-1), nll_loss_grad.unsqueeze(-1))
        grad_input = log_softmax_probs_grad_loss_out_sub_part - predictions_grad_loss_out
        if self.label_smoothing > 0:
            smooth_grad = smooth_loss_grad.unsqueeze(-1) * torch.ones_like(log_softmax_fp32)
            grad_input += smooth_grad
        return [grad_input.to(log_softmax.dtype)]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成 CrossEntropyLossGrad 模型的输入张量列表和标量。
    """
    batch_size = param.get('batch', 1)
    num_classes = param.get('num_classes', 1000)
    input_dtype_str = param.get('input_dtype', 'float32')
    input_dtype = getattr(torch, input_dtype_str)
    target_dtype_str = param.get('target_dtype', 'int64')
    target_dtype = getattr(torch, target_dtype_str)
    reduction = param.get('reduction', 'mean')
    if reduction == 'none':
        grad_loss = torch.rand((batch_size,), device=device, dtype=input_dtype)
    else:
        grad_loss = torch.rand((), device=device, dtype=input_dtype)
    if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
        random_logits = torch.rand([batch_size, num_classes], device=device, dtype=input_dtype) * 2.0 - 1.0
    else:
        random_logits = torch.randn([batch_size, num_classes], device=device, dtype=input_dtype) * 2.0
    log_prob = torch.log_softmax(random_logits, dim=-1)
    target = torch.randint(low=0, high=num_classes, size=(batch_size,), device=device, dtype=target_dtype)
    grad_zloss_type = param.get('grad_zloss_type', 'None')
    grad_zloss = None
    if grad_zloss_type == 'present':
        grad_zloss = torch.rand((1,), device=device, dtype=input_dtype)
    lse_for_zloss_type = param.get('lse_for_zloss_type', 'None')
    lse_for_zloss = None
    if lse_for_zloss_type == 'present':
        lse_for_zloss = torch.rand((batch_size,), device=device, dtype=input_dtype) + 1.0
    return (grad_loss, log_prob, target, grad_zloss, lse_for_zloss)

def get_init_inputs_per_case(param, device=None):
    num_classes = param.get('num_classes', 1000)
    weight_type = param.get('weight_type', 'present')
    weight_dtype_str = param.get('weight_dtype', 'float32')
    weight_dtype = getattr(torch, weight_dtype_str)
    ignore_index = param.get('ignore_index', -100)
    label_smoothing = float(param.get('label_smoothing', 0.0))
    reduction = param.get('reduction', 'mean')
    lse_square_scale_for_zloss = float(param.get('lse_square_scale_for_zloss', 0.0))
    weight = None
    if weight_type == 'present':
        weight = torch.rand(num_classes, device=device, dtype=weight_dtype)
    return [weight, ignore_index, label_smoothing, reduction, lse_square_scale_for_zloss]


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
    json_path = os.path.join(os.path.dirname(__file__), "CrossEntropyLossGrad.json")
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
