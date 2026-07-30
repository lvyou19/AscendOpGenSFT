"""Auto-generated benchmark file for CrossEntropyLoss.

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

    def __init__(self, weight: Optional[torch.Tensor], ignore_index: int, label_smoothing: float, reduction: str, lse_square_scale_for_zloss: float, return_zloss: bool):
        super(Model, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.lse_square_scale_for_zloss = lse_square_scale_for_zloss
        self.return_zloss = return_zloss

    def forward(self, input_predictions: torch.Tensor, target_labels: torch.Tensor) -> List[torch.Tensor]:
        n, c = input_predictions.shape
        input_dtype = input_predictions.dtype
        predictions_fp32 = input_predictions.to(torch.float32)
        if self.weight is None:
            weight_fp32 = torch.ones((c,), dtype=torch.float32, device=predictions_fp32.device)
        else:
            weight_fp32 = self.weight.to(torch.float32)
        predictions_max = torch.max(predictions_fp32, dim=1, keepdim=True)[0]
        lse = predictions_max + torch.log(torch.sum(torch.exp(predictions_fp32 - predictions_max), dim=1, keepdim=True))
        log_softmax_probs = predictions_fp32 - lse
        nll_loss_terms = torch.gather(log_softmax_probs, 1, target_labels.unsqueeze(-1)).squeeze(-1)
        weight_for_targets = torch.gather(weight_fp32, 0, target_labels)
        loss_out_unreduced = -nll_loss_terms * weight_for_targets
        if self.ignore_index >= 0:
            ignore_mask = (target_labels != self.ignore_index).float()
            loss_out_unreduced = loss_out_unreduced * ignore_mask
        else:
            ignore_mask = torch.ones((n,), dtype=torch.float32, device=predictions_fp32.device)
        smooth_loss_unreduced = -torch.sum(log_softmax_probs * weight_fp32.unsqueeze(0), dim=1, keepdim=False)
        if self.ignore_index >= 0:
            smooth_loss_unreduced = smooth_loss_unreduced * ignore_mask
        weight_after_mask_sum = torch.sum(weight_for_targets * ignore_mask, dim=-1, keepdim=False)
        base_loss_reduced = None
        if self.reduction == 'mean':
            base_loss_reduced = torch.sum(loss_out_unreduced, dim=-1, keepdim=False) / (weight_after_mask_sum + 1e-12)
        elif self.reduction == 'sum':
            base_loss_reduced = torch.sum(loss_out_unreduced, dim=-1, keepdim=False)
        else:
            base_loss_reduced = loss_out_unreduced
        smoothed_term_reduced = None
        if self.reduction == 'mean':
            smoothed_term_reduced = torch.sum(smooth_loss_unreduced, dim=-1, keepdim=False) / (weight_after_mask_sum + 1e-12) * self.label_smoothing / c
        elif self.reduction == 'sum':
            smoothed_term_reduced = torch.sum(smooth_loss_unreduced, dim=-1, keepdim=False) * self.label_smoothing / c
        else:
            smoothed_term_reduced = smooth_loss_unreduced * self.label_smoothing / c
        loss_out = (1 - self.label_smoothing) * base_loss_reduced + smoothed_term_reduced
        zloss_out_dtype = input_dtype if input_dtype in [torch.float16, torch.bfloat16] else torch.float32
        zloss_out = torch.zeros((1,), dtype=zloss_out_dtype, device=predictions_fp32.device)
        lse_for_zloss_out = lse.squeeze(-1)
        if self.return_zloss:
            zloss_out = self.lse_square_scale_for_zloss * torch.mean(lse.pow(2))
            zloss_out = zloss_out.reshape(1)
        return [loss_out.to(input_dtype), log_softmax_probs.to(input_dtype), zloss_out.to(input_dtype), lse_for_zloss_out.to(input_dtype)]

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the CrossEntropyLoss operator's forward method.
    """
    batch_size = param.get('batch', 1)
    num_classes = param.get('num_classes', 1000)
    input_dtype_str = param.get('input_dtype', 'float16')
    input_dtype = getattr(torch, input_dtype_str)
    target_dtype_str = param.get('target_dtype', 'int64')
    target_dtype = getattr(torch, target_dtype_str)
    input_predictions = torch.randn([batch_size, num_classes], device=device, dtype=input_dtype)
    target_labels = torch.randint(low=0, high=num_classes, size=(batch_size,), device=device, dtype=target_dtype)
    return (input_predictions, target_labels)

def get_init_inputs_per_case(param, device=None):
    """
    Extract initialization parameters for the CrossEntropyLoss model.
    """
    num_classes = param.get('num_classes', 1000)
    weight_type = param.get('weight_type', 'present')
    weight_dtype_str = param.get('weight_dtype', 'float32')
    weight_dtype = getattr(torch, weight_dtype_str)
    ignore_index = param.get('ignore_index', -100)
    label_smoothing = float(param.get('label_smoothing', 0.0))
    reduction = param.get('reduction', 'mean')
    lse_square_scale_for_zloss = float(param.get('lse_square_scale_for_zloss', 0.0))
    return_zloss_str = str(param.get('return_zloss', False)).lower()
    return_zloss = return_zloss_str == 'true' or return_zloss_str == '1'
    weight = None
    if weight_type == 'present':
        weight = torch.rand(num_classes, device=device, dtype=weight_dtype)
    return [weight, ignore_index, label_smoothing, reduction, lse_square_scale_for_zloss, return_zloss]


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
    json_path = os.path.join(os.path.dirname(__file__), "CrossEntropyLoss.json")
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
