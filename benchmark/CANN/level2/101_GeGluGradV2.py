"""Auto-generated benchmark file for GeGluGradV2.

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
import torch.nn.functional as F
import ast

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensor_dy: torch.Tensor, tensor_x: torch.Tensor, dim: int, approximate_int: int, activateLeft: bool) -> torch.Tensor:
        """
        实现 GeGLUGradV2 的前向和梯度计算。
        此方法旨在计算 y 对 tensor_x 的梯度，给定上游梯度 tensor_dy。

        Args:
            tensor_dy (torch.Tensor): 上游传来的梯度，形状应与 y 相同。
            tensor_x (torch.Tensor): 输入张量，将被分割并用于计算 GeLU。
            gelu_output (torch.Tensor): 原始代码中的占位符，修改后不再用于输出。
            dim (int): 用于 chunk 操作的维度。
            approximate_int (int): GeLU 函数的近似模式：0 表示 'none'/'erf'，1 表示 'tanh'。
            activateLeft (bool): 未在当前实现中使用。

        Returns:
            torch.Tensor: tensor_x 的梯度。
        """
        approximate_map = {0: 'none', 1: 'tanh'}
        approximate_str = approximate_map.get(approximate_int, 'none')
        with torch.enable_grad():
            x_chunk, gate_chunk = torch.chunk(tensor_x, 2, dim=dim)
            x_for_mul, gate_for_gelu = (gate_chunk, x_chunk)
            y_gelu = F.gelu(gate_for_gelu, approximate=approximate_str)
            y = x_for_mul * y_gelu
            grad_tensor_x = torch.autograd.grad(outputs=y, inputs=tensor_x, grad_outputs=tensor_dy)[0]
        return grad_tensor_x

# ---- prepare_inputs (cleaned from source prepare_inputs.py) ----
import torch
import ast
import torch.nn.functional as F

def get_inputs(param, device=None):
    """
    生成 GeGLUGradV2 算子的输入张量。

    Args:
        param (dict): 参数配置，如输入形状和数据类型
        device (torch.device): 输入张量所在设备

    Returns:
        tuple: 包含输入张量 (dy, x, gelu_param_tensor, dim, approximate, activateLeft)
        # Note: gelu_param_tensor 是为了匹配 Model.forward 的 gelu_output 参数
    """
    shape_str = param.get('input_shape', '[1, 2]')
    shape = ast.literal_eval(shape_str)
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    dim = int(param.get('dim', -1))
    approximate = int(param.get('approximate', 0))
    activateLeft = bool(param.get('activateLeft', True))
    dy = torch.rand(shape, device=device, dtype=dtype)
    x_shape = list(shape)
    if dim == -1:
        x_shape[-1] = x_shape[-1] * 2
    elif dim < len(x_shape):
        x_shape[dim] = x_shape[dim] * 2
    else:
        raise ValueError(f'Invalid dim: {dim} for shape {shape}')
    x = torch.rand(x_shape, device=device, dtype=dtype, requires_grad=True)
    return (dy, x, dim, approximate, activateLeft)

def get_init_inputs_per_case(param, device=None):
    """
    GeGluGradV2 没有模型初始化参数，返回空列表。

    Args:
        param (dict): 参数配置

    Returns:
        list: 空列表
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
    json_path = os.path.join(os.path.dirname(__file__), "GeGluGradV2.json")
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
