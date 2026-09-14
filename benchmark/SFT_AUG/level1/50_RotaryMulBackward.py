import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes the backward pass of rotary multiplication (RoPE).
        torch_npu.npu_rotary_mul_backward(grad, input, r1, r2, rotary_mode="half") -> (Tensor, Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, grad, input, r1, r2, rotary_mode="half"):
            # forward: out = input * r1 + rotate_half(input) * r2
            def rot(x):
                x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
                return torch.cat([-x2, x1], dim=-1)
            g = grad.float()
            d_input = g * r1.float() - rot(g) * r2.float()
            d_r1 = g * input.float()
            d_r2 = g * rot(input.float())
            return d_input.to(grad.dtype), d_r1.to(grad.dtype), d_r2.to(grad.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad: torch.Tensor, input: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, rotary_mode: str = "half"):
        """Backward of out = input * r1 + rotate_half(input) * r2.
            Returns (grad_input, grad_r1, grad_r2)."""
        return torch_npu.npu_rotary_mul_backward(grad, input, r1, r2, rotary_mode)


def get_input_groups():
    """Generate input groups from JSON test cases."""
    json_path = os.path.join(os.path.dirname(__file__), os.path.splitext(os.path.basename(__file__))[0] + '.json')
    input_groups = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            inputs = case['inputs']
            tensors = {}
            attrs = {}
            for inp in inputs:
                if inp['type'] == 'tensor':
                    name = inp['name']
                    dtype_str = inp.get('dtype', 'float32')
                    shape = inp.get('shape')
                    if shape is None:
                        tensors[name] = None
                        continue
                    dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16,
                             'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}[dtype_str]
                    if name == 'r1':
                        tensors[name] = torch.rand(shape, dtype=dtype) * 2 - 1
                    elif name == 'r2':
                        tensors[name] = torch.rand(shape, dtype=dtype) * 2 - 1
                    elif dtype in (torch.int32, torch.int64, torch.int8):
                        tensors[name] = torch.randint(0, 100, shape, dtype=dtype)
                    else:
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    attrs[inp['name']] = inp['value']

            group = []
            for inp in inputs:
                if inp['type'] == 'tensor':
                    group.append(tensors[inp['name']])
                else:
                    group.append(attrs[inp['name']])
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
