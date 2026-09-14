import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes the backward pass of RMS normalization.
        torch_npu.npu_rms_norm_backward(dy, input, gamma, rstd) -> (Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, dy, input, gamma, rstd):
            xf = input.float()
            g = dy.float() * gamma.float()
            ms = xf.pow(2).mean(dim=-1, keepdim=True)
            dx = (g - xf * (g * xf).mean(dim=-1, keepdim=True) / (ms + 1e-12)) * rstd
            dgamma = (dy.float() * xf * rstd).sum(dim=tuple(range(xf.ndim - 1)))
            return dx.to(input.dtype), dgamma.to(input.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dy: torch.Tensor, input: torch.Tensor, gamma: torch.Tensor, rstd: torch.Tensor):
        """Computes gradients of RMSNorm w.r.t. input and gamma.
            Returns (grad_input, grad_gamma)."""
        return torch_npu.npu_rms_norm_backward(dy, input, gamma, rstd)


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
                    if name == 'gamma':
                        tensors[name] = torch.rand(shape, dtype=dtype)
                    elif name == 'rstd':
                        tensors[name] = torch.rand(shape, dtype=dtype) + 0.5
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
