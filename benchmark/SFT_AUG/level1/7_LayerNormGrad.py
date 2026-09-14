import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes the backward pass of layer normalization.
        torch_npu.npu_layernorm_grad(grad_out, input, normalized_shape, mean, rstd, weight=None, bias=None) -> (Tensor, Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, grad_out, input, normalized_shape, mean, rstd, weight, bias):
            nd = len(normalized_shape)
            red = tuple(range(input.ndim - nd, input.ndim))
            g = grad_out.float()
            if weight is not None:
                g = g * weight.float()
            z = (input.float() - mean) * rstd
            m_g = g.mean(dim=red, keepdim=True)
            m_gz = (g * z).mean(dim=red, keepdim=True)
            dx = (g - m_g - z * m_gz) * rstd
            dw = (grad_out.float() * z).sum(dim=tuple(range(input.ndim - nd)))
            db = grad_out.float().sum(dim=tuple(range(input.ndim - nd)))
            return dx.to(input.dtype), dw.to(input.dtype), db.to(input.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_out: torch.Tensor, input: torch.Tensor, normalized_shape: tuple, mean: torch.Tensor, rstd: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        """Computes gradients of layer normalization w.r.t. input, weight and bias.
            Returns (grad_input, grad_weight, grad_bias)."""
        return torch_npu.npu_layernorm_grad(grad_out, input, normalized_shape, mean, rstd, weight, bias)


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
                    if name == 'weight':
                        tensors[name] = torch.rand(shape, dtype=dtype)
                    elif name == 'bias':
                        tensors[name] = torch.zeros(shape, dtype=dtype)
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
