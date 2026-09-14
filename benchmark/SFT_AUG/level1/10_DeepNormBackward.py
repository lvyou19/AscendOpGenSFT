import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes the backward pass of DeepNorm.
        torch_npu.npu_deep_norm_backward(dy, x, gx, gamma, mean, rstd, alpha=0.3) -> (Tensor, Tensor, Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, dy, x, gx, gamma, mean, rstd, alpha=0.3):
            z = alpha * x.float() + gx.float()
            g = dy.float() * gamma.float()
            zn = (z - mean) * rstd
            m_g = g.mean(dim=-1, keepdim=True)
            m_gz = (g * zn).mean(dim=-1, keepdim=True)
            dz = (g - m_g - zn * m_gz) * rstd
            dgamma = (dy.float() * zn).sum(dim=tuple(range(z.ndim - 1)))
            dbeta = dy.float().sum(dim=tuple(range(z.ndim - 1)))
            return (alpha * dz).to(x.dtype), dz.to(x.dtype), dgamma.to(x.dtype), dbeta.to(x.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dy: torch.Tensor, x: torch.Tensor, gx: torch.Tensor, gamma: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, alpha: float = 0.3):
        """Computes gradients of DeepNorm w.r.t. x, gx, gamma and beta.
            Returns (grad_x, grad_gx, grad_gamma, grad_beta)."""
        return torch_npu.npu_deep_norm_backward(dy, x, gx, gamma, mean, rstd, alpha)


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
