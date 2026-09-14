import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes the backward pass of fused add + layer normalization.
        torch_npu.npu_add_layer_norm_backward(dy_opt, x1, x2, rstd, mean, gamma, dsum_opt=None) -> (Tensor, Tensor, Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, dy_opt, x1, x2, rstd, mean, gamma, dsum_opt=None):
            z = x1.float() + x2.float()
            g = dy_opt.float() * gamma.float()
            if dsum_opt is not None:
                g = g + dsum_opt.float()
            zn = (z - mean) * rstd
            m_g = g.mean(dim=-1, keepdim=True)
            m_gz = (g * zn).mean(dim=-1, keepdim=True)
            dz = (g - m_g - zn * m_gz) * rstd
            dgamma = (dy_opt.float() * zn).sum(dim=tuple(range(z.ndim - 1)))
            dbeta = dy_opt.float().sum(dim=tuple(range(z.ndim - 1)))
            return dz.to(x1.dtype), dz.to(x1.dtype), dgamma.to(x1.dtype), dbeta.to(x1.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dy_opt: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, rstd: torch.Tensor, mean: torch.Tensor, gamma: torch.Tensor, dsum_opt: torch.Tensor):
        """Computes gradients of fused add+LayerNorm.
            Returns (grad_x1, grad_x2, grad_gamma, grad_beta)."""
        return torch_npu.npu_add_layer_norm_backward(dy_opt, x1, x2, rstd, mean, gamma, dsum_opt)


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
