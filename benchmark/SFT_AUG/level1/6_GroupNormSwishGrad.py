import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes the backward pass of GroupNorm with Swish activation.
        torch_npu.npu_group_norm_swish_grad(grad, input, num_groups, weight, bias, mean, rstd, grad_input_mask, swish_scale=1.0) -> (Tensor, Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, grad, input, num_groups, weight, bias, mean, rstd, grad_input_mask, swish_scale=1.0):
            N, C = input.shape[0], input.shape[1]
            dims = list(range(2, input.ndim))
            xf = input.float()
            xr = xf.view(N, num_groups, C // num_groups, *input.shape[2:])
            z = ((xr - mean.view(N, num_groups, *([1] * (input.ndim - 1)))) * rstd.view(N, num_groups, *([1] * (input.ndim - 1)))).view(xf.shape)
            w = weight.float().view(1, -1, *([1] * (input.ndim - 2)))
            a = z * w + bias.float().view(1, -1, *([1] * (input.ndim - 2)))
            s = torch.sigmoid(swish_scale * a)
            dsw = s + swish_scale * a * s * (1 - s)
            g = grad.float() * dsw
            grad_bias = g.sum(dim=(0,) + tuple(dims))
            grad_weight = (g * z).sum(dim=(0,) + tuple(dims))
            gz = g * w
            gz_r = gz.view(N, num_groups, C // num_groups, *input.shape[2:])
            z_r = z.view(N, num_groups, C // num_groups, *input.shape[2:])
            red = list(range(2, gz_r.ndim))
            m_gz = gz_r.mean(dim=red, keepdim=True)
            m_gzz = (gz_r * z_r).mean(dim=red, keepdim=True)
            dx = (gz_r - m_gz - z_r * m_gzz) * rstd.view(N, num_groups, *([1] * (input.ndim - 1)))
            return dx.view(xf.shape).to(input.dtype), grad_weight.to(input.dtype), grad_bias.to(input.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad: torch.Tensor, input: torch.Tensor, num_groups: int, weight: torch.Tensor, bias: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, grad_input_mask: tuple, swish_scale: float = 1.0):
        """Computes gradients of GroupNorm+Swish w.r.t. input, weight and bias.
            grad_input_mask selects which of (dx, dw, db) are computed.
            Returns (grad_input, grad_weight, grad_bias)."""
        return torch_npu.npu_group_norm_swish_grad(grad, input, num_groups, weight, bias, mean, rstd, grad_input_mask, swish_scale)


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
