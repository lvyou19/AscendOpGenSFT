import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs Group Normalization with SiLU activation.
        torch_npu.npu_group_norm_silu(input, weight, bias, group, eps=1e-5) -> (Tensor, Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, input, weight, bias, group, eps=1e-5):
            N, C = input.shape[0], input.shape[1]
            x = input.view(N, group, C // group, *input.shape[2:])
            dims = list(range(2, x.ndim))
            mean = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, unbiased=False, keepdim=True)
            rstd = 1.0 / torch.sqrt(var + eps)
            x_norm = ((x - mean) * rstd).view(input.shape)
            w = weight.view(1, -1, *([1] * (input.ndim - 2)))
            b = bias.view(1, -1, *([1] * (input.ndim - 2)))
            out = torch.nn.functional.silu(x_norm * w + b)
            return out, mean.view(N, group), rstd.view(N, group)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, group: int, eps: float = 1e-5):
        """Applies group normalization followed by SiLU activation.
            Returns (out, mean, rstd), out has the same shape as input, mean/rstd have shape (N, group)."""
        return torch_npu.npu_group_norm_silu(input, weight, bias, group, eps)


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
                        tensors[name] = torch.ones(shape, dtype=dtype)
                    elif name == 'bias':
                        tensors[name] = torch.zeros(shape, dtype=dtype)
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
