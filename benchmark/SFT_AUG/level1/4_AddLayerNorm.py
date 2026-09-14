import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs fused residual add + layer normalization.
        torch_npu.npu_add_layer_norm(x1, x2, gamma, beta, epsilon=1e-5, additional_output=False) -> (Tensor, Tensor, Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, x1, x2, gamma, beta, epsilon=1e-5, additional_output=False):
            z = x1.float() + x2.float()
            mean = z.mean(dim=-1, keepdim=True)
            var = z.var(dim=-1, unbiased=False, keepdim=True)
            rstd = 1.0 / torch.sqrt(var + epsilon)
            out = ((z - mean) * rstd * gamma.float() + beta.float()).to(x1.dtype)
            return out, mean, rstd, z.to(x1.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5, additional_output: bool = False):
        """Computes z = x1 + x2 then layer normalization of z over the last dimension.
            Returns (out, mean, rstd, add_out)."""
        return torch_npu.npu_add_layer_norm(x1, x2, gamma, beta, epsilon, additional_output)


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
                    elif name == 'beta':
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
