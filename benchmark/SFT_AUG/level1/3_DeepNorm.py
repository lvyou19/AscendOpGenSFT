import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs DeepNorm (residual + layer normalization, DeepNet style).
        torch_npu.npu_deep_norm(x, gx, beta, gamma, alpha=0.3, epsilon=1e-6) -> (Tensor, Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, x, gx, beta, gamma, alpha=0.3, epsilon=1e-6):
            z = alpha * x.float() + gx.float()
            mean = z.mean(dim=-1, keepdim=True)
            var = z.var(dim=-1, unbiased=False, keepdim=True)
            rstd = 1.0 / torch.sqrt(var + epsilon)
            out = ((z - mean) * rstd * gamma.float() + beta.float()).to(x.dtype)
            return out, mean, rstd
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, gx: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, alpha: float = 0.3, epsilon: float = 1e-6):
        """Applies DeepNorm: layer normalization of (alpha * x + gx).
            Returns (out, mean, rstd), mean/rstd have shape x.shape[:-1] + [1]."""
        return torch_npu.npu_deep_norm(x, gx, beta, gamma, alpha, epsilon)


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
