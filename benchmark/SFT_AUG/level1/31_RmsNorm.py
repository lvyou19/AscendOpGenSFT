import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs RMS normalization.
        torch_npu.npu_rms_norm(input, gamma, epsilon=1e-6) -> (Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, input, gamma, epsilon=1e-6):
            x = input.float()
            rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
            return (x * rstd * gamma.float()).to(input.dtype), rstd
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6):
        """Applies RMS normalization over the last dimension with scale gamma.
            Returns (out, rstd)."""
        return torch_npu.npu_rms_norm(input, gamma, epsilon)


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
