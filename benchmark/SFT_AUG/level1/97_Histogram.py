import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes a histogram of values over fixed bins.
        torch.histogram(x, bins, range) -> (hist, bin_edges)

        PyTorch native implementation of forward function
        def forward(self, x, bins, range):
            return torch.histogram(x.float(), bins=bins, range=tuple(range))
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, bins: int, range: tuple):
        """Computes a histogram with fixed bin count and value range.
            Returns (hist, bin_edges) with lengths bins and bins + 1."""
        return torch.histogram(x.float(), bins=bins, range=tuple(range))


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
                    if name == 'x':
                        tensors[name] = torch.randn(shape, dtype=dtype)
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
