import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that reduces values into a tensor at scatter positions with max.
        torch.scatter_reduce(x, dim, index, src, "amax") -> (values, indices)

        PyTorch native implementation of forward function
        def forward(self, x, dim, index, src):
            out = x.scatter_reduce(dim, index, src, reduce="amax", include_self=True)
            return out.values, out.indices
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor):
        """Takes element-wise max of x and src scattered at index positions along dim.
            Returns (values, indices) shaped like x."""
        out = x.scatter_reduce(dim, index, src, reduce="amax", include_self=True)
        return out.values, out.indices


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
                    if dtype in (torch.int32, torch.int64, torch.int8):
                        tensors[name] = torch.randint(0, 100, shape, dtype=dtype)
                    else:
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    attrs[inp['name']] = inp['value']
            idx = tensors['index']
            tensors['index'] = torch.randint(0, tensors['x'].shape[attrs['dim']], idx.shape, dtype=idx.dtype)
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
