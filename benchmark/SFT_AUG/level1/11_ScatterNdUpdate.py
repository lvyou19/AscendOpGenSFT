import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that updates a tensor at N-D indices (TF ScatterNdUpdate semantics).
        torch_npu.npu_scatter_nd_update(input, indices, updates) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, input, indices, updates):
            # indices: [M, K] coordinates into the first K dims of input
            # updates: [M] + input.shape[K:]
            out = input.clone()
            k = indices.shape[-1]
            for i, idx in enumerate(indices.reshape(-1, k)):
                out[tuple(idx.tolist())] = updates[i]
            return out
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor):
        """Writes updates into input at the N-D coordinates given by indices.
            indices has shape [M, K] (K <= input.ndim), updates has shape [M] + input.shape[K:].
            Returns the updated tensor."""
        return torch_npu.npu_scatter_nd_update(input, indices, updates)


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
            idx = tensors['indices']
            inp = tensors['input']
            k = idx.shape[-1]
            ranges = [inp.shape[d] for d in range(k)]
            cols = [torch.randint(0, ranges[d], (idx.shape[0],), dtype=idx.dtype) for d in range(k)]
            tensors['indices'] = torch.stack(cols, dim=-1)
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
