import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes the backward pass of gather (dense scatter-add).
        torch_npu.npu_gather_backward(grad, self_size, dim, index, sparse_grad) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, grad, self_size, dim, index, sparse_grad=False):
            out = torch.zeros(self_size, dtype=grad.dtype, device=grad.device)
            return out.scatter_add_(dim, index, grad)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad: torch.Tensor, self_size: tuple, dim: int, index: torch.Tensor, sparse_grad: bool = False):
        """Accumulates grad into a zeros tensor of shape self_size at positions given by index along dim.
            Returns a tensor of shape self_size."""
        return torch_npu.npu_gather_backward(grad, self_size, dim, index, sparse_grad)


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
            tensors['index'] = torch.randint(0, attrs['self_size'][attrs['dim']], idx.shape, dtype=idx.dtype)
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
