import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that applies a given dropout mask to the input (forward dropout with external mask).
        torch_npu.npu_dropout_do_mask(input, mask, p) -> (Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, input, mask, p):
            out = input * mask.to(input.dtype) / (1.0 - p)
            return out, mask
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, mask: torch.Tensor, p: float):
        """Applies input * mask / (1 - p) with an externally provided uint8 mask.
            Returns (out, mask)."""
        return torch_npu.npu_dropout_do_mask(input, mask, p)


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
                    if name == 'mask':
                        tensors[name] = (torch.rand(shape) > 0.5).to(torch.uint8)
                    elif dtype in (torch.int32, torch.int64, torch.int8):
                        tensors[name] = torch.randint(0, 100, shape, dtype=dtype)
                    else:
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    attrs[inp['name']] = inp['value']
            if tensors['mask'].dtype != torch.uint8:
                tensors['mask'] = tensors['mask'].to(torch.uint8)
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
