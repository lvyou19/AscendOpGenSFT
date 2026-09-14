import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs masked softmax with relative position bias (Bert/Deberta style).
        torch_npu.npu_masked_softmax_with_rel_pos_bias(x, atten_mask, relative_pos_bias, scale_value=1.0, inner_precision_mode=0) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, x, atten_mask, relative_pos_bias, scale_value=1.0, inner_precision_mode=0):
            y = x.float() * scale_value + relative_pos_bias.float()
            if atten_mask is not None:
                y = y.masked_fill(atten_mask, float('-inf'))
            return torch.softmax(y, dim=-1).to(x.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, atten_mask: torch.Tensor, relative_pos_bias: torch.Tensor, scale_value: float = 1.0, inner_precision_mode: int = 0):
        """Computes softmax(scale * x + relative_pos_bias) with optional attention mask.
            x has shape [B, N, T, S]. Returns a tensor shaped like x."""
        return torch_npu.npu_masked_softmax_with_rel_pos_bias(x, atten_mask, relative_pos_bias, scale_value, inner_precision_mode)


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
                    if name == 'atten_mask':
                        tensors[name] = (torch.rand(shape) > 0.7) if shape else None
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
