import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs static quantization with given scale and zero point.
        torch_npu.npu_quantize(input, scales, zero_points, dtype, axis=1, div_mode=True) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, input, scales, zero_points=None, axis=1, div_mode=True):
            y = input.float()
            if div_mode:
                y = y / scales.float()
            else:
                y = y * scales.float()
            if zero_points is not None:
                y = y + zero_points.float()
            return torch.round(y).clamp(-128, 127).to(torch.int8)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int = 1, div_mode: bool = True):
        """Quantizes input with per-tensor or per-channel scales/zero_points to int8.
            div_mode=True divides by scale. Returns an int8 tensor."""
        return torch_npu.npu_quantize(input, scales, zero_points, torch.int8, axis, div_mode)


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
                    if name == 'scales':
                        tensors[name] = torch.rand(shape, dtype=torch.float32) + 0.5
                    elif name == 'zero_points':
                        tensors[name] = torch.zeros(shape, dtype=torch.int8) if shape else None
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
