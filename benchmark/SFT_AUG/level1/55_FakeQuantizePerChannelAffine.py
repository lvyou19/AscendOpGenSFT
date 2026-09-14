import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs fake quantization with per-channel affine parameters.
        torch.fake_quantize_per_channel_affine(x, scale, zero_point, axis, quant_min, quant_max) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, x, scale, zero_point, axis, quant_min, quant_max):
            shape = [1] * x.ndim
            shape[axis] = -1
            s = scale.float().view(shape)
            zp = zero_point.float().view(shape)
            q = torch.round(x.float() / s + zp).clamp(quant_min, quant_max)
            return (q - zp) * s
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, axis: int, quant_min: int, quant_max: int):
        """Per-channel fake quantization along axis; scale/zero_point have length x.shape[axis].
            Returns a float tensor shaped like x."""
        return torch.fake_quantize_per_channel_affine(x, scale, zero_point, axis, quant_min, quant_max)


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
                    if name == 'scale':
                        tensors[name] = torch.rand(shape, dtype=torch.float32) * 0.1 + 0.01
                    elif name == 'zero_point':
                        tensors[name] = torch.zeros(shape, dtype=torch.int32)
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
