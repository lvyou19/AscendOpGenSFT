import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs per-token asymmetric dynamic int8 quantization.
        torch_npu.npu_dynamic_quant_asymmetric(input, *, smooth_scales=None, group_index=None, dst_type=None, quant_mode="pertoken") -> (Tensor, Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, input, quant_mode="pertoken"):
            xf = input.float()
            x_min = xf.amin(dim=-1, keepdim=True)
            x_max = xf.amax(dim=-1, keepdim=True)
            scale = ((x_max - x_min) / 255.0).clamp(min=1e-10)
            offset = torch.round(-128 - x_min / scale)
            q = torch.round(xf / scale + offset).clamp(-128, 127).to(torch.int8)
            return q, scale.squeeze(-1), offset.squeeze(-1)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, quant_mode: str = "pertoken"):
        """Per-token (last-dim) asymmetric dynamic quantization to int8.
            Returns (quantized_int8, scale, offset)."""
        return torch_npu.npu_dynamic_quant_asymmetric(input, quant_mode=quant_mode)


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
