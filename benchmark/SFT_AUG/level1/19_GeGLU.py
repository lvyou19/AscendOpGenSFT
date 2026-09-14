import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs GeGLU (GELU-gated linear unit).
        torch_npu.npu_geglu(input, dim=-1, approximate=1, activate_left=False) -> (Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, input, dim=-1, approximate=1, activate_left=False):
            xf = input.float()
            h = xf.shape[dim] // 2
            first, second = xf[..., :h], xf[..., h:]
            glu, lin = (first, second) if activate_left else (second, first)
            gelu = torch.nn.functional.gelu(glu, approximate="tanh" if approximate == 1 else "none")
            out = lin * gelu
            return out.to(input.dtype), gelu.to(input.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, dim: int = -1, approximate: int = 1, activate_left: bool = False):
        """Splits input along dim into two halves, applies GELU to the gate half and multiplies.
            approximate: 1 for tanh approximation, 0 for exact. activate_left swaps the gate side.
            Returns (out, gelu) where out is lin*gelu(glu)."""
        return torch_npu.npu_geglu(input, dim, approximate, activate_left)


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
