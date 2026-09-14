import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs layer normalization in eval mode (single output).
        torch_npu.npu_layer_norm_eval(input, normalized_shape, weight=None, bias=None, eps=1e-5) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, input, normalized_shape, weight=None, bias=None, eps=1e-5):
            return torch.nn.functional.layer_norm(input.float(), normalized_shape,
                                                  weight=None if weight is None else weight.float(),
                                                  None if bias is None else bias.float(), eps).to(input.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, normalized_shape: tuple, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
        """Applies layer normalization over the trailing normalized_shape dimensions.
            weight/bias are optional. Returns only the normalized tensor (no mean/rstd)."""
        return torch_npu.npu_layer_norm_eval(input, normalized_shape, weight, bias, eps)


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
                    if name == 'weight':
                        tensors[name] = torch.rand(shape, dtype=dtype) if shape else None
                    elif name == 'bias':
                        tensors[name] = torch.zeros(shape, dtype=dtype) if shape else None
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
