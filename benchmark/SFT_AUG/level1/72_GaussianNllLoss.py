import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes Gaussian negative log likelihood loss.
        torch.nn.functional.gaussian_nll_loss(input, target, var, reduction) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, input, target, var, reduction="mean"):
            return torch.nn.functional.gaussian_nll_loss(input, target, var, reduction=reduction)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, var: torch.Tensor, reduction: str = "mean"):
        """Gaussian NLL: 0.5 * (log(var) + (input - target)^2 / var). var must be positive.
            reduction is "mean"/"sum"/"none"."""
        return torch.nn.functional.gaussian_nll_loss(input, target, var, reduction=reduction)


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
                    if name == 'var':
                        tensors[name] = torch.rand(shape, dtype=dtype) + 0.1
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
