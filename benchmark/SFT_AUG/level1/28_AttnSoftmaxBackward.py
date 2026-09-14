import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes attention softmax backward in place.
        torch_npu.npu_attn_softmax_backward_(input, grad_output, values) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, input, grad_output, values):
            # input: softmax output y (modified in place to become the gradient)
            # grad_output: upstream gradient, values: softmax values y
            y = values.float()
            g = grad_output.float()
            sum_term = (y * g).sum(dim=-1, keepdim=True)
            dx = y * (g - sum_term)
            input.copy_(dx.to(input.dtype))
            return input
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, grad_output: torch.Tensor, values: torch.Tensor):
        """Computes softmax backward y * (g - sum(y * g)) along the last dim and writes the result
            into input in place. values holds the softmax output. Returns input."""
        torch_npu.npu_attn_softmax_backward_(input, grad_output, values)
        return input


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
                    if name == 'values':
                        tensors[name] = torch.nn.functional.softmax(torch.randn(shape, dtype=torch.float32), dim=-1).to(dtype)
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
