import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes the backward pass of SwiGLU.
        torch_npu.npu_swiglu_backward(grad_output, input, dim=-1) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, grad_output, input, dim=-1):
            xf = input.float()
            h = xf.shape[dim] // 2
            x1, x2 = xf[..., :h], xf[..., h:]
            # forward: out = x1 * sigmoid(x1) * x2  (swish(x1) * x2)
            s = torch.sigmoid(x1)
            g = grad_output.float()
            d_x2 = g * x1 * s
            d_x1 = g * x2 * (s + x1 * s * (1 - s))
            return torch.cat([d_x1, d_x2], dim=dim).to(input.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, input: torch.Tensor, dim: int = -1):
        """Computes the gradient of SwiGLU (out = swish(x1) * x2) w.r.t. input.
            Returns a tensor with the same shape as input."""
        return torch_npu.npu_swiglu_backward(grad_output, input, dim)


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
