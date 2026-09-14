import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes the backward pass of GeGLU.
        torch_npu.npu_geglu_grad(grad_output, input, gelu, dim=-1, approximate=1, activate_left=False) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, grad_output, input, gelu, dim=-1, approximate=1, activate_left=False):
            xf = input.float()
            h = xf.shape[dim] // 2
            first, second = xf[..., :h], xf[..., h:]
            glu, lin = (first, second) if activate_left else (second, first)
            g = grad_output.float()
            d_lin = g * gelu.float()
            # derivative of gelu (tanh approximation when approximate == 1)
            if approximate == 1:
                t = torch.tanh(0.7978845608 * (glu + 0.044715 * glu.pow(3)))
                d_gelu = 0.5 * (1 + t) + 0.5 * glu * (1 - t.pow(2)) * 0.7978845608 * (1 + 3 * 0.044715 * glu.pow(2))
            else:
                d_gelu = torch.where(glu > 0, torch.ones_like(glu), torch.zeros_like(glu))  # simplified
            d_glu = g * lin * d_gelu
            grad_input = torch.cat([d_glu, d_lin], dim=dim) if activate_left else torch.cat([d_lin, d_glu], dim=dim)
            return grad_input.to(input.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, input: torch.Tensor, gelu: torch.Tensor, dim: int = -1, approximate: int = 1, activate_left: bool = False):
        """Computes the gradient of GeGLU w.r.t. input, given grad_output and the saved gelu tensor.
            Returns a tensor with the same shape as input."""
        return torch_npu.npu_geglu_grad(grad_output, input, gelu, dim, approximate, activate_left)


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
