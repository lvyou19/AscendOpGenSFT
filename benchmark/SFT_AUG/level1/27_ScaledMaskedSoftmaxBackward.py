import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes the backward pass of scaled masked softmax (Megatron style).
        torch_npu.npu_scaled_masked_softmax_backward(y_grad, y, mask, scale, fixed_triu_mask) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, y_grad, y, mask, scale, fixed_triu_mask):
            # forward: y = softmax(scale * x + mask_fill); mask==True means masked out
            g = y_grad.float()
            sum_term = (y.float() * g).sum(dim=-1, keepdim=True)
            dx = float(scale) * y.float() * (g - sum_term)
            if mask is not None:
                dx = dx.masked_fill(mask, 0.0)
            return dx.to(y_grad.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, y_grad: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, scale: float, fixed_triu_mask: bool):
        """Computes the gradient of scaled masked softmax w.r.t. the pre-softmax input.
            y is the softmax output, mask marks masked-out positions (True = masked).
            Returns a tensor with the same shape as y_grad."""
        return torch_npu.npu_scaled_masked_softmax_backward(y_grad, y, mask, scale, fixed_triu_mask)


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
                    if name == 'y':
                        tensors[name] = torch.nn.functional.softmax(torch.randn(shape, dtype=torch.float32), dim=-1).to(dtype)
                    elif name == 'mask':
                        tensors[name] = (torch.rand(shape) > 0.7)
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
