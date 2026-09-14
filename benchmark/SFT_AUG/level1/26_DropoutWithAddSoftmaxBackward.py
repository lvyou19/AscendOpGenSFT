import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes the backward pass of dropout with add-softmax fusion.
        torch_npu.npu_dropout_with_add_softmax_backward(grad, mask, softmax_out, alpha, prob, dim) -> (Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, grad, mask, softmax_out, alpha, prob, dim):
            y = softmax_out.float()
            g = grad.float() * mask.float() / (1.0 - prob)
            sum_term = (y * g).sum(dim=dim, keepdim=True)
            dx = y * (g - sum_term)
            d_add = (g * y).sum(dim=dim, keepdim=False) * float(alpha)
            return dx.to(grad.dtype), d_add.to(grad.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad: torch.Tensor, mask: torch.Tensor, softmax_out: torch.Tensor, alpha: float, prob: float, dim: int):
        """Backward of fused (add + softmax + dropout): scales grad by the dropout mask,
            applies softmax backward along dim, and also returns the gradient of the added term.
            Returns (grad_input, grad_add)."""
        return torch_npu.npu_dropout_with_add_softmax_backward(grad, mask, softmax_out, alpha, prob, dim)


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
                    if name == 'mask':
                        tensors[name] = (torch.rand(shape) > 0.5).to(torch.uint8)
                    elif name == 'softmax_out':
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
