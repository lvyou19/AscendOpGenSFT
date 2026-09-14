import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs SwiGLU with clamping (gpt-oss variant).
        torch_npu.npu_clipped_swiglu(x, *, group_index=None, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, x, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True):
            xf = x.float()
            if interleaved:
                glu = xf[..., 0::2]   # even positions
                lin = xf[..., 1::2]   # odd positions
            else:
                h = xf.shape[dim] // 2
                glu, lin = xf[..., :h], xf[..., h:]
            glu_c = glu.clamp(max=limit)
            lin_c = lin.clamp(-limit, limit)
            out = glu_c * torch.sigmoid(alpha * glu_c) * (lin_c + bias)
            return out.to(x.dtype)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int = -1, alpha: float = 1.702, limit: float = 7.0, bias: float = 1.0, interleaved: bool = True):
        """Applies clipped SwiGLU: splits x along dim (interleaved even/odd or halves),
            clamps both branches, returns glu*sigmoid(alpha*glu)*(lin+bias).
            Last/split dimension must be even."""
        return torch_npu.npu_clipped_swiglu(x, dim=dim, alpha=alpha, limit=limit, bias=bias, interleaved=interleaved)


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
