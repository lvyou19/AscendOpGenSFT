import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs group-wise static quantization (MoE grouped tokens).
        torch_npu.npu_group_quant(x, scale, group_index, *, offset=None, dst_dtype=None) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, x, scale, group_index, offset=None):
            # group_index: [G] cumsum boundaries over rows; scale: [G, N] per-group per-channel scale
            xf = x.float()
            y = torch.empty_like(xf)
            prev = 0
            for g in range(group_index.shape[0]):
                end = int(group_index[g])
                y[prev:end] = torch.round(xf[prev:end] * scale[g].float())
                prev = end
            if offset is not None:
                y = y + offset.float()
            return y.clamp(-128, 127).to(torch.int8)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, scale: torch.Tensor, group_index: torch.Tensor, offset: torch.Tensor):
        """Quantizes rows of x with per-group scales; group_index gives row boundaries (cumsum).
            Returns an int8 tensor shaped like x."""
        return torch_npu.npu_group_quant(x, scale, group_index, offset=offset)


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
                    if name == 'scale':
                        tensors[name] = torch.rand(shape, dtype=torch.float32) + 0.5
                    elif dtype in (torch.int32, torch.int64, torch.int8):
                        tensors[name] = torch.randint(0, 100, shape, dtype=dtype)
                    else:
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    attrs[inp['name']] = inp['value']
            n = tensors['x'].shape[0]
            g = tensors['group_index'].shape[0]
            step = n // g
            tensors['group_index'] = torch.tensor([(i + 1) * step for i in range(g)], dtype=tensors['group_index'].dtype)
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
