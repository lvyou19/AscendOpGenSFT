import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs strided slicing with bit masks (TF StridedSlice semantics).
        torch_npu.npu_indexing(input, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, input, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
            slices = []
            out_dims = []
            for d in range(len(begin)):
                b = 0 if (begin_mask >> d) & 1 else begin[d]
                e = input.shape[d] if (end_mask >> d) & 1 else end[d]
                s = strides[d]
                slices.append(slice(b, e, s))
                if not (shrink_axis_mask >> d) & 1:
                    out_dims.append(d)
            out = input[tuple(slices)]
            return out
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, begin: tuple, end: tuple, strides: tuple, begin_mask: int = 0, end_mask: int = 0, ellipsis_mask: int = 0, new_axis_mask: int = 0, shrink_axis_mask: int = 0):
        """Extracts a strided slice of input. begin/end/strides are per-dimension lists;
            bit i of begin_mask/end_mask/shrink_axis_mask controls open bounds and dimension removal.
            Returns the sliced tensor."""
        return torch_npu.npu_indexing(input, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)


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
