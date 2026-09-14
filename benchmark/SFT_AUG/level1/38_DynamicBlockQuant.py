import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs per-block dynamic quantization.
        torch_npu.npu_dynamic_block_quant(x, *, min_scale=0.0, round_mode="rint", dst_type=1, row_block_size=1, col_block_size=128) -> (Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, x, row_block_size=1, col_block_size=128):
            # split last dims into [row_block_size, col_block_size] blocks, per-block absmax int8 quant
            xf = x.float()
            B = xf.shape[0]
            N = xf.shape[-1]
            y = torch.empty_like(xf)
            scale = torch.empty(B, N // col_block_size, dtype=torch.float32)
            for b in range(B):
                for j in range(N // col_block_size):
                    blk = xf[b, j * col_block_size:(j + 1) * col_block_size]
                    amax = blk.abs().max().clamp(min=1e-10)
                    scale[b, j] = amax / 127.0
                    y[b, j * col_block_size:(j + 1) * col_block_size] = torch.round(blk / scale[b, j])
            return y.to(torch.int8), scale
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, row_block_size: int = 1, col_block_size: int = 128):
        """Per-block dynamic int8 quantization along the last dimension.
            Returns (y_int8, scale_fp32)."""
        return torch_npu.npu_dynamic_block_quant(x, row_block_size=row_block_size, col_block_size=col_block_size)


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
