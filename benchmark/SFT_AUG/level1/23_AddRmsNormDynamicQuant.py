import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs fused residual add + RMS normalization + per-token dynamic int8 quantization.
        torch_npu.npu_add_rms_norm_dynamic_quant(x1, x2, gamma, *, smooth_scale1=None, smooth_scale2=None, beta=None, epsilon=1e-6, output_mask=[], y_dtype=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, x1, x2, gamma, beta=None, epsilon=1e-6):
            z = x1.float() + x2.float()
            rstd = torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + epsilon)
            n = z * rstd * gamma.float()
            if beta is not None:
                n = n + beta.float()
            absmax = n.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
            scale = absmax / 127.0
            q = torch.round(n / scale).clamp(-128, 127).to(torch.int8)
            return q, scale.squeeze(-1), n.to(x1.dtype), z.to(x1.dtype), rstd
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-6, output_mask: tuple = (True, True)):
        """Computes z = x1 + x2, RMS-normalizes z, then applies per-token dynamic int8 quantization.
            Returns (y_quant, scale, norm_out, add_out, rstd)."""
        return torch_npu.npu_add_rms_norm_dynamic_quant(x1, x2, gamma, beta=beta, epsilon=epsilon, output_mask=list(output_mask))


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
                    if name == 'gamma':
                        tensors[name] = torch.rand(shape, dtype=dtype)
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
