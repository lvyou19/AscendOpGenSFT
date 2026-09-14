import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes the inverse FFT of a real signal into Hermitian form.
        torch.fft.ihfft(x_freq, n) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, x_freq, n):
            return torch.fft.ihfft(x_freq, n=n)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x_freq: torch.Tensor, n: int):
        """Computes the inverse Hermitian FFT: real input of length n is transformed into
            a Hermitian complex spectrum of length n // 2 + 1. Returns a complex tensor
            (compared as real and imag parts)."""
        out = torch.fft.ihfft(x_freq, n=n)
        return out.real.contiguous(), out.imag.contiguous()


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
            x = tensors.pop('x')
            tensors['x_freq'] = torch.fft.hfft(x.float(), n=attrs['n'])
            group = [tensors['x_freq'], attrs['n']]
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
