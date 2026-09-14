import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes softmax cross entropy with soft labels (TF semantics).
        torch_npu.npu_softmax_cross_entropy_with_logits(input, labels) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, input, labels):
            log_prob = torch.nn.functional.log_softmax(input.float(), dim=-1)
            return -(labels.float() * log_prob).sum(dim=-1)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, labels: torch.Tensor):
        """Computes per-row cross entropy between input logits and soft label distributions
            with the same shape. Returns a 1D loss tensor of shape [N]."""
        return torch_npu.npu_softmax_cross_entropy_with_logits(input, labels)


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
                    if name == 'labels':
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
