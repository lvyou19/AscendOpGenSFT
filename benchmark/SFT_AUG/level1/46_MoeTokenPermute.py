import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that permutes tokens by expert assignment (MoE routing permutation).
        torch_npu.npu_moe_token_permute(tokens, indices, num_out_tokens=None, padded_mode=False) -> (Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, tokens, indices):
            # indices: [T] expert id per token; sort tokens by expert id
            sorted_indices = torch.argsort(indices, stable=True)
            permuted = tokens[sorted_indices]
            return permuted, sorted_indices
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tokens: torch.Tensor, indices: torch.Tensor):
        """Sorts tokens [T, H] by their expert assignment indices [T].
            Returns (permuted_tokens, sorted_indices)."""
        return torch_npu.npu_moe_token_permute(tokens, indices)


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
            idx = tensors['indices']
            tensors['indices'] = torch.randint(0, 4, idx.shape, dtype=idx.dtype)
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
