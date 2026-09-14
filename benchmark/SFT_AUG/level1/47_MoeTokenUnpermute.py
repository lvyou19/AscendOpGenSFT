import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that restores MoE-permuted tokens to their original order with optional probs scaling.
        torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=None, padded_mode=False, restore_shape=None) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, permuted_tokens, sorted_indices, probs=None):
            out = torch.empty_like(permuted_tokens)
            p = permuted_tokens.float()
            if probs is not None:
                p = p * probs.float().unsqueeze(-1)
            out[sorted_indices] = p.to(permuted_tokens.dtype)
            return out
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, permuted_tokens: torch.Tensor, sorted_indices: torch.Tensor, probs: torch.Tensor):
        """Scatters permuted tokens back to their original positions (inverse of MoeTokenPermute),
            optionally scaling rows by probs. Returns restored tokens."""
        return torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs)


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
                    if name == 'probs':
                        tensors[name] = torch.rand(shape, dtype=torch.float32) if shape else None
                    elif dtype in (torch.int32, torch.int64, torch.int8):
                        tensors[name] = torch.randint(0, 100, shape, dtype=dtype)
                    else:
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    attrs[inp['name']] = inp['value']
            si = tensors['sorted_indices']
            tensors['sorted_indices'] = torch.randperm(si.shape[0], dtype=si.dtype)
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
