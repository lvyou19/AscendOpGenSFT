import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes the backward pass of MoE token unpermutation.
        torch_npu.npu_moe_token_unpermute_grad(permuted_tokens, grad_unpermuted_tokens, sorted_indices, probs=None, padded_mode=False, restore_shape=None) -> (Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, permuted_tokens, grad_unpermuted_tokens, sorted_indices, probs=None):
            # forward: out[sorted] = permuted * probs; backward: grad_permuted = grad_unpermuted[sorted] * probs
            g = grad_unpermuted_tokens.float()[sorted_indices]
            if probs is not None:
                grad_permuted = g * probs.float().unsqueeze(-1)
                grad_probs = (g * permuted_tokens.float()).sum(dim=-1)
            else:
                grad_permuted = g
                grad_probs = None
            return grad_permuted.to(permuted_tokens.dtype), grad_probs
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, permuted_tokens: torch.Tensor, grad_unpermuted_tokens: torch.Tensor, sorted_indices: torch.Tensor, probs: torch.Tensor):
        """Backward of MoeTokenUnpermute. Returns (grad_permuted_tokens, grad_probs)."""
        return torch_npu.npu_moe_token_unpermute_grad(permuted_tokens, grad_unpermuted_tokens, sorted_indices, probs)


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
