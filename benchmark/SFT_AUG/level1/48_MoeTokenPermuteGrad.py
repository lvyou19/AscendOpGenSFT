import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes the backward pass of MoE token permutation.
        torch_npu.npu_moe_token_permute_grad(tokens, grad_permuted_tokens, indices, sorted_indices, padded_mode=False) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, tokens, grad_permuted_tokens, indices, sorted_indices):
            # forward permuted tokens by sorted_indices; backward scatters grads back
            grad = torch.empty_like(grad_permuted_tokens)
            grad[sorted_indices] = grad_permuted_tokens
            return grad
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tokens: torch.Tensor, grad_permuted_tokens: torch.Tensor, indices: torch.Tensor, sorted_indices: torch.Tensor):
        """Backward of MoeTokenPermute: scatters permuted gradients back to original token order.
            Returns a tensor shaped like tokens."""
        return torch_npu.npu_moe_token_permute_grad(tokens, grad_permuted_tokens, indices, sorted_indices)


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
            si = tensors['sorted_indices']
            tensors['sorted_indices'] = torch.argsort(tensors['indices'], stable=True).to(si.dtype)
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
