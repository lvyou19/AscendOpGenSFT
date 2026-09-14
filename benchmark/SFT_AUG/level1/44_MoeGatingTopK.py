import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs MoE gating with top-k expert selection.
        torch_npu.npu_moe_gating_top_k(x, k, *, bias=None, k_group=1, group_count=1, group_select_mode=0, renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1.0, eps=1e-20) -> (Tensor, Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, x, k, bias=None):
            logits = x.float()
            if bias is not None:
                logits = logits + bias.float()
            values, indices = torch.topk(logits, k, dim=-1)
            return values, indices, torch.softmax(logits, dim=-1)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, k: int, bias: torch.Tensor):
        """Selects top-k experts per token from gating logits x [num_tokens, num_experts].
            Returns (topk_values, topk_indices, softmax_out)."""
        return torch_npu.npu_moe_gating_top_k(x, k, bias=bias)


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
