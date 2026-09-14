import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs MoE initial routing with quantization.
        torch_npu.npu_moe_init_routing_quant(x, expert_idx, *, scale=None, offset=None, active_num=1024, expert_capacity=0, expert_num=256, drop_pad_mode=0, expert_tokens_num_mode=0, expert_tokens_before_capacity_flag=False, quant_mode=1) -> (Tensor, Tensor, Tensor, Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, x, expert_idx, active_num=1024, expert_num=256, quant_mode=1):
            # route tokens to experts by expert_idx, then dynamically quantize each routed row
            order = torch.argsort(expert_idx, stable=True)
            routed = x[order[:active_num]].float()
            absmax = routed.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
            scale_out = absmax / 127.0
            q = torch.round(routed / scale_out).clamp(-128, 127).to(torch.int8)
            counts = torch.bincount(expert_idx, minlength=expert_num)
            return q, order[:active_num], counts, scale_out.squeeze(-1), routed
    """
    def __init__(self):
        super(Model, self).__init__()
    def postprocess_output(self, output, inputs):
            return tuple(o for o in output if o is not None)
    


    def forward(self, x: torch.Tensor, expert_idx: torch.Tensor, active_num: int = 8, expert_capacity: int = 0, expert_num: int = 4, drop_pad_mode: int = 0, expert_tokens_num_mode: int = 0, expert_tokens_before_capacity_flag: bool = False, quant_mode: int = 1):
        """Routes tokens by expert_idx (sorted by expert), applies per-row dynamic int8 quantization.
            Returns (expanded_x_quant, expanded_expert_idx, expert_tokens_num, scale, aux)."""
        return torch_npu.npu_moe_init_routing_quant(x, expert_idx, active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num, drop_pad_mode=drop_pad_mode, expert_tokens_num_mode=expert_tokens_num_mode, expert_tokens_before_capacity_flag=expert_tokens_before_capacity_flag, quant_mode=quant_mode)


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
            e = tensors['expert_idx']
            tensors['expert_idx'] = torch.randint(0, attrs['expert_num'], e.shape, dtype=e.dtype)
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
