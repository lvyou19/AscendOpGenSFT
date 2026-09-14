import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that gathers K/V tokens from paged KV caches (vLLM style).
        torch_npu.npu_gather_pa_kv_cache(key_cache, value_cache, block_tables, seq_lens, key, value, *, seq_offset=None, is_seq_lens_cumsum=False) -> ()

        PyTorch native implementation of forward function
        def forward(self, key_cache, value_cache, block_tables, seq_lens, key, value):
            block_size = key_cache.shape[1]
            pos = 0
            for b in range(block_tables.shape[0]):
                for s in range(int(seq_lens[b])):
                    block = int(block_tables[b, s // block_size])
                    offset = s % block_size
                    key[pos] = key_cache[block, offset]
                    value[pos] = value_cache[block, offset]
                    pos += 1
            return key, value
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, key_cache: torch.Tensor, value_cache: torch.Tensor, block_tables: torch.Tensor, seq_lens: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """Reads tokens from paged caches [num_blocks, block_size, num_heads, head_dim] via block_tables
            and seq_lens into contiguous key/value [total_tokens, num_heads, head_dim].
            key/value are updated in place. Returns (key, value)."""
        torch_npu.npu_gather_pa_kv_cache(key_cache, value_cache, block_tables, seq_lens, key, value)
        return key, value


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
                    if name == 'key':
                        tensors[name] = torch.zeros(shape, dtype=dtype)
                    elif name == 'value':
                        tensors[name] = torch.zeros(shape, dtype=dtype)
                    elif dtype in (torch.int32, torch.int64, torch.int8):
                        tensors[name] = torch.randint(0, 100, shape, dtype=dtype)
                    else:
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    attrs[inp['name']] = inp['value']
            bt = tensors['block_tables']
            num_blocks = tensors['key_cache'].shape[0]
            tensors['block_tables'] = torch.stack([torch.randperm(num_blocks, dtype=torch.int32)[:bt.shape[1]] for _ in range(bt.shape[0])])
            block_size = tensors['key_cache'].shape[1]
            max_len = bt.shape[1] * block_size
            total_out = tensors['key'].shape[0]
            sl = torch.randint(1, max_len + 1, tensors['seq_lens'].shape, dtype=torch.int32)
            scale = total_out / max(int(sl.sum().item()), 1)
            sl = (sl.float() * scale).clamp(min=1, max=max_len).to(torch.int32)
            diff = total_out - int(sl.sum().item())
            sl[0] = max(1, min(max_len, int(sl[0].item()) + diff))
            tensors['seq_lens'] = sl
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
