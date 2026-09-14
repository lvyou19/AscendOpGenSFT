import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Paged Attention Decoder (v2) �� simulates vLLM-style paged attention.

    Computes attention for multiple sequences using paged KV cache layout
    (block tables). Supports GQA via repeat_interleave on KV heads, and
    optional sliding window masking.

    Inputs:
        query: (num_seqs, num_heads, head_size)
        key_cache: (num_blocks, block_size, num_kv_heads, head_size)
        value_cache: (num_blocks, block_size, num_kv_heads, head_size)
        block_tables: (num_seqs, max_num_blocks_per_seq), int32
        seq_lens: (num_seqs,), int32
        scale: float
        sliding_window: int or None
    """

    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, query, key_cache, value_cache, block_tables, seq_lens, scale, sliding_window=None):
        torch.manual_seed(42)
        device = query.device
        block_tables = block_tables.to(device)
        seq_lens = seq_lens.to(device)

        num_heads = query.shape[1]
        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        max_blocks_per_seq = block_tables.shape[1]

        # 每序列有效 kv 长度，并按页表 gather + padding 成批量稠密 K/V
        kv_lens = torch.clamp(seq_lens.long(), max=max_blocks_per_seq * block_size)
        max_kv_len = int(kv_lens.max().item())
        n_blocks = (max_kv_len + block_size - 1) // block_size
        pages = block_tables[:, :n_blocks].long()                     # [B, nb]
        k = key_cache[pages].reshape(len(kv_lens), n_blocks * block_size,
                                     num_kv_heads, -1)[:, :max_kv_len]  # [B, L, KH, D]
        v = value_cache[pages].reshape(len(kv_lens), n_blocks * block_size,
                                       num_kv_heads, -1)[:, :max_kv_len]

        # GQA: repeat interleave KV heads to match Q heads
        if num_heads != num_kv_heads:
            rep = num_heads // num_kv_heads
            k = torch.repeat_interleave(k, rep, dim=2)
            v = torch.repeat_interleave(v, rep, dim=2)

        # decode 语义: 单 query token 注意全部有效 kv token, 无因果 mask;
        # sliding_window 开启时仅保留每序列最后 sliding_window 个 token
        attn = torch.einsum('bhd,blhd->bhl', query * scale, k).float()
        col = torch.arange(max_kv_len, device=device)
        mask = col[None, :] >= kv_lens[:, None]                       # padding 位置
        if sliding_window is not None:
            mask |= col[None, :] < (kv_lens - sliding_window)[:, None]
        attn.masked_fill_(mask[:, None, :], float('-inf'))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        return torch.einsum('bhl,blhd->bhd', attn, v)


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "69_PagedAttentionV2.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype_):
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            if dtype_ is torch.bfloat16:
                return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype_)
            return torch.normal(mu, sigma, shape, dtype=dtype_)
        else:
            return torch.empty(shape, dtype=dtype_).uniform_(-5.0, 5.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat32": torch.bfloat16, "bfloat16": torch.bfloat16}

        q_info = next(inp for inp in inputs if inp["name"] == "query")
        kc_info = next(inp for inp in inputs if inp["name"] == "key_cache")
        vc_info = next(inp for inp in inputs if inp["name"] == "value_cache")
        bt_info = next(inp for inp in inputs if inp["name"] == "block_tables")
        sl_info = next(inp for inp in inputs if inp["name"] == "seq_lens")
        dtype = dtype_map[q_info["dtype"]]

        query = random_tensor(q_info["shape"], dtype)
        key_cache = random_tensor(kc_info["shape"], dtype)
        value_cache = random_tensor(vc_info["shape"], dtype)

        num_blocks, block_size = kc_info["shape"][0], kc_info["shape"][1]
        num_seqs, max_blocks_per_seq = bt_info["shape"][0], bt_info["shape"][1]
        max_kv_len = min(num_blocks, max_blocks_per_seq) * block_size
        max_kv_len = max(max_kv_len, 1)

        block_tables = torch.randint(0, num_blocks, bt_info["shape"], dtype=torch.int32)
        seq_lens = torch.randint(1, max_kv_len + 1, sl_info["shape"], dtype=torch.int32)
        scale = next(inp for inp in inputs if inp["name"] == "scale")["value"]
        sw_info = next((inp for inp in inputs if inp["name"] == "sliding_window"), None)
        sliding_window = sw_info["value"] if sw_info else None

        input_groups.append([query, key_cache, value_cache, block_tables, seq_lens, scale, sliding_window])
    return input_groups


def get_init_inputs():
    return []
