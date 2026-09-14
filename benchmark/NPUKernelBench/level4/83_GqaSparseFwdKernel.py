import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        sm_scale,
    ):
        torch.manual_seed(42)
        total_q, num_heads, head_dim = q.shape
        num_kv_heads = kv_cache.shape[1]
        if num_heads % num_kv_heads != 0:
            raise ValueError("query heads must be divisible by KV heads")
        group_size = num_heads // num_kv_heads
        output = torch.empty_like(q)
        batch = cu_seqlens_q.numel() - 1

        for request in range(batch):
            q_start = int(cu_seqlens_q[request].item())
            q_end = int(cu_seqlens_q[request + 1].item())
            seq_len = int(seq_lens[request].item())
            prefix_len = int(prefix_lens[request].item())
            q_request = q[q_start:q_end]
            positions = torch.arange(seq_len, device=q.device, dtype=torch.long)
            logical_blocks = torch.div(positions, 128, rounding_mode="floor")
            pages = block_table[request, logical_blocks].long()
            rows = positions.remainder(128)
            kv_request = kv_cache[pages, :, rows]
            keys = kv_request[..., :head_dim].float()
            values = kv_request[..., head_dim:].float()
            query_positions = prefix_len + torch.arange(
                q_end - q_start, device=q.device, dtype=torch.long
            )
            causal = positions.unsqueeze(0) <= query_positions.unsqueeze(1)

            for kv_head in range(num_kv_heads):
                selected = topk_idx[kv_head, q_start:q_end].long()
                selected_mask = (
                    logical_blocks[None, :, None] == selected[:, None, :]
                ).any(dim=-1)
                mask = causal & selected_mask
                head_start = kv_head * group_size
                head_end = head_start + group_size
                query_group = q_request[:, head_start:head_end].float()
                query_group = query_group.permute(1, 0, 2)
                key_group = keys[:, kv_head].transpose(0, 1)
                key_group = key_group.unsqueeze(0).expand(
                    group_size, -1, -1
                )
                scores = torch.matmul(query_group, key_group)
                scores = scores.permute(1, 0, 2) * sm_scale
                scores = scores.masked_fill(
                    ~mask[:, None, :], -float("inf")
                )
                probabilities = F.softmax(scores, dim=-1)
                result = torch.matmul(probabilities, values[:, kv_head])
                output[q_start:q_end, head_start:head_end] = result.to(q.dtype)

        return output


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(spec, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    tensor = torch.randn(
        tuple(spec["shape"]), generator=generator, dtype=torch.float32
    ) * 0.25
    return tensor.to(dtype=_DTYPE_MAP[spec["dtype"]]).npu()


def _split_lengths(total, count):
    base, remainder = divmod(total, count)
    return [base + (1 if index < remainder else 0) for index in range(count)]


def _metadata(specs, case_index):
    total_q = specs["q"]["shape"][0]
    batch, max_blocks = specs["block_table"]["shape"]
    num_kv_heads, _, topk = specs["topk_idx"]["shape"]
    q_lens = _split_lengths(total_q, batch)
    capacity = max_blocks * 128
    seq_lens = []
    for request, q_len in enumerate(q_lens):
        span = max(1, min(97, capacity - q_len + 1))
        margin = (case_index * 17 + request * 23) % span
        seq_lens.append(max(q_len, capacity - margin))
    prefix_lens = [
        seq_len - q_len
        for seq_len, q_len in zip(seq_lens, q_lens)
    ]
    cu_seqlens = [0]
    for length in q_lens:
        cu_seqlens.append(cu_seqlens[-1] + length)

    generator = torch.Generator()
    generator.manual_seed(1000 + case_index)
    pages = batch * max_blocks
    permutation = torch.randperm(
        pages, generator=generator, dtype=torch.int64
    ).to(torch.int32)
    block_table = permutation.reshape(batch, max_blocks)
    topk_idx = torch.full(
        (num_kv_heads, total_q, topk), -1, dtype=torch.int32
    )
    q_offset = 0
    for request, q_len in enumerate(q_lens):
        prefix = prefix_lens[request]
        for local_query in range(q_len):
            absolute_position = prefix + local_query
            current_block = absolute_position // 128
            valid_count = min(topk, current_block + 1)
            selected = torch.arange(
                current_block,
                current_block - valid_count,
                -1,
                dtype=torch.int32,
            )
            topk_idx[:, q_offset + local_query, :valid_count] = selected
        q_offset += q_len
    return (
        topk_idx.npu(),
        block_table.npu(),
        torch.tensor(cu_seqlens, dtype=torch.int32).npu(),
        torch.tensor(seq_lens, dtype=torch.int32).npu(),
        torch.tensor(prefix_lens, dtype=torch.int32).npu(),
    )


def _load_cases():
    path = os.path.splitext(__file__)[0] + ".json"
    with open(path, "r", encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_input_groups():
    torch.manual_seed(42)
    groups = []
    for case_index, case in enumerate(_load_cases()):
        specs = {item["name"]: item for item in case["inputs"]}
        topk_idx, block_table, cu_seqlens, seq_lens, prefix_lens = (
            _metadata(specs, case_index)
        )
        groups.append([
            _random_tensor(specs["q"], 42 + case_index * 2),
            _random_tensor(specs["kv_cache"], 43 + case_index * 2),
            topk_idx,
            block_table,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            specs["sm_scale"]["value"],
        ])
    return groups


def get_init_inputs():
    return []
