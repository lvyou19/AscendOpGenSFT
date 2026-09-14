import torch
import torch.nn as nn
import json
import os

LOG2E = 1.4426950408889634


class Model(nn.Module):
    """
    Model that performs TRTLLM-GEN batch decode MLA attention.
    FlashInfer trtllm_batch_decode_with_kv_cache_mla (flashinfer/mla/_core.py):
        trtllm_batch_decode_with_kv_cache_mla(query, kv_cache, workspace_buffer,
            qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim, block_tables,
            seq_lens, max_seq_len, sparse_mla_top_k=0, bmm1_scale=..., bmm2_scale=...)
        -> o (, lse)

    torch 原生小算子拼接参考实现(与官方 trace 模板参考一致, 已逐项对照验证):
    query 为 [q_nope | q_pe] 沿 head_dim 拼接 (kv_lora_rank + qk_rope_head_dim = 576),
    kv_cache 为 [ckv | kpe] 拼接的单张量 [num_pages, page_size, 576]。
    每个请求 b 的第 t 个 query 对每个 head:
        logits = (Qn @ Kn^T + Qp @ Kp^T) * bmm1_scale
        o = softmax(logits) @ Kn * bmm2_scale      # ckv 吸收后同时作为 K 和 V
    稠密路径 (sparse_mla_top_k == 0): block_tables 为稠密 2D 页表 [B, max_pages],
        取每请求前 ceil(kv_len/page_size) 页, decode 语义无因果掩码;
    稀疏路径 (sparse_mla_top_k > 0): block_tables 为 [B, q_len, top_k] 稀疏页索引,
        无效项 (<0 或越界) 跳过, 仅对有效页做注意力。
    注:
      1. bmm1_scale 常用 softmax scale (如 1/sqrt(192), 吸收前维度), bmm2_scale 常用 1.0;
      2. lse 约定与 FlashInfer MLA kernel 族一致: 默认 base-2,
         return_lse_base_on_e=True 时转自然对数;
      3. workspace_buffer/max_seq_len/is_var_seq/enable_pdl/backend 等仅为 kernel
         调度参数, 不影响数学语义, 未纳入;
      4. sinks (softmax 分母附加项) 与 skip_softmax 稀疏未纳入;
      5. 支持 cum_seq_lens_q 的 ragged query 输入 [total_q, H, 576]。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, kv_cache, block_tables, seq_lens,
                kv_lora_rank, qk_rope_head_dim,
                bmm1_scale=1.0, bmm2_scale=1.0, sparse_mla_top_k=0,
                cum_seq_lens_q=None, return_lse=False, return_lse_base_on_e=False):
        head_dim_qk = kv_lora_rank + qk_rope_head_dim
        cache = kv_cache.squeeze(1) if kv_cache.dim() == 4 else kv_cache
        page_size = cache.shape[1]
        device = query.device
        R = kv_lora_rank

        # 统一将 query 拉直为 [TQ, H, 576]; req_id 记录每个 q token 所属请求
        if cum_seq_lens_q is None:
            B, q_len, H, _ = query.shape
            q_flat = query.reshape(B * q_len, H, head_dim_qk).float()
            req_id = torch.arange(B, device=device).repeat_interleave(q_len)
            t_id = torch.arange(q_len, device=device).repeat(B)
        else:
            B = cum_seq_lens_q.numel() - 1
            H = query.shape[1]
            q_flat = query.float()
            cu = cum_seq_lens_q.long()
            req_id = torch.repeat_interleave(
                torch.arange(B, device=device), cu[1:] - cu[:-1])
            t_id = torch.arange(q_flat.shape[0], device=device) - cu[:-1].long()[req_id]
        TQ = q_flat.shape[0]
        Qn, Qp = q_flat[..., :R], q_flat[..., R:]

        if sparse_mla_top_k > 0:
            # 稀疏路径: 每个 query token 独立的稀疏页索引, 无效项 (<0/越界) 屏蔽
            pages = block_tables.long()[req_id, t_id]                 # [TQ, topk]
            tok_valid = (pages >= 0) & (pages < cache.shape[0])       # [TQ, topk]
            pg = pages.clamp(0, cache.shape[0] - 1)
            flat = cache[pg].float().reshape(TQ, -1, head_dim_qk)     # [TQ, topk*page, 576]
            kv_mask = tok_valid.repeat_interleave(page_size, dim=1)   # [TQ, S]
        else:
            # 稠密路径: 页表前 n_pages 项, gather + padding 成 [B, L, 576]
            kv_lens = seq_lens.long()
            npages = (kv_lens + page_size - 1) // page_size
            P = int(npages.max().item())
            L = int(kv_lens.max().item())
            pages = block_tables[:, :P].long()                        # [B, P]
            flat = cache[pages].float().reshape(B, P * page_size,
                                                head_dim_qk)[:, :L]   # [B, L, 576]
            kv_mask = (torch.arange(L, device=device)[None, :]
                       < kv_lens[:, None])                            # [B, L]
            kv_mask = kv_mask[req_id]                                 # [TQ, L]
            flat = flat[req_id]                                       # [TQ, L, 576]

        Kn = flat[..., :R]                       # ckv: 同时作为 K 和 V
        Kp = flat[..., R:]                       # kpe: 仅作为 K
        logits = (torch.einsum('thr,tsr->ths', Qn, Kn)
                  + torch.einsum('thr,tsr->ths', Qp, Kp)) * bmm1_scale
        logits = logits.masked_fill(~kv_mask[:, None, :], float("-inf"))

        # 与 kernel 内部一致按 base-2 计算 softmax;
        # 无有效页的行 (仅稀疏路径可能出现) 输出 0 / lse 0
        s2 = logits * LOG2E
        m = s2.amax(dim=-1, keepdim=True)
        m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
        e = torch.exp2(s2 - m)
        denom = e.sum(dim=-1, keepdim=True)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        out = (e / denom) @ Kn * bmm2_scale                           # [TQ, H, R]
        lse2 = m.squeeze(-1) + torch.log2(denom.squeeze(-1))          # [TQ, H]

        o = out.to(query.dtype)
        lse = lse2 / LOG2E if return_lse_base_on_e else lse2
        if cum_seq_lens_q is None:
            o = o.reshape(B, q_len, H, R)
            lse = lse.reshape(B, q_len, H)
        if return_lse:
            return o, lse
        return o


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "77_TrtllmBatchDecodeWithKvCacheMla.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        # 标准输入分布: 50% 均匀 [-5, 5] + 50% 正态 (mu ∈ [-5, 5], sigma ∈ [0.1, 2])
        if torch.rand(1).item() < 0.5:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)
        else:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,  # block_tables 仅声明形状, 实际由页表逻辑生成
    }

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 固定随机种子保证标杆可复现: 每个 case 独立种子, 重复调用结果完全一致
        torch.manual_seed(3407 + case_idx)

        shapes, dtypes = {}, {}
        kv_lens = None
        page_size = None
        kv_lora_rank = None
        qk_rope_head_dim = None
        bmm1_scale = 1.0
        bmm2_scale = 1.0
        sparse_mla_top_k = 0
        qo_lens = None
        return_lse = False
        return_lse_base_on_e = False
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "kv_lens":
                kv_lens = inp["value"]
            elif name == "page_size":
                page_size = inp["value"]
            elif name == "kv_lora_rank":
                kv_lora_rank = inp["value"]
            elif name == "qk_rope_head_dim":
                qk_rope_head_dim = inp["value"]
            elif name == "bmm1_scale":
                bmm1_scale = inp["value"]
            elif name == "bmm2_scale":
                bmm2_scale = inp["value"]
            elif name == "sparse_mla_top_k":
                sparse_mla_top_k = inp["value"]
            elif name == "qo_lens":
                qo_lens = None if inp["value"] is None else inp["value"]
            elif name == "return_lse":
                return_lse = inp["value"]
            elif name == "return_lse_base_on_e":
                return_lse_base_on_e = inp["value"]

        query = random_tensor(shapes["query"], dtypes["query"])
        kv_cache = random_tensor(shapes["kv_cache"], dtypes["kv_cache"])
        seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
        cum_seq_lens_q = None
        if qo_lens is not None:
            cum_seq_lens_q = torch.tensor(
                [0] + [sum(qo_lens[:i + 1]) for i in range(len(qo_lens))], dtype=torch.int32)

        npages = [(l + page_size - 1) // page_size for l in kv_lens]
        total_pages = sum(npages) + 3  # 3 个冗余页, 模拟真实稀疏页分配
        assert shapes["kv_cache"][0] == total_pages and shapes["kv_cache"][-2] == page_size

        if sparse_mla_top_k > 0:
            # 稀疏路径: block_tables [B, q_len, top_k], 每行随机采页, 末尾插 -1 无效项
            B, q_len = shapes["block_tables"][0], shapes["block_tables"][1]
            block_tables = torch.zeros(B, q_len, sparse_mla_top_k, dtype=torch.int32)
            for b in range(B):
                for t in range(q_len):
                    n_valid = sparse_mla_top_k - (b + t) % 2  # 部分行含无效项
                    pages = torch.randperm(total_pages)[:n_valid]
                    block_tables[b, t, :n_valid] = pages.int()
                    block_tables[b, t, n_valid:] = -1
        else:
            # 稠密路径: 稠密 2D 页表 [B, max_pages], 页id随机打乱, 未用列填 0
            max_pages = max(npages)
            perm = torch.randperm(total_pages)
            block_tables = torch.zeros(len(kv_lens), max_pages, dtype=torch.int32)
            off = 0
            for b, n in enumerate(npages):
                block_tables[b, :n] = perm[off:off + n].int()
                off += n
            assert shapes["block_tables"] == [len(kv_lens), max_pages]

        input_groups.append([query, kv_cache, block_tables, seq_lens,
                             kv_lora_rank, qk_rope_head_dim,
                             bmm1_scale, bmm2_scale, sparse_mla_top_k,
                             cum_seq_lens_q, return_lse, return_lse_base_on_e])
    return input_groups


def get_init_inputs():
    return []