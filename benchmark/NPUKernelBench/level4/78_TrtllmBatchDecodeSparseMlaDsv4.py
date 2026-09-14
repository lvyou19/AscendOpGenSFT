import torch
import torch.nn as nn
import json
import os

DSV4_HEAD_DIM = 512
DSV4_SWA_TOPK = 128  # sparse_indices 前 128 列固定为 SWA 段


class Model(nn.Module):
    """
    Model that performs TRTLLM-GEN DeepSeek-V4 sparse MLA decode.
    FlashInfer trtllm_batch_decode_sparse_mla_dsv4 (flashinfer/mla/_core.py):
        trtllm_batch_decode_sparse_mla_dsv4(query, swa_kv_cache, workspace_buffer,
            sparse_indices, compressed_kv_cache, sparse_topk_lens, seq_lens, out,
            bmm1_scale, bmm2_scale, sinks, kv_layout, cum_seq_lens_q, max_q_len,
            enable_pdl, swa_topk_lens, extra_sparse_indices, extra_sparse_topk_lens)
        -> out (bf16)

    torch 原生小算子拼接参考实现(与官方单测参考 ref_sparse_attn_decode
    (tests/attention/test_trtllm_gen_sparse_mla_dsv4.py) 逐项对照):
    query/kv 的 head_dim 均为 512 (DSv4 无 rope 拆分), 两个分页 KV 池:
      swa_kv_cache       — 滑窗(SWA)池, 保留序列尾部 token;
      compressed_kv_cache — 压缩池, 存放 indexer 选中的压缩 token。
    sparse_indices [sum_q, 128 + topk] 为扁平 token 槽索引
    (= physical_page * page_size + page_offset), -1 表示无效:
      前 128 列索引 SWA 池, 其余列索引压缩池。
    sparse_topk_lens [sum_q] 为该 query token 的总有效长度, 已包含固定 128 个
    SWA 项, 故压缩段有效数 = sparse_topk_lens - 128 (按列前缀截取)。
    每个 query token j 对每个 head:
        kv      = concat(gather(swa_pool, idx[:128]), gather(comp_pool, idx[128:]))
        logits  = q_j @ kv^T * bmm1_scale        # 无效项 -inf
        lse     = logsumexp(logits)
        o       = exp(logits - lse) @ kv         # (= softmax(logits) @ kv)
        若 sinks 非空: o *= 1 / (1 + exp(sinks - lse))   # 等价 softmax 分母加 exp(sinks)
        o      *= bmm2_scale
    注:
      1. 输出恒为 BF16 (kernel 语义), 内部 fp32 计算;
      2. 掩码完全由 -1 索引与 sparse_topk_lens 决定 (同官方参考); seq_lens 供
         kernel 推导 SWA-128 有效窗口/校验 (要求 seq_lens >= q_len), 标杆不改变
         掩码语义, 真实 SWA 因果窗口已由索引生成侧的 -1 填充表达;
      3. 无任何有效项的 query token 输出为 0;
      4. workspace_buffer/out/max_q_len/enable_pdl/kv_layout 仅为 kernel 调度与
         内存布局参数, 不影响数学语义 (kv_layout 不同仅改变池的物理排布, 扁平化
         后等价); SM120 packed uint8/FP8 路径未纳入, query 取 BF16;
      5. 支持 cum_seq_lens_q 的 ragged query 输入 [sum_q, H, 512]。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, swa_kv_cache, compressed_kv_cache, sparse_indices,
                sparse_topk_lens, seq_lens, bmm1_scale=1.0, bmm2_scale=1.0,
                sinks=None, cum_seq_lens_q=None):
        D = DSV4_HEAD_DIM
        # 任意分页布局 (HND/NHD, 3D/4D) 扁平化为 [num_pages * page_size, 512]
        swa_flat = swa_kv_cache.reshape(-1, D).float()
        comp_flat = compressed_kv_cache.reshape(-1, D).float()

        if query.ndim == 4:
            B, q_len, H, _ = query.shape
            sum_q = B * q_len
            out_shape = (B, q_len, H, D)
        else:
            B = cum_seq_lens_q.numel() - 1 if cum_seq_lens_q is not None else None
            sum_q, H, _ = query.shape
            out_shape = (sum_q, H, D)

        q_flat = query.reshape(sum_q, H, D).float()
        o_flat = torch.zeros(sum_q, H, D, dtype=torch.float32, device=query.device)

        for j in range(sum_q):
            idx = sparse_indices[j].long()
            topk_len = int(sparse_topk_lens[j].item())
            swa_idx = idx[:DSV4_SWA_TOPK]
            comp_idx = idx[DSV4_SWA_TOPK:]

            kv_parts, valid_parts = [], []
            # SWA 段: 掩码仅由 -1 (及越界保护) 决定; gather 上下界均需夹取,
            # 否则越界正索引会在被掩码排除前先触发 IndexError
            swa_valid = (swa_idx >= 0) & (swa_idx < swa_flat.shape[0])
            kv_parts.append(swa_flat[swa_idx.clamp(0, swa_flat.shape[0] - 1)])
            valid_parts.append(swa_valid)
            # 压缩段: -1 掩码 且 列号 < sparse_topk_lens - 128 (前缀有效)
            if comp_idx.numel() > 0:
                comp_valid_len = max(topk_len - DSV4_SWA_TOPK, 0)
                comp_valid = (comp_idx >= 0) & (comp_idx < comp_flat.shape[0])
                comp_valid &= (
                    torch.arange(comp_idx.numel(), device=comp_idx.device)
                    < comp_valid_len
                )
                kv_parts.append(comp_flat[comp_idx.clamp(0, comp_flat.shape[0] - 1)])
                valid_parts.append(comp_valid)

            kv = torch.cat(kv_parts, dim=0)          # [L, 512]
            valid = torch.cat(valid_parts, dim=0)    # [L]
            if not bool(valid.any()):
                continue                             # 无有效项 -> 输出 0
            kv_v = kv[valid]

            q = q_flat[j]                            # [H, 512]
            logits = (q @ kv_v.t()) * bmm1_scale     # [H, Lv]
            lse = torch.logsumexp(logits, dim=-1)    # [H]
            out = torch.exp(logits - lse.unsqueeze(-1)) @ kv_v

            if sinks is not None:
                # 等价于 softmax 分母附加 exp(sinks): o *= 1/(1+exp(sinks-lse))
                sink_scale = 1.0 / (1.0 + torch.exp(sinks.float() - lse))
                sink_scale = torch.where(torch.isfinite(sink_scale), sink_scale,
                                         torch.zeros_like(sink_scale))
                out = out * sink_scale.unsqueeze(-1)

            o_flat[j] = out * bmm2_scale

        return o_flat.reshape(out_shape).to(torch.bfloat16)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__),
                             "78_TrtllmBatchDecodeSparseMlaDsv4.json")
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
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,  # sparse_indices/sparse_topk_lens/seq_lens/sinks 由逻辑生成
    }

    def make_block_table(npages, total_pages):
        # 页 id 随机打乱的稠密页表, 模拟真实分页分配
        perm = torch.randperm(total_pages)
        max_pages = max(npages)
        bt = torch.zeros(len(npages), max_pages, dtype=torch.int64)
        off = 0
        for b, n in enumerate(npages):
            bt[b, :n] = perm[off:off + n]
            off += n
        return bt

    def abs_to_flat(abs_idx, bt_row, ps):
        # 绝对 token 位置 -> 扁平 token 槽索引 (页内映射), -1 保持不变
        safe = abs_idx.clamp_min(0)
        flat = bt_row[safe // ps] * ps + (safe % ps)
        flat[abs_idx < 0] = -1
        return flat

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 固定随机种子保证标杆可复现: 每个 case 独立种子, 重复调用结果完全一致
        torch.manual_seed(3407 + case_idx)

        shapes, dtypes = {}, {}
        kv_lens = qo_lens = None
        swa_ps = comp_ps = comp_ratio = comp_topk = None
        bmm1_scale, bmm2_scale = 1.0, 1.0
        has_sinks = False
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "kv_lens":
                kv_lens = inp["value"]
            elif name == "qo_lens":
                qo_lens = None if inp["value"] is None else inp["value"]
            elif name == "swa_page_size":
                swa_ps = inp["value"]
            elif name == "compressed_page_size":
                comp_ps = inp["value"]
            elif name == "compressed_ratio":
                comp_ratio = inp["value"]
            elif name == "compressed_topk":
                comp_topk = inp["value"]
            elif name == "bmm1_scale":
                bmm1_scale = inp["value"]
            elif name == "bmm2_scale":
                bmm2_scale = inp["value"]
            elif name == "has_sinks":
                has_sinks = inp["value"]

        B = len(kv_lens)
        D = DSV4_HEAD_DIM
        H = shapes["query"][-2]
        capacity = DSV4_SWA_TOPK + comp_topk
        assert capacity % 4 == 0

        if qo_lens is None:
            q_len = shapes["query"][1]
            qo_lens = [q_len] * B
            cum_seq_lens_q = None
            sum_q = B * q_len
        else:
            cum_seq_lens_q = torch.tensor(
                [0] + [sum(qo_lens[:i + 1]) for i in range(B)], dtype=torch.int32)
            sum_q = sum(qo_lens)
        assert all(l >= q for l, q in zip(kv_lens, qo_lens))  # kernel 校验要求

        query = random_tensor(shapes["query"], dtypes["query"])
        swa_kv_cache = random_tensor(shapes["swa_kv_cache"], dtypes["swa_kv_cache"])
        compressed_kv_cache = random_tensor(shapes["compressed_kv_cache"],
                                            dtypes["compressed_kv_cache"])
        seq_lens = torch.tensor(kv_lens, dtype=torch.int32)

        # 分页池: 每请求页数 + 3 个冗余未引用页, 页 id 随机打乱
        swa_npages = [(l + swa_ps - 1) // swa_ps for l in kv_lens]
        swa_total = sum(swa_npages) + 3
        assert shapes["swa_kv_cache"][0] == swa_total
        swa_bt = make_block_table(swa_npages, swa_total)

        c_lens = [(l + comp_ratio - 1) // comp_ratio for l in kv_lens] \
            if comp_topk > 0 else [0] * B
        comp_npages = [max((l + comp_ps - 1) // comp_ps, 0) for l in c_lens]
        if comp_topk > 0:
            comp_total = sum(comp_npages) + 3
            assert shapes["compressed_kv_cache"][0] == comp_total
            comp_bt = make_block_table(comp_npages, comp_total)
        else:
            comp_bt = None

        # sparse_indices [sum_q, 128 + comp_topk]
        sparse_indices = torch.full((sum_q, capacity), -1, dtype=torch.int32)
        sparse_topk_lens = torch.full((sum_q,), DSV4_SWA_TOPK, dtype=torch.int32)
        for b in range(B):
            kv_len, q_len_b = kv_lens[b], qo_lens[b]
            for t in range(q_len_b):
                j = (b * qo_lens[0] + t) if cum_seq_lens_q is None \
                    else (int(cum_seq_lens_q[b].item()) + t)
                # SWA 段 (官方因果窗生成): token_idx = kv_len - q_len + t,
                # 取最后 min(128, token_idx+1) 个绝对位置, 其余 -1
                token_idx = kv_len - q_len_b + t
                n_swa = min(DSV4_SWA_TOPK, token_idx + 1)
                abs_swa = torch.arange(token_idx - n_swa + 1, token_idx + 1,
                                       dtype=torch.int64)
                flat_swa = abs_to_flat(abs_swa, swa_bt[b], swa_ps)
                sparse_indices[j, :DSV4_SWA_TOPK] = torch.cat([
                    flat_swa.int(),
                    torch.full((DSV4_SWA_TOPK - n_swa,), -1, dtype=torch.int32)])
                # 压缩段: 从该请求压缩 token 中无放回随机采样, 有效数随 (b,t) 变化
                if comp_topk > 0:
                    base = min(comp_topk, c_lens[b])
                    n_comp = max(base - (b + t) % 4, 0)
                    if n_comp > 0:
                        abs_comp = torch.randperm(c_lens[b])[:n_comp]
                        flat_comp = abs_to_flat(abs_comp, comp_bt[b], comp_ps)
                        sparse_indices[j, DSV4_SWA_TOPK:
                                       DSV4_SWA_TOPK + n_comp] = flat_comp.int()
                    sparse_topk_lens[j] = DSV4_SWA_TOPK + n_comp

        sinks = random_tensor([H], torch.float32) if has_sinks else None

        input_groups.append([query, swa_kv_cache, compressed_kv_cache,
                             sparse_indices, sparse_topk_lens, seq_lens,
                             bmm1_scale, bmm2_scale, sinks, cum_seq_lens_q])
    return input_groups


def get_init_inputs():
    return []