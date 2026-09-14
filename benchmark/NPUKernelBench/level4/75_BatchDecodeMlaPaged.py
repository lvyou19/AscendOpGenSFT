import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs batch decode MLA attention with paged KV cache.
    FlashInfer BatchDecodeMlaWithPagedKVCacheWrapper (flashinfer/decode.py, 已 deprecated,
    官方建议 BatchMLAPagedAttentionWrapper, 数学语义一致):
        wrapper.run(q_nope, q_pe, paged_ckv_cache, paged_kpe_cache, ...) -> o (, lse)

    torch 原生小算子拼接参考实现(基于矩阵吸收 Matrix Absorption trick):
    每个请求 b 按页表 (kv_indptr/kv_indices/kv_last_page_len) gather 出 kv_len 个 token:
        scores[b,h,j] = sm_scale * (q_nope[b,h]·ckv_j + q_pe[b,h]·kpe_j)
        o[b,h] = softmax(scores) @ ckv          # ckv 吸收后同时作为 K 和 V
        lse[b,h] = logsumexp(scores)            # return_lse=True 时返回
    注:
      1. decode 阶段单 query token 注意全部 kv token, 无因果 mask;
      2. window_left >= 0 时只注意每请求最后 window_left+1 个 token;
      3. logits_soft_cap > 0 时 logits = cap * tanh(logits / cap);
      4. sm_scale 应为 1/sqrt(qk_nope_head_dim + qk_rope_head_dim) = 1/sqrt(576);
      5. 页表索引可以乱序且允许存在未被任何请求引用的冗余页,
         只有 indptr/indices 指定的页参与计算;
      6. q_scale/k_scale/v_scale (FP8) 与 rope_scale/rope_theta (decode 不生效) 未纳入。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q_nope, q_pe, paged_ckv_cache, paged_kpe_cache,
                kv_indptr, kv_indices, kv_last_page_len, sm_scale,
                window_left=-1, logits_soft_cap=0.0, return_lse=False):
        torch.manual_seed(42)
        B, H, Dc = q_nope.shape
        page_size = paged_ckv_cache.shape[1]
        Dr = q_pe.shape[-1]
        device = q_nope.device

        indptr = kv_indptr.long()
        last_len = kv_last_page_len.long()
        npages = indptr[1:] - indptr[:-1]                             # [B]
        kv_lens = (npages - 1) * page_size + last_len                 # [B]
        L = int(kv_lens.max().item())
        P = int(npages.max().item())

        # 页表 gather + padding: 每请求取前 npages 页; cache 为
        # [pages, page_size, Hc, dim], 与原实现一致将 (page, Hc) 展平为
        # token 维后按 kv_len 截断, 拼成 [B, L, *] 稠密 K/V
        Hc = paged_ckv_cache.shape[2]
        j = torch.arange(P, device=device)
        flat = (indptr[:-1, None] + j[None, :]).clamp(max=kv_indices.numel() - 1)
        pages = kv_indices.long()[flat]                               # [B, P]
        ckv = paged_ckv_cache[pages].float() \
            .reshape(B, P * page_size * Hc, Dc)[:, :L]
        kpe = paged_kpe_cache[pages].float() \
            .reshape(B, P * page_size * Hc, Dr)[:, :L]

        # scores = sm_scale * (q_nope·ckv + q_pe·kpe)  (ckv 同时作为 K 和 V)
        s = sm_scale * (q_nope.float() @ ckv.transpose(1, 2)
                        + q_pe.float() @ kpe.transpose(1, 2))         # [B, H, L]
        col = torch.arange(L, device=device)
        pad_mask = col[None, :] >= kv_lens[:, None]                   # [B, L]
        mask = pad_mask.clone()
        if window_left >= 0:
            mask |= col[None, :] < (kv_lens - (window_left + 1))[:, None]
        s = s.masked_fill(mask[:, None, :], float("-inf"))
        if logits_soft_cap and logits_soft_cap > 0:
            # 与原实现一致: mask 在 softcap 之前, -inf 经 tanh 变为 -cap,
            # 被 mask 的有效位置仍参与 softmax; padding 位置显式置 0 排除
            s = logits_soft_cap * torch.tanh(s / logits_soft_cap)
        m = s.max(dim=-1, keepdim=True).values
        e = torch.exp(s - m)
        if logits_soft_cap and logits_soft_cap > 0:
            e = e.masked_fill(pad_mask[:, None, :], 0.0)
        denom = e.sum(dim=-1, keepdim=True)
        o = (e / denom) @ ckv                                         # [B, H, Dc]
        lse = m.squeeze(-1) + torch.log(denom.squeeze(-1))            # [B, H]

        o = o.to(q_nope.dtype)
        if return_lse:
            return o, lse
        return o


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "75_BatchDecodeMlaPaged.json")
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
    }

    input_groups = []
    for case in cases:
        shapes, dtypes = {}, {}
        kv_lens = None
        page_size = None
        sm_scale = None
        window_left = -1
        logits_soft_cap = 0.0
        return_lse = False
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "kv_lens":
                kv_lens = inp["value"]
            elif name == "page_size":
                page_size = inp["value"]
            elif name == "sm_scale":
                sm_scale = inp["value"]
            elif name == "window_left":
                window_left = inp["value"]
            elif name == "logits_soft_cap":
                logits_soft_cap = inp["value"]
            elif name == "return_lse":
                return_lse = inp["value"]

        q_nope = random_tensor(shapes["q_nope"], dtypes["q_nope"])
        q_pe = random_tensor(shapes["q_pe"], dtypes["q_pe"])

        # 构造分页 KV cache 与页表: 页id随机打乱, 并附加 3 个不被引用的冗余页,
        # 模拟真实 serving 中页不连续的场景
        npages = [(l + page_size - 1) // page_size for l in kv_lens]
        total_pages = sum(npages) + 3
        perm = torch.randperm(total_pages)
        indptr, indices, last_len = [0], [], []
        off = 0
        for n, l in zip(npages, kv_lens):
            indices += perm[off:off + n].tolist()
            off += n
            indptr.append(len(indices))
            last_len.append(l - (n - 1) * page_size)
        kv_indptr = torch.tensor(indptr, dtype=torch.int32)
        kv_indices = torch.tensor(indices, dtype=torch.int32)
        kv_last_page_len = torch.tensor(last_len, dtype=torch.int32)

        assert shapes["paged_ckv_cache"][0] == total_pages and shapes["paged_ckv_cache"][1] == page_size
        paged_ckv_cache = random_tensor(shapes["paged_ckv_cache"], dtypes["paged_ckv_cache"])
        paged_kpe_cache = random_tensor(shapes["paged_kpe_cache"], dtypes["paged_kpe_cache"])

        input_groups.append([q_nope, q_pe, paged_ckv_cache, paged_kpe_cache,
                             kv_indptr, kv_indices, kv_last_page_len, sm_scale,
                             window_left, logits_soft_cap, return_lse])
    return input_groups


def get_init_inputs():
    return []