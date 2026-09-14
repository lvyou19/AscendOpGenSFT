import torch
import torch.nn as nn
import json
import os

LOG2E = 1.4426950408889634


class Model(nn.Module):
    """
    Model that performs batch MLA paged attention (decode / MTP / incremental prefill).
    FlashInfer BatchMLAPagedAttentionWrapper (flashinfer/mla/_core.py):
        wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_len_arr, num_heads,
                     head_dim_ckv, head_dim_kpe, page_size, causal, sm_scale, ...)
        wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache, ...) -> o (, lse)
     deprecated 的 BatchDecodeMlaWithPagedKVCacheWrapper 的继任者,
     新增 q_len > 1 (MTP/spec decode) 与增量因果掩码支持。

    torch 原生小算子拼接参考实现(基于矩阵吸收 Matrix Absorption trick):
    每个请求 b 的第 i 个 query (块内序号) 对每个 head:
        scores[h, j] = sm_scale * (q_nope·ckv_j + q_pe·kpe_j),  j ∈ [0, kv_len)
        causal 时掩码 j > kv_len - qo_len + i   (q 块视为追加在 kv 末尾)
        o = softmax(scores) @ ckv                # ckv 吸收后同时作为 K 和 V
        lse = log2(sum(exp2(log2e * scores)))    # 默认 base-2;
                                                 # return_lse_base_on_e=True 时转自然对数
    注:
      1. sm_scale 官方示例取 1/sqrt(128+64) = 1/sqrt(192) (吸收前维度),
         与旧 wrapper 常用的 1/sqrt(576) 不同, 仅为输入超参, 按 case 给定;
      2. lse 默认 base-2 (与 kernel 内部 log 底数一致), base_on_e=True 时除以 log2(e);
      3. o_scale/ckv_scale/kpe_scale 为 FP8 反量化参数, 非 FP8 场景不纳入;
      4. 页表索引乱序且允许冗余页, 只有 kv_indptr/kv_indices 指定的页参与计算;
      5. backend (fa2/fa3/cutlass/trtllm-gen) 仅 kernel 实现选择, 数学语义一致。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache,
                qo_indptr, kv_indptr, kv_indices, kv_len_arr,
                causal, sm_scale, return_lse=False, return_lse_base_on_e=False):
        total_q, H, Dc = q_nope.shape
        Dp = q_pe.shape[-1]
        page_size = ckv_cache.shape[1]

        q_nope_f, q_pe_f = q_nope.float(), q_pe.float()
        o = torch.zeros(total_q, H, Dc, dtype=torch.float32, device=q_nope.device)
        lse = torch.zeros(total_q, H, dtype=torch.float32, device=q_nope.device)

        qo = qo_indptr.tolist()
        indptr = kv_indptr.tolist()
        indices = kv_indices.tolist()
        kv_lens = kv_len_arr.tolist()

        for b in range(len(kv_lens)):
            qo_len = qo[b + 1] - qo[b]
            kv_len = kv_lens[b]
            npages = (kv_len + page_size - 1) // page_size
            pages = indices[indptr[b]:indptr[b] + npages]
            # 按页表 gather 出该请求的 ckv / kpe: [kv_len, dim]
            ckv = ckv_cache[pages].float().reshape(-1, Dc)[:kv_len]
            kpe = kpe_cache[pages].float().reshape(-1, Dp)[:kv_len]

            for i in range(qo_len):
                s = sm_scale * (q_nope_f[qo[b] + i] @ ckv.t() + q_pe_f[qo[b] + i] @ kpe.t())
                if causal:
                    mask = torch.arange(kv_len, device=q_nope.device) > kv_len - qo_len + i
                    s = s.masked_fill(mask[None, :], float("-inf"))
                # 与 kernel 内部一致按 base-2 计算 softmax
                s2 = s * LOG2E
                m = s2.max(dim=-1, keepdim=True).values
                e = torch.exp2(s2 - m)
                denom = e.sum(dim=-1, keepdim=True)
                o[qo[b] + i] = (e / denom) @ ckv
                lse2 = m.squeeze(-1) + torch.log2(denom.squeeze(-1))
                lse[qo[b] + i] = lse2 / LOG2E if return_lse_base_on_e else lse2

        o = o.to(q_nope.dtype)
        if return_lse:
            return o, lse
        return o


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "76_BatchMLAPagedAttention.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
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
    for case_idx, case in enumerate(cases):
        torch.manual_seed(3407 + case_idx)

        shapes, dtypes = {}, {}
        qo_lens = None
        kv_lens = None
        page_size = None
        causal = None
        sm_scale = None
        return_lse = False
        return_lse_base_on_e = False
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "qo_lens":
                qo_lens = inp["value"]
            elif name == "kv_lens":
                kv_lens = inp["value"]
            elif name == "page_size":
                page_size = inp["value"]
            elif name == "causal":
                causal = inp["value"]
            elif name == "sm_scale":
                sm_scale = inp["value"]
            elif name == "return_lse":
                return_lse = inp["value"]
            elif name == "return_lse_base_on_e":
                return_lse_base_on_e = inp["value"]

        q_nope = random_tensor(shapes["q_nope"], dtypes["q_nope"])
        q_pe = random_tensor(shapes["q_pe"], dtypes["q_pe"])
        qo_indptr = torch.tensor([0] + [sum(qo_lens[:i + 1]) for i in range(len(qo_lens))],
                                 dtype=torch.int32)
        assert shapes["q_nope"][0] == qo_indptr[-1].item()
        npages = [(l + page_size - 1) // page_size for l in kv_lens]
        total_pages = sum(npages) + 3
        perm = torch.randperm(total_pages)
        indptr, indices = [0], []
        off = 0
        for n in npages:
            indices += perm[off:off + n].tolist()
            off += n
            indptr.append(len(indices))
        kv_indptr = torch.tensor(indptr, dtype=torch.int32)
        kv_indices = torch.tensor(indices, dtype=torch.int32)
        kv_len_arr = torch.tensor(kv_lens, dtype=torch.int32)

        assert shapes["ckv_cache"][0] == total_pages and shapes["ckv_cache"][1] == page_size
        ckv_cache = random_tensor(shapes["ckv_cache"], dtypes["ckv_cache"])
        kpe_cache = random_tensor(shapes["kpe_cache"], dtypes["kpe_cache"])

        input_groups.append([q_nope, q_pe, ckv_cache, kpe_cache,
                             qo_indptr, kv_indptr, kv_indices, kv_len_arr,
                             causal, sm_scale, return_lse, return_lse_base_on_e])
    return input_groups


def get_init_inputs():
    return []