import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    Model that performs chunked KDA intra-chunk backward computation.
    chunk_kda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg,
                        cu_seqlens, chunk_indices, chunk_size, safe_gate)
        -> (dq_out, dk_out, db_out, dg_out)

    torch 原生小算子拼接参考实现(与 Triton kernel 数学语义一致, 已用 autograd 闭环验证):
    每条序列内按 chunk_size 分块, 块内局部位置 t, s, 逐通道门控 g (log2 域
    chunk 内 inclusive cumsum, 与前向 chunk_kda_fwd_intra 的 gk 同一约定),
    记 E[t,s,d] = 2^(g[t,d]-g[s,d]) (s <= t, 其余为 0), q/k 经 i_h = i_hv // (HV//H) 映射:
        dq2[t,d] = sum_{s<=t} dAqk[t,s] * k[s,d] * E[t,s,d]
        dkA[t,d] = sum_{s<=t} dAkk[t,s] * k[s,d] * E[t,s,d]        (未乘 beta)
        dkt[t,d] = sum_{s>=t} (dAqk[s,t]*q[s,d] + dAkk[s,t]*beta_s*k[s,d]) * E[s,t,d]
        dq_out = dq + dq2
        dk_out = dk + beta_t * dkA + dkt
        db_out = db + sum_d dkA * k                                 (fp32)
        dg_out = dg + q * dq2 + (beta_t * dkA - dkt) * k            (fp32)
    注:
      1. dg 是"对自然对数门控 cumsum 的梯度": kernel 全程 exp2, 链式求导的 ln2
         与下游 chunk_local_cumsum 反向的 RCP_LN2 相消, 故 kernel/golden 均不含 ln2;
      2. dAkk 语义是对前向"原始 Akk(未求逆)"的梯度(经上游逆矩阵梯度恒等式换算后传入);
      3. safe_gate 仅改变 kernel 内部数值路径, 数学语义不变;
      4. q/k 输入为 L2 归一化向量, 与前向算子及真实 KDA 场景(qk-norm)一致;
      5. 衰减矩阵 E 为前向中间量, 由 get_input_groups() 预计算后传入 (fp32),
         forward 内不再出现任何前向计算; 布局为稠密填充 [B, T, BT, HV, K]:
         E[b, t, j, hv, d] = 2^(g[b,t,hv,d] - g[b,c+j,hv,d]) (c 为 t 所在
         chunk 起点, j <= t-c, 其余为 0), 与 dAqk 的 [B,T,HV,BT] 约定一致;
      6. g 保留在签名中与 kernel 参数对齐, forward 内不再使用。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, E,
                cu_seqlens=None, chunk_indices=None, chunk_size=64, safe_gate=False):
        B, T, H, K = k.shape
        HV = g.shape[2]
        BT = chunk_size
        G = HV // H

        q_flat = q.float().reshape(B * T, H, K)
        k_flat = k.float().reshape(B * T, H, K)
        b_flat = beta.float().reshape(B * T, HV)
        dAqk_flat = dAqk.float().reshape(B * T, HV, BT)
        dAkk_flat = dAkk.float().reshape(B * T, HV, BT)
        dq_flat = dq.float().reshape(B * T, HV, K)
        dk_flat = dk.float().reshape(B * T, HV, K)
        db_flat = db.float().reshape(B * T, HV)
        dg_flat = dg.float().reshape(B * T, HV, K)
        E_flat = E.float().reshape(B * T, BT, HV, K)   # 预计算前向中间量

        dq_out = dq_flat.clone()
        dk_out = dk_flat.clone()
        db_out = db_flat.clone()
        dg_out = dg_flat.clone()

        if cu_seqlens is None:
            bounds = [(b * T, (b + 1) * T) for b in range(B)]
        else:
            cu = cu_seqlens.tolist()
            bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]

        head_map = torch.arange(HV, device=k.device) // G

        for (t0, t1) in bounds:
            c = t0
            while c < t1:
                e = min(c + BT, t1)  # 不足 BT 的尾块按实际长度
                L = e - c
                q_c = q_flat[c:e][:, head_map]       # [L, HV, K]
                k_c = k_flat[c:e][:, head_map]
                b_c = b_flat[c:e]                    # [L, HV]
                dAq_c = dAqk_flat[c:e, :, :L]        # [L, HV, L]
                dAk_c = dAkk_flat[c:e, :, :L]
                E_c = E_flat[c:e, :L]                # [L(t), L(s), HV, K]

                dq2 = torch.einsum('ths,shd,tshd->thd', dAq_c, k_c, E_c)
                dkA = torch.einsum('ths,shd,tshd->thd', dAk_c, k_c, E_c)
                db2 = torch.einsum('thd,thd->th', dkA, k_c)
                kb = k_c * b_c[..., None]
                dkt = torch.einsum('sht,shd,sthd->thd', dAq_c, q_c, E_c) + \
                    torch.einsum('sht,shd,sthd->thd', dAk_c, kb, E_c)

                dq_out[c:e] += dq2
                dk_out[c:e] += b_c[..., None] * dkA + dkt
                db_out[c:e] += db2
                dg_out[c:e] += q_c * dq2 + (b_c[..., None] * dkA - dkt) * k_c
                c = e

        dq_out = dq_out.reshape(B, T, HV, K).to(dq.dtype)
        dk_out = dk_out.reshape(B, T, HV, K).to(dk.dtype)
        db_out = db_out.reshape(B, T, HV).float()   # kernel: db 以 fp32 累加返回
        dg_out = dg_out.reshape(B, T, HV, K).float()
        return dq_out, dk_out, db_out, dg_out


def _decay_matrix(g, cu_seqlens, chunk_size):
    """前向衰减矩阵 (原 forward 内的 E 构造, 挪到输入生成阶段)。

    返回 fp32 稠密填充张量 [B, T, BT, HV, K]:
        E[b, t, j, hv, d] = 2^(g[b,t,hv,d] - g[b,c+j,hv,d])
        c 为 t 所在 chunk 的起点, 仅 j <= t-c 有效, 块外及 s > t 处为 0。
    """
    B, T, HV, K = g.shape
    BT = chunk_size
    g_flat = g.float().reshape(B * T, HV, K)
    E = torch.zeros(B * T, BT, HV, K)

    if cu_seqlens is None:
        bounds = [(b * T, (b + 1) * T) for b in range(B)]
    else:
        cu = cu_seqlens.tolist()
        bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]

    for (t0, t1) in bounds:
        c = t0
        while c < t1:
            e = min(c + BT, t1)
            L = e - c
            g_c = g_flat[c:e]                            # [L, HV, K]
            D = g_c.unsqueeze(1) - g_c.unsqueeze(0)      # [L(t), L(s), HV, K]
            lower = torch.tril(torch.ones(L, L, dtype=torch.bool))
            E[c:e, :L] = torch.where(lower[:, :, None, None],
                                     torch.exp2(D), torch.zeros(1))
            c = e
    return E.reshape(B, T, BT, HV, K)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "73_ChunkKdaBwdIntra.json")
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
        # 先解析 attr (g 的 cumsum 依赖 cu_seqlens 和 chunk_size)
        shapes, dtypes = {}, {}
        cu_seqlens = None
        chunk_size = 64
        chunk_indices = None
        safe_gate = False
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "cu_seqlens":
                cu_seqlens = torch.tensor(inp["value"], dtype=torch.int64)
            elif name == "chunk_size":
                chunk_size = inp["value"]
            elif name == "chunk_indices":
                chunk_indices = None if inp["value"] is None else \
                    torch.tensor(inp["value"], dtype=torch.int64)
            elif name == "safe_gate":
                safe_gate = inp["value"]

        # q/k: L2 归一化, 与前向算子及真实 KDA 场景(qk-norm)保持一致
        q = F.normalize(random_tensor(shapes["q"], torch.float32), p=2, dim=-1).to(dtypes["q"])
        k = F.normalize(random_tensor(shapes["k"], torch.float32), p=2, dim=-1).to(dtypes["k"])
        # beta: sigmoid 门控系数, 取值 (0, 1)
        beta = torch.empty(shapes["beta"], dtype=dtypes["beta"]).uniform_(0.0, 1.0)
        # 梯度张量: 50% 均匀 + 50% 正态
        dAqk = random_tensor(shapes["dAqk"], dtypes["dAqk"])
        dAkk = random_tensor(shapes["dAkk"], dtypes["dAkk"])
        dq = random_tensor(shapes["dq"], dtypes["dq"])
        dk = random_tensor(shapes["dk"], dtypes["dk"])
        db = random_tensor(shapes["db"], dtypes["db"])
        dg = random_tensor(shapes["dg"], dtypes["dg"])

        # g: kernel 约定输入为 log2 域的 chunk 内 inclusive cumsum。
        # 先生成逐 token 负增量 (log2 域, 保证块内衰减 <= 1),
        # 再按序列边界和 chunk_size 分段做 cumsum, 与前向算子输入分布对齐
        B, T, HV, K = shapes["g"]
        delta = -torch.empty(B, T, HV, K).uniform_(0.01, 0.2)
        if cu_seqlens is None:
            bounds = [(b * T, (b + 1) * T) for b in range(B)]
        else:
            cu = cu_seqlens.tolist()
            bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]
        g = delta.clone()
        delta_flat = delta.reshape(B * T, HV, K)
        g_flat = g.reshape(B * T, HV, K)
        for (t0, t1) in bounds:
            for c0 in range(t0, t1, chunk_size):
                c1 = min(c0 + chunk_size, t1)
                g_flat[c0:c1] = torch.cumsum(delta_flat[c0:c1], dim=0)
        g = g.to(dtypes["g"])

        # 前向中间量: 衰减矩阵 E, 挪到输入生成阶段预计算 (fp32),
        # forward 内零前向计算
        E = _decay_matrix(g, cu_seqlens, chunk_size)     # [B, T, BT, HV, K]

        input_groups.append([q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, E,
                             cu_seqlens, chunk_indices, chunk_size, safe_gate])
    return input_groups


def get_init_inputs():
    return []