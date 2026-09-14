import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    Model that performs chunked KDA intra-chunk forward computation.
    chunk_kda_fwd_intra(q, k, v, gk, beta, scale, cu_seqlens, chunk_size,
                        chunk_indices, safe_gate, disable_recompute)
        -> (w, u, qg, kg, Aqk, Akk)

    torch 原生小算子拼接参考实现(与 Triton kernel 数学语义一致, 已闭环验证):
    每条序列内按 chunk_size 分块, 块内局部位置 t, s, 逐通道门控 gk (log2 域
    chunk 内 inclusive cumsum, 即上游 chunk_local_cumsum 的产物):
        Aqk[t,s] = scale * sum_d q[t,d]*k[s,d]*2^(gk[t,d]-gk[s,d])      (s <= t)
        Akk_raw[t,s] = beta_t * sum_d k[t,d]*k[s,d]*2^(gk[t,d]-gk[s,d]) (s < t)
        Akk = (I + tril(Akk_raw, -1))^{-1}        # 单位下三角逆 (solve_tril)
        w  = Akk @ (k * beta * 2^gk)              # WY 表示
        u  = Akk @ (v * beta)
        kg = k * 2^(gk_last - gk)                 # 衰减到块末, 供块间状态更新
        qg = q * 2^gk                             # 仅 disable_recompute=True 时输出
    注:
      1. Aqk 严格上三角 kernel 侧是 empty 未初始化垃圾值, golden 置 0,
         该部分不参与精度比较;
      2. safe_gate 仅改变 kernel 内部数值路径(防溢出取参考点), 数学语义不变;
      3. q/k 输入必须是 L2 归一化后的向量(真实 KDA 场景层内先做 qk-norm),
         否则 ||k||~sqrt(K) 时 (I+Akk)^{-1} 组合爆炸, kernel 与 golden 同溢出。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, v, gk, beta, scale, cu_seqlens=None, chunk_size=64,
                chunk_indices=None, safe_gate=False, disable_recompute=False):
        torch.manual_seed(42)
        B, T, H, K = k.shape
        HV, V = v.shape[2], v.shape[-1]
        BT = chunk_size
        G = HV // H  # qk 头与 v 头的分组映射: i_h = i_hv // G

        q_f, k_f, v_f = q.float(), k.float(), v.float()
        g_f, b_f = gk.float(), beta.float()

        q_flat = q_f.reshape(B * T, H, K)
        k_flat = k_f.reshape(B * T, H, K)
        v_flat = v_f.reshape(B * T, HV, V)
        g_flat = g_f.reshape(B * T, HV, K)
        b_flat = b_f.reshape(B * T, HV)

        Aqk = torch.zeros(B * T, HV, BT, dtype=torch.float32, device=k.device)
        Akk = torch.zeros(B * T, HV, BT, dtype=torch.float32, device=k.device)
        w = torch.zeros(B * T, HV, K, dtype=torch.float32, device=k.device)
        u = torch.zeros(B * T, HV, V, dtype=torch.float32, device=k.device)
        kg = torch.zeros(B * T, HV, K, dtype=torch.float32, device=k.device)
        qg = torch.zeros(B * T, HV, K, dtype=torch.float32, device=k.device) \
            if disable_recompute else None

        if cu_seqlens is None:
            bounds = [(b * T, (b + 1) * T) for b in range(B)]
        else:
            cu = cu_seqlens.tolist()
            bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]

        head_map = torch.arange(HV, device=k.device) // G

        for (t0, t1) in bounds:
            c = t0
            while c < t1:
                e = min(c + BT, t1)  # chunk 结束(不含), 不足 BT 的尾块按实际长度
                L = e - c
                q_c = q_flat[c:e][:, head_map]           # [L, HV, K]
                k_c = k_flat[c:e][:, head_map]
                g_c = g_flat[c:e]                        # [L, HV, K]
                v_c = v_flat[c:e]                        # [L, HV, V]
                b_c = b_flat[c:e]                        # [L, HV]

                # 逐通道衰减 D[t,s,d] = 2^(gk[t,d]-gk[s,d])
                D = g_c.unsqueeze(1) - g_c.unsqueeze(0)  # [L(t), L(s), HV, K]
                lower = torch.tril(torch.ones(L, L, dtype=torch.bool, device=k.device))
                strict = torch.tril(torch.ones(L, L, dtype=torch.bool, device=k.device), -1)
                zero = torch.zeros(1, device=k.device)
                Dec = torch.where(lower[:, :, None, None], torch.exp2(D), zero)
                DecS = torch.where(strict[:, :, None, None], torch.exp2(D), zero)

                Aqk_l = scale * torch.einsum('thd,shd,tshd->hts', q_c, k_c, Dec)
                Akk_raw = torch.einsum('thd,shd,tshd->hts', k_c, k_c, DecS) * b_c.t()[:, :, None]
                # (I + tril(Akk_raw,-1))^{-1}: 单位下三角, 解下三角系统即得逆
                eye = torch.eye(L, device=k.device).expand(HV, L, L)
                Akk_inv = torch.linalg.solve_triangular(eye + Akk_raw, eye, upper=False)

                kb = k_c * b_c[..., None] * torch.exp2(g_c)  # [L, HV, K]
                vb = v_c * b_c[..., None]                    # [L, HV, V]
                w_l = torch.einsum('hts,shd->thd', Akk_inv, kb)
                u_l = torch.einsum('hts,shv->thv', Akk_inv, vb)
                kg_l = k_c * torch.exp2(g_c[L - 1:L] - g_c)
                if qg is not None:
                    qg[c:e] = q_c * torch.exp2(g_c)

                Aqk[c:e, :, :L] = Aqk_l.permute(1, 0, 2)
                Akk[c:e, :, :L] = Akk_inv.permute(1, 0, 2)
                w[c:e] = w_l
                u[c:e] = u_l
                kg[c:e] = kg_l
                c = e

        dt = k.dtype
        Aqk = Aqk.reshape(B, T, HV, BT).to(dt)
        Akk = Akk.reshape(B, T, HV, BT).to(dt)
        w = w.reshape(B, T, HV, K).to(dt)
        u = u.reshape(B, T, HV, V).to(v.dtype)
        kg = kg.reshape(B, T, HV, K).to(dt)
        if qg is not None:
            qg = qg.reshape(B, T, HV, K).to(dt)
        return w, u, qg, kg, Aqk, Akk


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "72_ChunkKdaFwdIntra.json")
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
        # 先解析 attr (gk 的 cumsum 依赖 cu_seqlens 和 chunk_size)
        shapes, dtypes = {}, {}
        scale = None
        cu_seqlens = None
        chunk_size = 64
        chunk_indices = None
        safe_gate = False
        disable_recompute = False
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "scale":
                scale = inp["value"]
            elif name == "cu_seqlens":
                cu_seqlens = torch.tensor(inp["value"], dtype=torch.int64)
            elif name == "chunk_size":
                chunk_size = inp["value"]
            elif name == "chunk_indices":
                chunk_indices = None if inp["value"] is None else \
                    torch.tensor(inp["value"], dtype=torch.int64)
            elif name == "safe_gate":
                safe_gate = inp["value"]
            elif name == "disable_recompute":
                disable_recompute = inp["value"]

        # q/k: 必须 L2 归一化 (真实 KDA 场景层内先做 qk-norm; 且 ||k||~sqrt(K) 时
        # (I+Akk)^{-1} 会组合爆炸, kernel 与 golden 同时溢出, 已实测)
        q = F.normalize(random_tensor(shapes["q"], torch.float32), p=2, dim=-1).to(dtypes["q"])
        k = F.normalize(random_tensor(shapes["k"], torch.float32), p=2, dim=-1).to(dtypes["k"])
        v = random_tensor(shapes["v"], dtypes["v"])
        # beta 物理含义是 sigmoid 门控系数, 取值 (0, 1)
        beta = torch.empty(shapes["beta"], dtype=dtypes["beta"]).uniform_(0.0, 1.0)

        # gk: kernel 约定输入为 log2 域的 chunk 内 inclusive cumsum。
        # 这里先生成逐 token 负增量 (log2 域, 保证块内衰减 <= 1),
        # 再按序列边界和 chunk_size 分段做 cumsum, 与上游 chunk_local_cumsum 对齐
        B, T, HV, K = shapes["gk"]
        delta = -torch.empty(B, T, HV, K).uniform_(0.01, 0.2)
        if cu_seqlens is None:
            bounds = [(b * T, (b + 1) * T) for b in range(B)]
        else:
            cu = cu_seqlens.tolist()
            bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]
        gk = delta.clone()
        delta_flat = delta.reshape(B * T, HV, K)
        gk_flat = gk.reshape(B * T, HV, K)
        for (t0, t1) in bounds:
            for c0 in range(t0, t1, chunk_size):
                c1 = min(c0 + chunk_size, t1)
                gk_flat[c0:c1] = torch.cumsum(delta_flat[c0:c1], dim=0)
        gk = gk.to(dtypes["gk"])

        input_groups.append([q, k, v, gk, beta, scale, cu_seqlens, chunk_size,
                             chunk_indices, safe_gate, disable_recompute])
    return input_groups


def get_init_inputs():
    return []