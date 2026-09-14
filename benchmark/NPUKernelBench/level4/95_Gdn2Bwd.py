import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    GDN-2 (Gated DeltaNet 2) chunkwise 反向标杆。
    对标 fla-org/flash-linear-attention @ main 的 chunk_gdn2_bwd
    (fla/ops/gdn2/chunk_bwd.py, 由 chunk.py:133 ChunkGDN2Function.backward
    调用), 数学定义为前向递推 (fla/ops/gdn2/naive.py) 的精确伴随。

    前向 (逐 token, 矩阵状态 S ∈ R^{K×V}, * 为 Hadamard 积):
        S_dec = Diag(exp(g_t)) S_{t-1}
        erase = (b_t * k_t)^T S_dec
        v_new = w_t * v_t - erase
        S_t   = S_dec + k_t ⊗ v_new
        o_t   = (scale * q_t)^T S_t
    反向 (对偶递推, dS 从 dht 初始化, 逆序传播; 下标 t 为当前步):
        dqs_t    = S_t @ do_t ; dq_t = scale * dqs_t
        dS      += (scale * q_t) ⊗ do_t
        dk_t     = dS @ v_new_t                      (秩一更新路径)
        dv_new   = dS^T @ k_t
        dw_t     = dv_new * v_t ; dv_t = dv_new * w_t
        derase   = -dv_new
        dbk      = S_dec_t @ derase                  (擦除门读取路径)
        dS_dec   = dS + (b_t * k_t) ⊗ derase
        db_t     = dbk * k_t ; dk_t += dbk * b_t
        dg_t     = sum_V(dS_dec * S_dec_t)           (衰减路径)
        dS      <- dS_dec * exp(g_t)                 (传给上一步)
        dh0      = 递推结束时的 dS

    布局约定:
        q, k, g, b   [B, T, H, K]; v, w [B, T, H, V] (同前向)
        do           [B, T, H, V]  bf16/fp16, 输出梯度
        dht          [N, H, K, V]  fp32, 末状态梯度; [0,...] 表示无
        initial_state [N, H, K, V] fp32; [0,...] 表示无
        输出         dq, dk, db (q.dtype), dv, dw (v.dtype), dg (g.dtype),
                     dh0 [N, H, K, V] fp32

    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, g, b, w, initial_state, do, dht,
                scale, use_qk_l2norm_in_kernel,
                S_dec_stack, v_new_stack, S_final, o,
                cu_seqlens=None):
        """
        参数：
          q,k,v,g,b,w : 原始输入叶子张量（fp32，不要求梯度）
          initial_state : [B,H,K,V] 或 [N,H,K,V] fp32 或 None
          do : [B,T,H,V] 输出梯度系数
          dht : [B,H,K,V] 或 [N,H,K,V] fp32 或 None（末状态梯度）
          scale, use_qk_l2norm_in_kernel : 前向使用的超参数
          S_dec_all : list of length T，每步衰减后的状态 [B,H,K,V] fp32
          v_new_all : list of length T，每步新写入门后的向量 [B,H,V] fp32
          S_final, o : 前向最终状态/输出（未使用，仅保持接口）
          cu_seqlens : varlen 前缀和，None 表示等长 [B,T]
        返回：
          dq, dk, dv, dg, db, dw, dh0  (均为 fp32)
        """
        B, T, H, K = q.shape
        V = v.shape[-1]

        qf, kf, vf = q.float(), k.float(), v.float()
        gf, bf, wf = g.float(), b.float(), w.float()
        dof = do.float()

        if use_qk_l2norm_in_kernel:
            q_rstd = torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
            k_rstd = torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
            qn = qf * q_rstd
            kn = kf * k_rstd
        else:
            qn, kn = qf, kf
            q_rstd = k_rstd = None

        dq = torch.zeros_like(qf)
        dk = torch.zeros_like(kf)
        dv = torch.zeros_like(vf)
        dg = torch.zeros_like(gf)
        db = torch.zeros_like(bf)
        dw = torch.zeros_like(wf)

        has_init = initial_state is not None and initial_state.shape[0] > 0
        has_dht = dht is not None and dht.shape[0] > 0

        if cu_seqlens is not None and len(cu_seqlens) > 0:
            # varlen: B 必须为 1，按 segment 独立反向
            N = len(cu_seqlens) - 1
            dh0 = torch.zeros(N, H, K, V, dtype=torch.float32, device=q.device)
            for n in range(N):
                s, e = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
                dS0 = dht[n:n + 1].float() if has_dht else torch.zeros(1, H, K, V, device=q.device)
                r = self._segment_bwd(
                    qn[:, s:e], kn[:, s:e], vf[:, s:e],
                    gf[:, s:e], bf[:, s:e], wf[:, s:e],
                    dof[:, s:e], S_dec_stack, v_new_stack,
                    s, e, dS0, scale)
                dq[:, s:e], dk[:, s:e], dv[:, s:e] = r[0], r[1], r[2]
                dg[:, s:e], db[:, s:e], dw[:, s:e] = r[3], r[4], r[5]
                dh0[n] = r[6][0]
        else:
            dS0 = dht.float() if has_dht else torch.zeros(B, H, K, V, device=q.device)
            dq, dk, dv, dg, db, dw, dh0 = self._segment_bwd(
                qn, kn, vf, gf, bf, wf, dof,
                S_dec_stack, v_new_stack, 0, T, dS0, scale)

        if use_qk_l2norm_in_kernel:
            dq = q_rstd * (dq - qn * (dq * qn).sum(-1, keepdim=True))
            dk = k_rstd * (dk - kn * (dk * kn).sum(-1, keepdim=True))

        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                dg.to(g.dtype), db.to(b.dtype), dw.to(w.dtype), dh0)

    @staticmethod
    def _segment_bwd(qn, kn, vf, gf, bf, wf, dof,
                     S_dec_stack, v_new_stack, s, e, dS0, scale):
        """对 token 区间 [s, e) 执行反向递推。"""
        dq = torch.zeros_like(qn)
        dk = torch.zeros_like(kn)
        dv = torch.zeros_like(vf)
        dg = torch.zeros_like(gf)
        db = torch.zeros_like(bf)
        dw = torch.zeros_like(wf)
        dS = dS0
        for t in range(e - 1, s - 1, -1):
            S_dec = S_dec_stack[t]
            v_new = v_new_stack[t]
            S_t = S_dec + kn[:, t - s].unsqueeze(-1) * v_new.unsqueeze(-2)
            dq[:, t - s] = scale * (S_t * dof[:, t - s].unsqueeze(-2)).sum(-1)
            dS = dS + (scale * qn[:, t - s]).unsqueeze(-1) * dof[:, t - s].unsqueeze(-2)
            dk_t = (dS * v_new.unsqueeze(-2)).sum(-1)
            dv_new = (dS * kn[:, t - s].unsqueeze(-1)).sum(-2)
            dw[:, t - s] = dv_new * vf[:, t - s]
            dv[:, t - s] = dv_new * wf[:, t - s]
            derase = -dv_new
            dbk = (S_dec * derase.unsqueeze(-2)).sum(-1)
            dS_dec = dS + (bf[:, t - s] * kn[:, t - s]).unsqueeze(-1) * derase.unsqueeze(-2)
            db[:, t - s] = dbk * kn[:, t - s]
            dk_t = dk_t + dbk * bf[:, t - s]
            dg[:, t - s] = (dS_dec * S_dec).sum(-1)
            dS = dS_dec * gf[:, t - s].unsqueeze(-1).exp()
            dk[:, t - s] = dk_t
        return dq, dk, dv, dg, db, dw, dS


def get_input_groups():
    """
    生成所有测试用例的输入数据，执行完整前向递推，
    并保存每步的 S_dec 和 v_new，连同最终状态 S_final 和输出 o 一起返回。
    返回元组顺序：
      (q, k, v, g, b, w, initial_state, do, dht,
       scale, use_l2norm,
       S_dec_all, v_new_all, S_final, o)
    其中 S_dec_all, v_new_all 为列表，包含每个时间步的中间量。
    """
    json_path = os.path.join(os.path.dirname(__file__), "95_Gdn2Bwd.json")
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
        "int32": torch.int32,
    }

    input_groups = []
    for case_idx, case in enumerate(cases):
        torch.manual_seed(3407 + case_idx)

        shapes, dtypes = {}, {}
        cu_seqlens, scale, use_l2norm = None, None, 1
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "cu_seqlens":
                cu_seqlens = torch.tensor(inp["value"], dtype=torch.int64)
            elif name == "scale":
                scale = inp["value"]
            elif name == "use_qk_l2norm_in_kernel":
                use_l2norm = inp["value"]

        B, T, H, K = shapes["q"]
        V = shapes["v"][-1]

        # ---- 生成叶子张量（q/k 做 L2 归一化，与真实 GDN2 qk-norm 场景对齐） ----
        q = F.normalize(random_tensor(shapes["q"], torch.float32), p=2, dim=-1).to(dtypes["q"])
        k = F.normalize(random_tensor(shapes["k"], torch.float32), p=2, dim=-1).to(dtypes["k"])
        v = random_tensor(shapes["v"], dtypes["v"])
        # 物理约束：g 为 log 域衰减 ≤0；范围收小防止状态指数爆炸
        g = torch.empty(shapes["g"], dtype=torch.float32).uniform_(-1.0, -0.02).to(dtypes["g"])
        # b, w 为 sigmoid 门控，恒 (0,1)
        b = torch.rand(shapes["b"], dtype=dtypes["b"])
        w = torch.rand(shapes["w"], dtype=dtypes["w"])

        # initial_state / dht
        if shapes["initial_state"][0] == 0:
            initial_state = None
        else:
            initial_state = random_tensor(shapes["initial_state"], torch.float32)
        if shapes["dht"][0] == 0:
            dht = None
        else:
            dht = random_tensor(shapes["dht"], torch.float32)

        # do 不要求梯度
        do = random_tensor(shapes["do"], dtypes["do"])

        # ---- 执行前向递推，保存中间量 ----
        # 转为 fp32
        qf = q.float()
        kf = k.float()
        if use_l2norm:
            q_rstd = torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
            k_rstd = torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
            qn = qf * q_rstd
            kn = kf * k_rstd
        else:
            qn = qf
            kn = kf

        vf, gf, bf, wf = v.float(), g.float(), b.float(), w.float()

        # 保存列表
        S_dec_all = []
        v_new_all = []
        o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)

        has_init = initial_state is not None and initial_state.shape[0] > 0

        if cu_seqlens is not None and len(cu_seqlens) > 0:
            # varlen: B 必须为 1，按 segment 独立递推
            N = len(cu_seqlens) - 1
            for n in range(N):
                s, e = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
                S = qf.new_zeros(1, H, K, V)
                if has_init:
                    S = S + initial_state[n:n + 1].float()
                for t in range(s, e):
                    q_t = qn[:, t] * scale
                    k_t = kn[:, t]
                    v_t = vf[:, t]
                    g_t = gf[:, t]
                    b_t = bf[:, t]
                    w_t = wf[:, t]

                    # 衰减
                    S_dec = S * g_t.unsqueeze(-1).exp()
                    # 擦除
                    erase = ((b_t.unsqueeze(-1) * k_t.unsqueeze(-1)) * S_dec).sum(-2)
                    v_new = w_t * v_t - erase
                    # 更新状态
                    S = S_dec + k_t.unsqueeze(-1) * v_new.unsqueeze(-2)
                    # 输出
                    o[:, t] = (q_t.unsqueeze(-1) * S).sum(-2)

                    S_dec_all.append(S_dec)
                    v_new_all.append(v_new)
        else:
            # 初始化状态
            S = qf.new_zeros(B, H, K, V)
            if has_init:
                S = S + initial_state.float()

            for t in range(T):
                q_t = qn[:, t] * scale
                k_t = kn[:, t]
                v_t = vf[:, t]
                g_t = gf[:, t]
                b_t = bf[:, t]
                w_t = wf[:, t]

                # 衰减
                S_dec = S * g_t.unsqueeze(-1).exp()
                # 擦除
                erase = ((b_t.unsqueeze(-1) * k_t.unsqueeze(-1)) * S_dec).sum(-2)
                v_new = w_t * v_t - erase
                # 更新状态
                S = S_dec + k_t.unsqueeze(-1) * v_new.unsqueeze(-2)
                # 输出
                o[:, t] = (q_t.unsqueeze(-1) * S).sum(-2)

                S_dec_all.append(S_dec)
                v_new_all.append(v_new)

        S_final = S

        S_dec_stack = torch.stack(S_dec_all, dim=0)
        v_new_stack = torch.stack(v_new_all, dim=0)

        input_groups.append((q, k, v, g, b, w, initial_state, do, dht,
                             scale, use_l2norm,
                             S_dec_stack, v_new_stack, S_final, o, cu_seqlens))
    return input_groups


def get_init_inputs():
    return []