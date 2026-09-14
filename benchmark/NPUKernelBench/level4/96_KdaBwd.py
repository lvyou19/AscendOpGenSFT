import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    chunk_kda 反向 —— KDA 门控 delta 规则线性注意力的梯度 golden reference。
    采用 **显式反向公式**，前向中间量由 get_input_groups 提供。

    数学语义：
      前向（fp32）：S ← S·exp(g_t)；S ← S + (β_t k_t)⊗(v_t − S^T k_t)；o_t = S^T q_t
      损失：L = <o, do> + <final_state, dht>
    反向递推（从 T-1 到 0）：
      dS 初始化为 dht（若存在）或零
      对每个 t：
        S_t = S_dec_t + β_t * k_t ⊗ v_new_t
        dq_t = scale * (S_t @ do_t)          # q 梯度
        dS += scale * q_t ⊗ do_t             # 来自输出路径
        # 更新路径
        dk_t += β_t * (dS @ v_new_t)
        dv_new = β_t * (dS^T @ k_t)
        dβ_t = (dS * (k_t ⊗ v_new_t)).sum()  # β 是标量（每个头）
        # 擦除路径
        dv_t = dv_new
        dS_dec = dS - k_t ⊗ dv_new            # 因为 v_new = v_t - S_dec^T k_t
        # 衰减路径
        dg_t = (dS_dec * S_dec_t).sum(dim=-1) # [B,HV,K]
        dS = dS_dec * exp(g_t)                # 传给上一步
      dh0 = dS（若 h0 存在）

    布局约定：同原 kernel，但 q/k 在 forward 中会进行 GVA 扩展（repeat_interleave）。
    返回梯度：dq,dk,dv,dg,dbeta,dh0（均回铸输入 dtype，dh0 为 fp32）。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, v, g, beta, do, dht, scale, h0,
                S_dec_stack, v_new_stack):
        """
        参数：
          q,k,v,g,beta: 原始输入叶子张量（fp32，不要求梯度）
          do: 输出梯度系数 [B,T,HV,V]
          dht: 末状态梯度 [B,HV,K,V] 或 None
          scale: 缩放因子
          h0: 初始状态 [B,HV,K,V] 或 None
          S_dec_stack: [T,B,HV,K,V] 每步衰减后的状态 fp32
          v_new_stack: [T,B,HV,V]   每步擦除后的向量 fp32
        返回：
          dq, dk, dv, dg, dbeta, dh0
        """
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        G = HV // H

        qf, kf, vf = q.float(), k.float(), v.float()
        gf, bf = g.float(), beta.float()
        dof = do.float()

        q_exp = qf.repeat_interleave(G, dim=2) * scale
        k_exp = kf.repeat_interleave(G, dim=2)

        dq_exp = torch.zeros_like(q_exp)
        dk_exp = torch.zeros_like(k_exp)
        dv = torch.zeros_like(vf)
        dg = torch.zeros_like(gf)
        dbeta = torch.zeros_like(bf)

        if dht is not None:
            dS = dht.float().clone()
        else:
            dS = torch.zeros(B, HV, K, V, device=q.device)
        for t in range(T - 1, -1, -1):
            S_dec = S_dec_stack[t]
            v_new = v_new_stack[t]

            beta_t = bf[:, t]
            k_t = k_exp[:, t]
            S_t = S_dec + beta_t.unsqueeze(-1).unsqueeze(-1) * k_t.unsqueeze(-1) * v_new.unsqueeze(-2)
            do_t = dof[:, t]
            dq_exp[:, t] = scale * (S_t @ do_t.unsqueeze(-1)).squeeze(-1)
            dS = dS + scale * q_exp[:, t].unsqueeze(-1) * do_t.unsqueeze(-2)
            dk_exp[:, t] = beta_t.unsqueeze(-1) * (dS @ v_new.unsqueeze(-1)).squeeze(-1)
            dv_new = beta_t.unsqueeze(-1) * (dS.transpose(-2, -1) @ k_t.unsqueeze(-1)).squeeze(-1)
            dbeta[:, t] = (dS * (k_t.unsqueeze(-1) * v_new.unsqueeze(-2))).sum(dim=(-2, -1))
            dv[:, t] = dv_new
            dS_dec = dS - k_t.unsqueeze(-1) * dv_new.unsqueeze(-2)
            g_t = gf[:, t]
            dg[:, t] = (dS_dec * S_dec).sum(dim=-1)
            dS = dS_dec * g_t.unsqueeze(-1).exp()

        dq = dq_exp.reshape(B, T, H, G, K).sum(dim=3)
        dk = dk_exp.reshape(B, T, H, G, K).sum(dim=3)

        if h0 is not None:
            dh0 = dS
        else:
            dh0 = torch.empty(0, HV, K, V, dtype=torch.float32, device=q.device)
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                dg.to(g.dtype), dbeta.to(beta.dtype), dh0)

_GATE_LOGIT_NORMALIZER = [1.0, 1.0, 1.0, 1.0, 10.0, 0.1]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "96_KdaBwd.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def random_tensor(shape, dtype):
        if torch.rand(1).item() < 0.5:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)
        else:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)

    input_groups = []
    for case_idx, case in enumerate(cases):
        torch.manual_seed(3407 + case_idx)

        inputs = case["inputs"]

        def info(name):
            return next(i for i in inputs if i["name"] == name)

        q_dtype = dtype_map[info("q")["dtype"]]
        v_dtype = dtype_map[info("v")["dtype"]]
        scale = info("scale")["value"]
        B, T, H, K = info("q")["shape"]
        HV, V = info("v")["shape"][2], info("v")["shape"][3]
        G = HV // H
        q_float = F.normalize(random_tensor((B, T, H, K), torch.float32), p=2, dim=-1)
        q = q_float.to(q_dtype)
        k_float = F.normalize(random_tensor((B, T, H, K), torch.float32), p=2, dim=-1)
        k = k_float.to(q_dtype)
        v = random_tensor(info("v")["shape"], v_dtype)
        g = (F.logsigmoid(random_tensor(info("g")["shape"], torch.float32))
             / _GATE_LOGIT_NORMALIZER[case_idx % len(_GATE_LOGIT_NORMALIZER)])
        g = g.to(dtype_map[info("g")["dtype"]])
        beta = torch.rand(info("beta")["shape"], dtype=torch.float32).to(dtype_map[info("beta")["dtype"]])
        h0_shape = info("initial_state")["shape"]
        if h0_shape[0] == 0:
            h0 = None
        else:
            h0 = random_tensor(h0_shape, torch.float32)
        do = random_tensor(info("do")["shape"], v_dtype)
        dht = random_tensor(info("dht")["shape"], torch.float32)
        qf = q.float().repeat_interleave(G, dim=2) * scale
        kf = k.float().repeat_interleave(G, dim=2)
        vf, gf, bf = v.float(), g.float(), beta.float()

        S = qf.new_zeros(B, HV, K, V)
        if h0 is not None:
            S = S + h0

        S_dec_list = []
        v_new_list = []
        for t in range(T):
            k_t = kf[:, t]       # [B,HV,K]
            v_t = vf[:, t]       # [B,HV,V]
            g_t = gf[:, t]       # [B,HV,K]
            b_t = bf[:, t]       # [B,HV]

            S_dec = S * g_t.unsqueeze(-1).exp()          # 衰减
            # 擦除
            v_new = v_t - (S_dec.transpose(-2, -1) @ k_t.unsqueeze(-1)).squeeze(-1)  # [B,HV,V]
            # 更新
            S = S_dec + b_t.unsqueeze(-1).unsqueeze(-1) * k_t.unsqueeze(-1) * v_new.unsqueeze(-2)

            S_dec_list.append(S_dec)
            v_new_list.append(v_new)

        # 返回 (输入叶子, do, dht, scale, h0, 中间量栈)
        S_dec_stack = torch.stack(S_dec_list, dim=0)
        v_new_stack = torch.stack(v_new_list, dim=0)
        input_groups.append((q, k, v, g, beta, do, dht, scale, h0, S_dec_stack, v_new_stack))
    return input_groups


def get_init_inputs():
    return []