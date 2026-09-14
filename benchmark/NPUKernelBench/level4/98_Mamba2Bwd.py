import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import os


class Model(nn.Module):
    """
    Mamba-2 SSD chunk-scan 反向标杆。
    对标 state-spaces/mamba @ main 的 _mamba_chunk_scan_combined_bwd
    (mamba_ssm/ops/triton/ssd_combined.py:397, 由
    MambaChunkScanCombinedFn.backward :596 调用), 返回
    (dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states),
    支持 dfinal_states (autograd 的 args[0], return_final_states=True
    时末态梯度回传)。前向数学定义见 Mamba2Fwd.py。

    反向 (逐 token 精确伴随, dS 从 dfinal_states 初始化, 逆序传播;
    记 a_t = exp(A_h * dt'_t), 写 W_t = dt'_t * x_t ⊗ B_t):
        dy    = dout_t * silu(z_t)
        dz_t  = dout_t * y_t * silu'(z_t)     silu'(z)=σ(z)(1+z(1-σ(z)))
        dC_t  = S_t^T · dy ; dS += dy ⊗ C_t   (读出路径)
        dD   += dy ⊙ x_t ; dx_t = D ⊙ dy      (D 跳连路径)
        dx_t += dt'_t * (dS · B_t)            (写入路径)
        dB_t  = dt'_t * (dS^T · x_t)
        ddt'_t += sum_{p,n} dS * x_t ⊗ B_t
        ddA_t = a_t * sum(dS ⊙ S_{t-1})       (∂S_t/∂dA_t = a_t ⊙ S_{t-1})
        dA_h += ddA_t * dt'_t ; ddt'_t += ddA_t * A_h   (衰减路径)
        dS   <- a_t ⊙ dS                      (传给上一步)
        dinitial_states = 递推结束时的 dS
    dt 预处理链反传 (与 _chunk_cumsum_bwd 一致):
        d(raw) = ddt' * [lo <= sp <= hi] * (σ(raw) 若 raw<=20 否则 1),
        sp = softplus(raw), raw = dt + dt_bias
        ddt = d(raw), ddt_bias[h] = sum_{b,t} d(raw)

    布局约定:
        前 11 个输入同 Mamba2Fwd + 反向梯度 (x, dt, A, B, C, D, z,
        dt_bias, initial_states, dout, dfinal_states);
        随后 8 个为 get_input_groups() 预计算的前向中间量 (均 fp32,
        forward 内不再做任何前向计算):
            S_all [T,B,H,P,N]  逐步前向状态 S_t (t=0..T-1)
            y_all [B,T,H,P]    前向输出 y (含 D 残差, 供 dz 使用)
            a     [B,T,H]      a_t = exp(A * dt'_t)
            dtr   [B,T,H]      dt' = clamp(softplus(dt+dt_bias), lo, hi)
            raw   [B,T,H]      raw = dt + dt_bias   (dt 链反传用)
            sp    [B,T,H]      softplus(raw) 或未加 softplus 的 raw
            sil   [B,T,H,P]    silu(z);  z 缺省时为空张量 [0]
            dsil  [B,T,H,P]    silu'(z); z 缺省时为空张量 [0]
        chunk_size / dt_softplus / dt_limit_lo / dt_limit_hi 同前向
        输出 (与 kernel 返回顺序一致):
            dx [B,T,H,P] (x.dtype), ddt [B,T,H] (dt.dtype),
            dA [H] fp32, dB/dC [B,T,G,N] (B/C.dtype, GQA 组内各头
            梯度求和), dD 同 D 形状 fp32, dz [B,T,H,P] (z.dtype,
            z 缺省时为空张量 [0,...], 对应 kernel 的 None),
            ddt_bias [H] fp32, dinitial_states [B,H,P,N] fp32
            (initial_states 缺省时为空张量 [0,...])
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, dt, A, B, C, D, z, dt_bias, initial_states,
                dout, dfinal_states,
                S_all, y_all, a, dtr, raw, sp, sil, dsil,
                chunk_size, dt_softplus, dt_limit_lo, dt_limit_hi):
        device = x.device
        Bsz, T, H, P = x.shape
        G, N = B.shape[2], B.shape[3]
        R = H // G

        xf = x.float()
        Bh = B.float().repeat_interleave(R, dim=2)   # [B,T,H,N]
        Ch = C.float().repeat_interleave(R, dim=2)
        has_z = z.shape[0] > 0
        has_init = initial_states.shape[0] > 0
        has_dD = D.numel() > 0
        Af = A.float()
        Df = D.float() if has_dD else None
        Dv = (Df.view(1, H, 1) if Df.dim() == 1
              else Df.view(1, H, P)) if has_dD else None

        # S0 仅作 t=0 时的 S_prev (递推初值, 来自输入张量的类型转换)
        S0 = (initial_states.float() if has_init
              else torch.zeros(Bsz, H, P, N, device=device))

        # 反向递推 (S_all/y_all/a/dtr/raw/sp/sil/dsil 均为预计算输入)
        dof = dout.float()
        dxf = torch.zeros_like(xf)
        dBh = torch.zeros_like(Bh)
        dCh = torch.zeros_like(Ch)
        ddtp = torch.zeros_like(dtr)                 # dL/d(dt')
        dAf = torch.zeros_like(Af)
        dDf = torch.zeros_like(Df) if has_dD else None
        dzf = (torch.zeros(Bsz, T, H, P, device=device)
               if has_z else None)
        dS = (dfinal_states.float().clone() if dfinal_states.shape[0] > 0
              else torch.zeros(Bsz, H, P, N, device=device))

        for t in reversed(range(T)):
            dy = dof[:, t] * sil[:, t] if has_z else dof[:, t]
            if has_z:
                dzf[:, t] = dof[:, t] * y_all[:, t] * dsil[:, t]
            S_t = S_all[t]
            dCh[:, t] = (S_t * dy.unsqueeze(-1)).sum(-2)            # [B,H,N]
            dS = dS + dy.unsqueeze(-1) * Ch[:, t].unsqueeze(-2)     # dy ⊗ C_t
            if has_dD:
                acc = dy * xf[:, t]                                 # [B,H,P]
                dDf += acc.sum((0, 2)) if Df.dim() == 1 else acc.sum(0)
                dxf[:, t] = Dv * dy
            # S_t = a ⊙ S_{t-1} + dt' · x ⊗ B
            dxf[:, t] += dtr[:, t].unsqueeze(-1) * (dS * Bh[:, t].unsqueeze(-2)).sum(-1)
            dBh[:, t] = dtr[:, t].unsqueeze(-1) * (dS * xf[:, t].unsqueeze(-1)).sum(-2)
            ddtp[:, t] += (dS * (xf[:, t].unsqueeze(-1)
                                 * Bh[:, t].unsqueeze(-2))).sum((-1, -2))
            S_prev = S_all[t - 1] if t > 0 else S0
            ddA = a[:, t] * (dS * S_prev).sum((-1, -2))             # [B,H]
            dAf += (ddA * dtr[:, t]).sum(0)
            ddtp[:, t] += ddA * Af
            dS = a[:, t].unsqueeze(-1).unsqueeze(-1) * dS           # 传给上一步
        dinitial = dS

        # dt 预处理链反传
        unclamped = (sp >= dt_limit_lo) & (sp <= dt_limit_hi)
        draw = ddtp * unclamped
        if dt_softplus:
            draw = draw * torch.where(raw <= 20.0, torch.sigmoid(raw),
                                      torch.ones_like(raw))
        ddt_bias = draw.sum((0, 1))                                 # [H]

        # GQA: 组内各头梯度求和, 恢复 [B,T,G,N]
        dB = dBh.view(Bsz, T, G, R, N).sum(3).to(B.dtype)
        dC = dCh.view(Bsz, T, G, R, N).sum(3).to(C.dtype)

        dz_out = dzf.to(z.dtype) if has_z else torch.empty(0, device=device)
        dinit_out = dinitial if has_init else torch.empty(0, H, P, N, device=device)
        return (dxf.to(x.dtype), draw.to(dt.dtype), dAf, dB, dC,
                dDf if has_dD else torch.empty(0, device=device),
                dz_out, ddt_bias, dinit_out)


def _forward_intermediates(x, dt, A, B, C, D, z, dt_bias, initial_states,
                           dt_softplus, dt_limit_lo, dt_limit_hi):
    """前向计算 (原 forward 内的重算部分), 全部在输入生成阶段完成。

    返回 (均 fp32):
        S_all [T,B,H,P,N]  逐步状态 S_t
        y_all [B,T,H,P]    前向输出 y (含 D 残差)
        a     [B,T,H]      exp(A * dt')
        dtr   [B,T,H]      clamp 后的 dt'
        raw   [B,T,H]      dt + dt_bias
        sp    [B,T,H]      softplus(raw) 或 raw
        sil   [B,T,H,P]    silu(z), 无 z 时为 [0]
        dsil  [B,T,H,P]    silu'(z), 无 z 时为 [0]
    """
    Bsz, T, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    R = H // G

    xf = x.float()
    Bh = B.float().repeat_interleave(R, dim=2)   # [B,T,H,N]
    Ch = C.float().repeat_interleave(R, dim=2)
    has_z = z.shape[0] > 0
    has_init = initial_states.shape[0] > 0
    has_dD = D.numel() > 0

    # dt 预处理链 (与 _chunk_cumsum_fwd 一致)
    raw = dt.float() + dt_bias.float()           # [B,T,H]
    sp = F.softplus(raw) if dt_softplus else raw
    dtr = sp.clamp(dt_limit_lo, dt_limit_hi)
    a = (dtr * A.float()).exp()                  # [B,T,H]
    Dv = None
    if has_dD:
        Df = D.float()
        Dv = Df.view(1, H, 1) if Df.dim() == 1 else Df.view(1, H, P)

    # 前向递推, 保存 S_t 与 y_t (含 D 残差, 供 dz 使用)
    S = (initial_states.float().clone() if has_init
         else torch.zeros(Bsz, H, P, N))
    S_all = torch.empty(T, Bsz, H, P, N)
    y_all = torch.empty(Bsz, T, H, P)
    for t in range(T):
        S = (a[:, t].unsqueeze(-1).unsqueeze(-1) * S
             + (dtr[:, t].unsqueeze(-1).unsqueeze(-1)
                * xf[:, t].unsqueeze(-1)) * Bh[:, t].unsqueeze(-2))
        S_all[t] = S
        y_all[:, t] = (S * Ch[:, t].unsqueeze(-2)).sum(-1)
    if has_dD:
        y_all = y_all + xf * Dv.unsqueeze(0)     # [1,1,H,P] 广播

    # z 门控统计量: silu(z) 与 silu'(z)
    if has_z:
        zf = z.float()
        sig = torch.sigmoid(zf)
        sil = zf * sig
        dsil = sig * (1.0 + zf * (1.0 - sig))
    else:
        sil = torch.empty(0)
        dsil = torch.empty(0)

    return S_all, y_all, a, dtr, raw, sp, sil, dsil


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "98_Mamba2Bwd.json")
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
    for case_idx, case in enumerate(cases):
        shapes, dtypes, attrs = {}, {}, {}
        for inp in case["inputs"]:
            if inp.get("type") == "tensor":
                shapes[inp["name"]] = inp["shape"]
                dtypes[inp["name"]] = dtype_map[inp["dtype"]]
            else:
                attrs[inp["name"]] = inp["value"]

        Bsz, T, H, P = shapes["x"]
        G, N = shapes["B"][2], shapes["B"][3]
        assert shapes["dt"] == [Bsz, T, H] and shapes["A"] == [H]
        assert shapes["B"] == shapes["C"] == [Bsz, T, G, N]
        assert H % G == 0
        assert shapes["D"] in ([H], [H, P])
        assert shapes["dt_bias"] == [H]
        assert shapes["dout"] == [Bsz, T, H, P]
        assert shapes["z"][0] in (0, Bsz)
        assert shapes["initial_states"][0] in (0, Bsz)
        assert shapes["dfinal_states"][0] in (0, Bsz)
        chunk_size = int(attrs["chunk_size"])
        dt_softplus = int(attrs["dt_softplus"])
        dt_limit_lo = float(attrs["dt_limit_lo"])
        dt_limit_hi = float(attrs["dt_limit_hi"])

        x = random_tensor(shapes["x"], dtypes["x"])
        dt = random_tensor(shapes["dt"], dtypes["dt"])
        # 物理约束: A = -exp(A_log) 恒负, HF/fla 初始化 A_log ∈ (0, ln16)
        A = (-torch.exp(torch.rand(shapes["A"]) * math.log(16.0))).to(dtypes["A"])
        B = random_tensor(shapes["B"], dtypes["B"])
        C = random_tensor(shapes["C"], dtypes["C"])
        D = random_tensor(shapes["D"], dtypes["D"])
        if shapes["z"][0] == 0:      # rmsnorm 路径: kernel 内无 z 门控
            z = torch.empty(shapes["z"], dtype=dtypes["z"])
        else:
            z = random_tensor(shapes["z"], dtypes["z"])
        # 物理约束: dt_bias 初始化为 inv_softplus(U(0.001,0.1)) 量级, 恒负
        dt_bias = (torch.rand(shapes["dt_bias"]) * 4.0 - 5.0).to(dtypes["dt_bias"])
        if shapes["initial_states"][0] == 0:
            initial_states = torch.empty(shapes["initial_states"],
                                         dtype=dtypes["initial_states"])
        else:
            initial_states = random_tensor(shapes["initial_states"],
                                           dtypes["initial_states"])
        dout = random_tensor(shapes["dout"], dtypes["dout"])
        if shapes["dfinal_states"][0] == 0:   # 末态无梯度回传
            dfinal_states = torch.empty(shapes["dfinal_states"],
                                        dtype=dtypes["dfinal_states"])
        else:
            dfinal_states = random_tensor(shapes["dfinal_states"],
                                          dtypes["dfinal_states"])

        # 前向中间量: 挪到输入生成阶段预计算, forward 内零前向计算
        S_all, y_all, a, dtr, raw, sp, sil, dsil = _forward_intermediates(
            x, dt, A, B, C, D, z, dt_bias, initial_states,
            dt_softplus, dt_limit_lo, dt_limit_hi)

        input_groups.append([x, dt, A, B, C, D, z, dt_bias, initial_states,
                             dout, dfinal_states,
                             S_all, y_all, a, dtr, raw, sp, sil, dsil,
                             chunk_size, dt_softplus, dt_limit_lo, dt_limit_hi])
    return input_groups


def get_init_inputs():
    return []