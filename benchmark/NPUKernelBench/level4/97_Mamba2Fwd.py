import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import os


class Model(nn.Module):
    """
    Mamba-2 SSD chunk-scan 前向标杆。
    对标 state-spaces/mamba @ main 的 mamba_chunk_scan_combined
    (mamba_ssm/ops/triton/ssd_combined.py:628, autograd Function 在 :596),
    即 fla-org/flash-linear-attention 的 fla/layers/mamba2.py cuda 路径
    (:433) 所调用的算子; 该层另有同数学的纯 torch 参考
    torch_forward (:585-669)。官方 torch 参考链: ssd_chunk_state.py
    chunk_state_ref + ssd_state_passing.py state_passing_ref (:327) +
    ssd_chunk_scan.py chunk_scan_ref (:1846)。

    数学定义 (逐 token, 对每个 batch b、头 h; GQA: 头 h 用组
    g = h // (H/G) 的 B/C; 状态 S ∈ R^{P×N}, P=headdim, N=dstate):
        1) dt 预处理 (fp32, 与 _chunk_cumsum_fwd 顺序一致):
             raw = dt[b,t,h] + dt_bias[h]
             dt' = softplus(raw)         (dt_softplus=1 时; kernel 阈值
                   raw<=20 用 log(1+exp(raw)), 否则线性 raw, 与
                   F.softplus 默认 threshold=20 完全一致)
             dt' = clamp(dt', dt_limit_lo, dt_limit_hi)
             a[b,t,h] = exp(dt' * A[h])  (A 恒负, = -exp(A_log); 标量衰减)
        2) SSD 状态递推:
             S_0 = initial_states (缺省为 0)
             S_t = a_t * S_{t-1} + dt'_t * (x_t ⊗ B_t)   (外积写入)
             y_t = S_t · C_t + D ⊙ x_t                    (更新后读出)
             out_t = y_t                   (z 缺省 / rmsnorm 路径)
                   = y_t * silu(z_t)       (z 存在时; 门控在 D 残差之后,
                     见 chunk_scan_ref: out = out + x*D 后再 * F.silu(z))

    布局约定:
        x              [B, T, H, P]   bf16/fp16/fp32
        dt             [B, T, H]      raw 值 (未加 bias / 未激活)
        A              [H]            fp32, 恒负
        B, C           [B, T, G, N]   G=ngroups, GQA 组内共享
        D              [H] 或 [H, P]  fp32 (D_has_hdim 时为 [H,P])
        z              [B, T, H, P]   可选; 形状 [0,...] 表示 None
        dt_bias        [H]            fp32
        initial_states [B, H, P, N]   fp32, 可选; 形状 [0,...] 表示 None
        chunk_size     int            kernel 分块大小 (本标杆逐 token
                                      实现, 数学上与分块无关)
        dt_softplus    int (0/1)
        dt_limit_lo/hi float          dt 截断区间, 1e38 代表 +inf
        输出           out [B,T,H,P] (x.dtype),
                       final_states [B,H,P,N] (C.dtype; kernel
                       _state_passing_fwd 以 out_dtype=C.dtype 返回末态)

    """

    def __init__(self):
        super().__init__()

    def forward(self, x, dt, A, B, C, D, z, dt_bias, initial_states,
                chunk_size, dt_softplus, dt_limit_lo, dt_limit_hi):
        device = x.device
        Bsz, T, H, P = x.shape
        G, N = B.shape[2], B.shape[3]
        R = H // G

        xf = x.float()
        Bh = B.float().repeat_interleave(R, dim=2)   # [B,T,H,N]
        Ch = C.float().repeat_interleave(R, dim=2)   # [B,T,H,N]

        # dt 预处理 (fp32, 与 _chunk_cumsum_fwd 完全一致的顺序)
        dtr = dt.float() + dt_bias.float()           # [B,T,H]
        if dt_softplus:
            dtr = F.softplus(dtr)   # threshold=20 ↔ kernel where(dt<=20, sp, dt)
        dtr = dtr.clamp(dt_limit_lo, dt_limit_hi)
        a = (dtr * A.float()).exp()                  # [B,T,H] 标量衰减

        # SSD 逐 token 递推
        if initial_states.shape[0] > 0:
            S = initial_states.float().clone()       # [B,H,P,N]
        else:
            S = torch.zeros(Bsz, H, P, N, device=device)
        ys = []
        for t in range(T):
            S = (a[:, t].unsqueeze(-1).unsqueeze(-1) * S
                 + (dtr[:, t].unsqueeze(-1).unsqueeze(-1)
                    * xf[:, t].unsqueeze(-1)) * Bh[:, t].unsqueeze(-2))
            ys.append((S * Ch[:, t].unsqueeze(-2)).sum(-1))   # [B,H,P]
        y = torch.stack(ys, dim=1)                   # [B,T,H,P]

        # D 跳连 (在 z 门控之前); D [H] 按头广播, [H,P] 逐通道
        if D.numel() > 0:
            Df = D.float()
            Dv = Df.view(1, 1, H, 1) if Df.dim() == 1 else Df.view(1, 1, H, P)
            y = y + xf * Dv
        # z 门控 (silu, D 残差之后)
        if z.shape[0] > 0:
            y = y * F.silu(z.float())

        return y.to(x.dtype), S.to(C.dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "97_Mamba2Fwd.json")
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
        assert shapes["z"][0] in (0, Bsz)
        assert shapes["initial_states"][0] in (0, Bsz)
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

        input_groups.append([x, dt, A, B, C, D, z, dt_bias, initial_states,
                             chunk_size, dt_softplus, dt_limit_lo, dt_limit_hi])
    return input_groups


def get_init_inputs():
    return []