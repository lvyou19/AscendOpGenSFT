import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    GDN-2 (Gated DeltaNet 2) chunkwise 前向标杆。
    对标 fla-org/flash-linear-attention @ main 的 chunk_gdn2
    (fla/ops/gdn2/chunk.py:195, Triton 管线在 chunk_fwd.py /
    chunk_intra.py / wy_fast.py), 数学定义见官方参考
    fla/ops/gdn2/naive.py (naive_recurrent_gdn2 / naive_chunk_gdn2)。

    数学定义 (逐 token, 矩阵状态 S ∈ R^{K×V}, * 为逐通道 Hadamard 积):
        S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1}
              + k_t (w_t * v_t)^T
        o_t = (scale * q_t)^T S_t          (读出发生在写入之后)
    展开形式 (与标杆实现逐步对应):
        S <- S * exp(g_t)                    (K 轴逐通道衰减)
        erase = (b_t * k_t)^T S              (擦除门读取衰减后状态)
        v_new = w_t * v_t - erase            (写入门 delta 规则)
        S <- S + k_t ⊗ v_new                 (秩一更新)
        o_t = (scale * q_t)^T S
    b = w = beta (标量) 时退化为 KDA; g 再退化为标量时为 GDN v1。

    布局约定:
        q, k, g, b   [B, T, H, K]   q/k 为 bf16/fp16; g 为 fp32 自然对数域
                                    衰减 (恒 <= 0); b 为擦除门 [B,T,H,K]
        v, w         [B, T, H, V]   v 为 bf16/fp16; w 为写入门 (fp32 计算)
        initial_state [N, H, K, V]  fp32, 形状 [0,...] 表示无初始状态
        cu_seqlens   [N+1] int32 可选 (varlen, 此时 B=1); None 表示等长 batch
        scale        float; use_qk_l2norm_in_kernel int (0/1)
        输出         o [B, T, H, V] (v.dtype), final_state [N, H, K, V] fp32

    use_qk_l2norm_in_kernel=1 时 kernel 内先做 L2 归一化
    (fla/modules/l2norm.py: y = x * rsqrt(sum(x^2) + 1e-6)), 标杆同样先归一化。

    与 kernel 的差异:
        - kernel 内部按 chunk_size=64 分块 + WY 表示计算, 并在 log2 域做
          cumsum; 标杆按逐 token 精确递推 (官方 naive 参考的同义形式),
          二者数学等价, 仅有浮点累加顺序差异。
        - use_gate_in_kernel (A_log/dt_bias/safe_gate 门激活) 不属于本算子
          的递推主体, 标杆不覆盖 (该门计算见 FusedGdnGating 算子);
          标杆的 g 直接是 log 域衰减。
        - state_v_first=False 固定 (状态恒为 [N, H, K, V])。
        - 标杆内部 fp32 计算, 输出舍入回 v.dtype; final_state 恒 fp32。
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _segment_fwd(q, k, v, g, b, w, S, scale):
        # q,k,g,b [X,T,H,K]; v,w [X,T,H,V]; S [X,H,K,V] (fp32)
        device = q.device
        T = q.shape[1]
        V = v.shape[-1]
        o = torch.empty(q.shape[0], T, q.shape[2], V, dtype=torch.float32, device=device)
        for t in range(T):
            S = S * g[:, t].exp().unsqueeze(-1)
            erase = ((b[:, t] * k[:, t]).unsqueeze(-1) * S).sum(-2)
            v_new = w[:, t] * v[:, t] - erase
            S = S + k[:, t].unsqueeze(-1) * v_new.unsqueeze(-2)
            o[:, t] = ((scale * q[:, t]).unsqueeze(-1) * S).sum(-2)
        return o, S

    def forward(self, q, k, v, g, b, w, initial_state, cu_seqlens, scale,
                use_qk_l2norm_in_kernel):
        device = q.device
        B, T, H, K = q.shape
        V = v.shape[-1]
        qf, kf = q.float(), k.float()
        if use_qk_l2norm_in_kernel:
            # fla l2norm: y = x * rsqrt(sum(x^2) + 1e-6)
            qf = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
            kf = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
        vf, gf, bf, wf = v.float(), g.float(), b.float(), w.float()

        has_init = initial_state is not None and initial_state.shape[0] > 0
        if cu_seqlens is not None and len(cu_seqlens) > 0:
            assert B == 1, "varlen 要求 B=1"
            N = len(cu_seqlens) - 1
            o = torch.empty(1, T, H, V, dtype=torch.float32, device=device)
            final = torch.empty(N, H, K, V, dtype=torch.float32, device=device)
            for n in range(N):
                s, e = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
                S0 = (initial_state[n:n + 1].float() if has_init
                      else torch.zeros(1, H, K, V, device=device))
                o_n, S_n = self._segment_fwd(
                    qf[:, s:e], kf[:, s:e], vf[:, s:e],
                    gf[:, s:e], bf[:, s:e], wf[:, s:e], S0, scale)
                o[:, s:e] = o_n
                final[n] = S_n[0]
        else:
            N = B
            S0 = initial_state.float() if has_init else torch.zeros(B, H, K, V, device=device)
            o, S_n = self._segment_fwd(qf, kf, vf, gf, bf, wf, S0, scale)
            final = S_n
        return o.to(v.dtype), final


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "94_Gdn2Fwd.json")
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
        "int32": torch.int32,
    }

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 固定随机种子保证标杆可复现: 每个 case 独立种子, 重复调用结果完全一致
        torch.manual_seed(3407 + case_idx)

        shapes, dtypes = {}, {}
        cu_seqlens, scale, use_l2norm = None, None, 1
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "cu_seqlens":
                cu_seqlens = inp["value"]
            elif name == "scale":
                scale = inp["value"]
            elif name == "use_qk_l2norm_in_kernel":
                use_l2norm = inp["value"]

        B, T, H, K = shapes["q"]
        V = shapes["v"][-1]
        N = len(cu_seqlens) - 1 if cu_seqlens else B
        if cu_seqlens:
            assert B == 1 and cu_seqlens[-1] == T
        assert shapes["initial_state"][0] in (0, N)

        q = random_tensor(shapes["q"], dtypes["q"])
        k = random_tensor(shapes["k"], dtypes["k"])
        v = random_tensor(shapes["v"], dtypes["v"])
        # 物理约束: use_qk_l2norm_in_kernel=0 时上游必须先对 q/k 做 L2 归一化
        # (否则 delta rule 的 (I - k*(b*k)^T) 在 ||k|| 大时特征值越界发散,
        # kernel 本身也会 nan), 这里按真实模型行为预归一化
        if not use_l2norm:
            q = F.normalize(q.float(), p=2, dim=-1).to(dtypes["q"])
            k = F.normalize(k.float(), p=2, dim=-1).to(dtypes["k"])
        # 物理约束: g 是 log 域逐通道衰减, 恒 <= 0 (真实模型来自
        # -exp(A_log)*softplus(...))。沿用官方测试生成器 _rand_inputs 的
        # uniform(-5, -0.1), dtype 跟随输入 (tests/ops/test_gdn2.py)
        g = torch.empty(shapes["g"], dtype=torch.float32).uniform_(-5.0, -0.1).to(dtypes["g"])
        # 物理约束: b (擦除门) 与 w (写入门) 是 sigmoid 类门控, 恒在 (0, 1)
        b = torch.rand(shapes["b"], dtype=dtypes["b"])
        w = torch.rand(shapes["w"], dtype=dtypes["w"])
        if shapes["initial_state"][0] == 0:
            initial_state = torch.zeros(0, H, K, V, dtype=torch.float32)
        else:
            # 初始状态是历史递推结果, 幅值与 v 同量级; 标准分布生成即可
            initial_state = random_tensor(shapes["initial_state"],
                                          torch.float32)

        cu_t = torch.tensor(cu_seqlens, dtype=torch.int32) if cu_seqlens else None
        input_groups.append([q, k, v, g, b, w, initial_state, cu_t,
                             scale, use_l2norm])
    return input_groups


def get_init_inputs():
    return []