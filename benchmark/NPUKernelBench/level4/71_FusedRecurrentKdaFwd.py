import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    Model that performs fused recurrent KDA (gated delta rule) forward computation.
    fused_recurrent_kda_fwd(q, k, v, g, beta, scale, initial_state, ...)
        -> (o, final_state)

    torch 原生小算子拼接参考实现(与 Triton kernel 逐 token 递归语义一致):
        q_t = l2norm(q_t); k_t = l2norm(k_t)   # 可选, use_qk_l2norm_in_kernel
        k_t = k_t * scale
        v_new = (v_t - S @ k_t) * beta_t       # S: [V, K]; beta: 标量/头 或 逐通道 [V]
        S = S * exp(g_t) + v_new ⊗ k_t         # KDA: g_t 为逐通道 log 衰减 [K]
        o_t = S @ q_t                          # [V]
    每条序列的 S 从 initial_state[ssm_state_indices[n]] 载入,
    在序列末尾(或 num_accepted_tokens 指定的已接受 token 处)写回 final_state。
    注: kernel 侧 o = empty_like(k), 隐含约束 H == HV 且 K == V, case 按此生成。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, v, g, beta, scale, initial_state,
                inplace_final_state=True, cu_seqlens=None,
                ssm_state_indices=None, num_accepted_tokens=None,
                use_qk_l2norm_in_kernel=False):
        torch.manual_seed(42)
        B, T, H, K = k.shape
        HV, V = v.shape[2], v.shape[-1]
        G = HV // H  # v 头与 qk 头的分组比 (KDA 中通常 HV == H, G == 1)

        q_f, k_f, v_f = q.float(), k.float(), v.float()
        g_f, beta_f = g.float(), beta.float()

        if initial_state is None:
            if cu_seqlens is None:
                num_states = B
            else:
                num_states = (len(cu_seqlens) - 1) if ssm_state_indices is None \
                    else (int(ssm_state_indices.max().item()) + 1)
            initial_state = torch.zeros(num_states, HV, V, K, dtype=v.dtype, device=v.device)
        else:
            # 若提供的状态数少于序列数, 广播复用最后一个状态
            num_states = initial_state.shape[0]
            target_states = B if cu_seqlens is None else (len(cu_seqlens) - 1)
            if num_states < target_states:
                repeat = (target_states + num_states - 1) // num_states
                initial_state = initial_state.repeat(
                    repeat, *(1 for _ in range(initial_state.ndim - 1)))[:target_states]

        if use_qk_l2norm_in_kernel:
            q_f = F.normalize(q_f, p=2, dim=-1)
            k_f = F.normalize(k_f, p=2, dim=-1)
        k_f = k_f * scale

        # 不与输入共享内存, 避免评测多次调用时 initial_state 被原地污染
        final_state = initial_state.clone()
        head_map = torch.arange(HV, device=v.device) // G

        if cu_seqlens is None:
            # 批量递推: S 带 batch 维 [B, HV, V, K], 仅保留时间维循环
            # （逐 token 递推是 gated delta rule 的数学语义本身, 非分核逻辑）
            q_exp = q_f[:, :, head_map]                    # [B, T, HV, K]
            k_exp = k_f[:, :, head_map]
            S = initial_state[:B].float()                  # [B, HV, V, K]
            o = torch.zeros(B, T, HV, V, dtype=torch.float32, device=v.device)
            accepted = None if num_accepted_tokens is None \
                else num_accepted_tokens.long()
            for t in range(T):
                k_t = k_exp[:, t]                          # [B, HV, K]
                q_t = q_exp[:, t]
                v_new = v_f[:, t] - torch.einsum('bhvk,bhk->bhv', S, k_t)
                v_new = v_new * beta_f[:, t]               # 广播: 标量/头 或 [B, HV, V]
                S = S * torch.exp(g_f[:, t]).unsqueeze(2) \
                    + v_new.unsqueeze(-1) * k_t.unsqueeze(2)
                o[:, t] = torch.einsum('bhvk,bhk->bhv', S, q_t)
                if accepted is not None:
                    # 已接受 token 处写回对应序列的 final_state
                    hit = (accepted == t + 1)[:, None, None, None]
                    final_state[:B] = torch.where(
                        hit, S.to(initial_state.dtype), final_state[:B])
            if accepted is None:
                final_state[:B] = S.to(initial_state.dtype)
            return o.to(v.dtype), final_state

        # varlen（约定 B == 1, T = 总 token 数）: 按 cu_seqlens 分段,
        # 每条序列独立递推; 分段与状态槽位映射是 varlen 的语义边界
        q_flat = q_f.reshape(B * T, H, K)
        k_flat = k_f.reshape(B * T, H, K)
        v_flat = v_f.reshape(B * T, HV, V)
        g_flat = g_f.reshape(B * T, HV, -1)        # [TT, HV, K] 或 [TT, HV, 1]
        beta_flat = beta_f.reshape(B * T, HV, -1)  # [TT, HV, V] 或 [TT, HV, 1]
        o_flat = torch.zeros(B * T, HV, V, dtype=torch.float32, device=v.device)

        cu = cu_seqlens.tolist()
        bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]
        slots = list(range(len(cu) - 1)) if ssm_state_indices is None \
            else ssm_state_indices.tolist()

        for n, (t0, t1) in enumerate(bounds):
            slot = slots[n]
            S = initial_state[slot].float()  # [HV, V, K]
            accepted = None if num_accepted_tokens is None else int(num_accepted_tokens[n])
            for t in range(t0, t1):
                k_t = k_flat[t, head_map]              # [HV, K]
                q_t = q_flat[t, head_map]              # [HV, K]
                v_new = v_flat[t] - torch.einsum('hvk,hk->hv', S, k_t)
                v_new = v_new * beta_flat[t]           # 广播: 标量/头 或 [HV, V]
                S = S * torch.exp(g_flat[t]).unsqueeze(1) + v_new.unsqueeze(-1) * k_t.unsqueeze(1)
                o_flat[t] = torch.einsum('hvk,hk->hv', S, q_t)
                if accepted is not None and (t - t0 + 1) == accepted:
                    final_state[slot] = S.to(initial_state.dtype)
            if accepted is None:
                final_state[slot] = S.to(initial_state.dtype)

        return o_flat.reshape(v.shape).to(v.dtype), final_state


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "71_FusedRecurrentKdaFwd.json")
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
        q = k = v = g = beta = initial_state = None
        scale = None
        inplace_final_state = True
        cu_seqlens = None
        ssm_state_indices = None
        num_accepted_tokens = None
        use_qk_l2norm_in_kernel = False

        for inp in case["inputs"]:
            name = inp.get("name", "")
            if name in ("q", "k", "v", "initial_state"):
                if name == "q":
                    q = random_tensor(inp["shape"], dtype_map[inp["dtype"]]) * 0.2
                elif name == "k":
                    k = random_tensor(inp["shape"], dtype_map[inp["dtype"]]) * 0.2
                elif name == "v":
                    v = random_tensor(inp["shape"], dtype_map[inp["dtype"]]) * 0.2
                else:
                    initial_state = random_tensor(inp["shape"], dtype_map[inp["dtype"]]) * 0.2
            elif name == "g":
                # g 是逐通道 log 衰减, 必须 <= 0; 为避免接近 0 时状态累乘发散,
                # 限制在 [-8, -0.5] 范围内
                g = -torch.empty(inp["shape"], dtype=dtype_map[inp["dtype"]]).uniform_(0.5, 8.0)
            elif name == "beta":
                # beta 物理含义是 sigmoid 输出的门控系数, 取值 (0, 1)
                beta = torch.empty(inp["shape"], dtype=dtype_map[inp["dtype"]]).uniform_(0.0, 1.0)
            elif name == "scale":
                scale = inp["value"]
            elif name == "inplace_final_state":
                inplace_final_state = inp["value"]
            elif name == "cu_seqlens":
                cu_seqlens = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "ssm_state_indices":
                ssm_state_indices = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "num_accepted_tokens":
                num_accepted_tokens = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "use_qk_l2norm_in_kernel":
                use_qk_l2norm_in_kernel = inp["value"]

        input_groups.append([q, k, v, g, beta, scale, initial_state,
                             inplace_final_state, cu_seqlens, ssm_state_indices,
                             num_accepted_tokens, use_qk_l2norm_in_kernel])
    return input_groups


def get_init_inputs():
    return []