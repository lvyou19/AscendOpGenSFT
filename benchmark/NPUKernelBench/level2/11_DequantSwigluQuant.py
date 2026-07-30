import torch
import torch.nn as nn
import torch_npu  # noqa: F401  (kept for runtime device registration parity)


class Model(nn.Module):
    """
    Reference for fused dequantize + SwiGLU + per-row dynamic int8 quant.

    Documentation-grounded fp32 CPU implementation. Replaces the previous direct
    call to ``torch_npu.npu_dequant_swiglu_quant``; see PR description for why
    the NPU kernel is no longer used as the oracle.

    SwiGLU mode 0 (traditional, halves split + Swish):
        front, back = x[:, :H], x[:, H:]
        glu, lin    = (front, back) if activate_left else (back, front)
        out         = lin * Swish(glu)               # Swish(z) = z * sigmoid(z)

    SwiGLU mode 1 (gpt-oss variant, per CANN 9.0 V2 doc + empirical reconciliation):
        glu = x[:, 0::2]                             # ALWAYS even
        lin = x[:, 1::2]                             # ALWAYS odd  (activate_left ignored)
        glu_c = min(glu, L)                          # one-side clamp
        lin_c = clamp(lin, -L, L)                    # two-side clamp
        out   = glu_c * sigmoid(alpha * glu_c) * (lin_c + beta)

    Followed by smooth quant (out * quant_scale[0]) and per-row dynamic int8.
    Returns (int8 [M, K], fp32 [M]) on the input's original device.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, weight_scale: torch.Tensor = None, activation_scale: torch.Tensor = None,
                bias: torch.Tensor = None, quant_scale: torch.Tensor = None, quant_offset: torch.Tensor = None,
                group_index: torch.Tensor = None, activate_left: bool = False, quant_mode: int = 0,
                swiglu_mode: int = 0, clamp_limit: float = 7.0, glu_alpha: float = 1.702,
                glu_bias: float = 1.0) -> tuple:
        """Doc-grounded fp32 CPU reference. Returns (int8 [M,K], fp32 [M]) on x.device."""
        target_device = x.device

        # Move + cast all tensor inputs to CPU fp32 for deterministic fp32 math.
        x_f32  = x.detach().cpu().to(torch.float32)
        ws_f32 = weight_scale.detach().cpu().to(torch.float32)
        as_f32 = activation_scale.detach().cpu().to(torch.float32)
        qs_f32 = quant_scale.detach().cpu().to(torch.float32)

        sm = int(swiglu_mode)
        al = bool(activate_left)
        L = float(clamp_limit)
        alpha = float(glu_alpha)
        beta = float(glu_bias)
        M, N = x_f32.shape
        H = N // 2

        # Dequant: int32 -> fp32 * weight_scale[0] * activation_scale (per-row)
        xf32 = x_f32 * ws_f32[0] * as_f32

        if sm == 0:
            # Mode 0: halves split + Swish; activate_left toggles which half is glu.
            front = xf32[:, :H]
            back  = xf32[:, H:]
            if al:
                glu, lin = front, back
            else:
                glu, lin = back, front
            sw = lin * (glu * torch.sigmoid(glu))
        else:
            # Mode 1: even/odd split + gpt-oss formula. activate_left is IGNORED here:
            # CANN 8.5's V2 dispatch path does not honor it (verified empirically across
            # all INPUT_CASES; see PR description).
            glu = xf32[:, 0::2]
            lin = xf32[:, 1::2]
            glu_c = glu.clamp(max=L)
            lin_c = lin.clamp(-L, L)
            sw = glu_c * torch.sigmoid(alpha * glu_c) * (lin_c + beta)

        # Smooth quant (broadcast row-wise over K)
        s = sw * qs_f32[0]

        # Per-row dynamic int8 quant: scale = absmax / 127
        absmax = s.abs().amax(dim=-1)
        scale = absmax / 127.0
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        i8 = (s / safe_scale.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
        # Rows whose absmax==0 must yield zero quantized output.
        i8 = torch.where(scale.unsqueeze(-1) > 0, i8, torch.zeros_like(i8))

        # Return on the input's original device so callers see the same device contract.
        return i8.to(target_device), scale.to(target_device)
