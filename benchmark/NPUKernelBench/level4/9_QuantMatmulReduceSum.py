import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs quantized matmul reduce sum.
    Computes quantized grouped matrix multiplication and sums results across all groups.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                x1_scale=None, x2_scale=None):
        """
        Performs quantized matmul reduce sum.

        Args:
            x1 (Tensor): Left matrix, shape [batch, m, k], dtype int8.
            x2 (Tensor): Right matrix, shape [batch, k, n], dtype int8.
            x1_scale (Tensor): Left matrix quantization scale, shape [batch, m], dtype float32.
            x2_scale (Tensor): Right matrix quantization scale, shape [n], dtype bfloat16.

        Returns:
            Tensor: Sum of all group matmul results, shape [m, n], dtype float32.
        """


        mid = torch.bmm(x1.float(), x2.float()) * x1_scale.unsqueeze(-1)
        M, N = mid.shape[1], mid.shape[2]
        acc = torch.zeros(M, N, device=x1.device, dtype=torch.float32)
        for b in range(mid.shape[0]):
            acc = acc + mid[b]  #fp32 顺序累加
        return acc * x2_scale.float()

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "9_QuantMatmulReduceSum.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        """独立随机选择正态/均匀分布"""
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)
        else:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]

        x1_info = inputs[0]
        x2_info = inputs[1]
        x1_scale_info = inputs[2]
        x2_scale_info = inputs[3]

        # 整数 matrix 保持原有 randint 逻辑
        x1 = torch.randint(-5, 5, x1_info["shape"], dtype=torch.int8)
        x2 = torch.randint(-5, 5, x2_info["shape"], dtype=torch.int8)

        # 浮点 scale 改用独立随机分布
        x1_scale = random_tensor(x1_scale_info["shape"], torch.float32)

        x2_scale_dtype_map = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        x2_scale_dtype = x2_scale_dtype_map.get(x2_scale_info.get("dtype", "bfloat16"), torch.bfloat16)
        x2_scale = random_tensor(x2_scale_info["shape"], x2_scale_dtype)

        input_groups.append([x1, x2, x1_scale, x2_scale])
    return input_groups


def get_init_inputs():
    return []