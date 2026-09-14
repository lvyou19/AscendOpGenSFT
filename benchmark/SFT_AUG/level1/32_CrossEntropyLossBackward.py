import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes the backward pass of cross entropy loss.
        torch_npu.npu_cross_entropy_loss_backward(grad_loss, log_prob, target, weight=None, grad_zloss=None, lse_for_zloss=None, reduction="mean", ignore_index=-100, label_smoothing=0.0, lse_square_scale_for_zloss=0.0) -> Tensor

        PyTorch native implementation of forward function (reduction="none")
        def forward(self, grad_loss, log_prob, target, weight=None, ignore_index=-100):
            N, C = log_prob.shape
            grad_input = torch.zeros_like(log_prob)
            for i in range(N):
                t = int(target[i])
                if t == ignore_index:
                    continue
                w = 1.0 if weight is None else float(weight[t])
                grad_input[i] = grad_loss[i] * w * torch.exp(log_prob[i])
                grad_input[i, t] -= grad_loss[i] * w
            return grad_input
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_loss: torch.Tensor, log_prob: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, reduction: str = "none", ignore_index: int = -100):
        """Computes the gradient of cross entropy w.r.t. log-probabilities.
            grad_loss has shape [N] when reduction="none". Returns a tensor shaped like log_prob."""
        return torch_npu.npu_cross_entropy_loss_backward(grad_loss, log_prob, target, weight=weight, reduction=reduction, ignore_index=ignore_index)


def get_input_groups():
    """Generate input groups from JSON test cases."""
    json_path = os.path.join(os.path.dirname(__file__), os.path.splitext(os.path.basename(__file__))[0] + '.json')
    input_groups = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            inputs = case['inputs']
            tensors = {}
            attrs = {}
            for inp in inputs:
                if inp['type'] == 'tensor':
                    name = inp['name']
                    dtype_str = inp.get('dtype', 'float32')
                    shape = inp.get('shape')
                    if shape is None:
                        tensors[name] = None
                        continue
                    dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16,
                             'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}[dtype_str]
                    if name == 'log_prob':
                        tensors[name] = torch.nn.functional.log_softmax(torch.randn(shape, dtype=torch.float32), dim=-1)
                    elif dtype in (torch.int32, torch.int64, torch.int8):
                        tensors[name] = torch.randint(0, 100, shape, dtype=dtype)
                    else:
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    attrs[inp['name']] = inp['value']
            tgt = tensors['target']
            tensors['target'] = torch.randint(0, tensors['log_prob'].shape[1], tgt.shape, dtype=tgt.dtype)
            if tensors['weight'] is not None:
                tensors['weight'] = torch.rand(tensors['weight'].shape, dtype=torch.float32) + 0.5
            group = []
            for inp in inputs:
                if inp['type'] == 'tensor':
                    group.append(tensors[inp['name']])
                else:
                    group.append(attrs[inp['name']])
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
