import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes cross entropy loss.
        torch_npu.npu_cross_entropy_loss(input, target, weight=None, reduction="mean", ignore_index=-100, label_smoothing=0.0, lse_square_scale_for_zloss=0.0, return_zloss=False) -> (Tensor, Tensor, Tensor, Tensor)

        PyTorch native implementation of forward function
        def forward(self, input, target, weight=None, reduction="mean", ignore_index=-100, label_smoothing=0.0):
            loss = torch.nn.functional.cross_entropy(input.float(), target, weight=weight,
                                                     ignore_index=ignore_index, reduction=reduction,
                                                     label_smoothing=label_smoothing)
            return loss
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, reduction: str = "mean", ignore_index: int = -100, label_smoothing: float = 0.0):
        """Computes cross entropy loss between input logits [N, C] and target class indices [N].
            Returns the loss (extra outputs of the NPU op are auxiliary values)."""
        return torch_npu.npu_cross_entropy_loss(input, target, weight=weight, reduction=reduction, ignore_index=ignore_index, label_smoothing=label_smoothing)


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
                    if dtype in (torch.int32, torch.int64, torch.int8):
                        tensors[name] = torch.randint(0, 100, shape, dtype=dtype)
                    else:
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    attrs[inp['name']] = inp['value']
            tgt = tensors['target']
            tensors['target'] = torch.randint(0, tensors['input'].shape[1], tgt.shape, dtype=tgt.dtype)
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
