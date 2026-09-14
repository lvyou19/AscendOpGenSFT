import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes cross entropy loss (aten, log-softmax + NLL fused).
        torch.nn.functional.cross_entropy(input, target, weight, reduction, ignore_index) -> Tensor

        PyTorch native implementation of forward function
        def forward(self, input, target, weight=None, reduction="mean", ignore_index=-100):
            return torch.nn.functional.cross_entropy(input, target, weight=weight,
                                                     reduction=reduction, ignore_index=ignore_index)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, reduction: str = "mean", ignore_index: int = -100):
        """Computes cross entropy between raw logits [N, C] and target class indices [N].
            Supports class weights, ignore_index and mean/sum/none reduction."""
        return torch.nn.functional.cross_entropy(input, target, weight=weight, reduction=reduction, ignore_index=ignore_index)


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
                    if name == 'weight':
                        tensors[name] = torch.rand(shape, dtype=dtype) + 0.5 if shape else None
                    elif dtype in (torch.int32, torch.int64, torch.int8):
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
