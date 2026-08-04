# 算子描述文件和 Case 生成指导

生成算子的输入是一对同名文件：

```
{id}_{OpName}.py    ← PyTorch 参考实现 + 解释注释
{id}_{OpName}.json  ← 每行一组测试输入（JSONL）
```

下游（单算子生成或批量评测）加载这两个文件，跑参考实现得到标准输出，生成器再产出算子 kernel 与之对齐。**.py 文件必须能正常 import 不报错、且 `get_input_groups()` 返回的 case 数与 JSON 行数一致**——任一不满足，下游直接报错。

---

# 第一部分 · 算子描述文件（.py）

## 1.1 写法决策树（按这个顺序走）

```
              算子描述文件怎么写
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  ① 先查 CANN 仓            （必做第一步）
  /home/t00893162/AscendOpGenAgent/benchmarks/CANN/level*/
        │
        ├─ 命中 → 直接用，跳到 1.2
        │
        └─ 没命中 ↓
                ② 查 torch 官方 API
                https://pytorch.org/docs/stable/
                    │
                    ├─ 命中 → 基于 torch API 写，跳到 1.3
                    │
                    └─ 没命中（组合算子 / 私有语义）
                                → 拼小算子实现，跳到 1.4
```

**关键**：①和②不是并列选择，是**先后顺序**。先查 CANN，没有再去 torch。这样能保证风格统一、避免重复造轮子。

---

## 1.2 情况 ①：CANN 仓已有 —— 直接用

**识别**：`/home/t00893162/AscendOpGenAgent/benchmarks/CANN/level{1,2,3,4}/` 下已经有 `{id}_{OpName}.py` + `{OpName}.json`（**注意 CANN 仓的 json 没有 id 前缀**）。

CANN 仓的 `.py` 长这样（带 NPU stub、`get_inputs(param, device)`、扁平 JSON 读取）：

```python
# 顶部 stub 一堆 NPU 依赖
for _n in ("torch_npu", "torch_npu.contrib", "framework", ...):
    ...

class Model(nn.Module):
    def __init__(self, ...):
        ...
    def forward(self, x):
        return torch.cos(x)

def get_inputs(param, device=None):
    shape = eval(param.get('input_shape', '[1]'))
    ...
```

**直接用的工作流**：

1. 复制 `{id}_{OpName}.py` 和 `{OpName}.json` 到你的 benchmark 目录。
2. 重命名 `.json` 为带 id 前缀（`{id}_{OpName}.json`），让两个文件同名。
3. **同步修改 `.py` 里读 JSON 的文件名**（搜索 `_load_cases()` 或 `os.path.join(...)` 那行）。
4. 跑一下自检脚本（见 1.5），确认能 import、`get_input_groups()` 能返回非空 list。

**什么时候不要直接用**：

- case 数量太少（<5 个）或 dtype 只有 float16 → 仍用 CANN 的 `.py` 框架，但**追加 case 到 JSON**（参见第二部分）。
- `Model.forward` 的实现是 NPU 私有接口（如 `npu_xxx`）→ 这种不能直接用，跳到 1.3 用 torch API 重写。

> **不要重写已经能跑的 CANN 文件**。CANN 旧格式 JSON 字段是算子特定的（`n`、`incx`、`input_shape`），下游对齐脚本有适配。重写容易破坏字段名约定。

---

## 1.3 情况 ②：基于 torch 官方 API 新写

**核心思路**：把 torch 官网对这个 API 的描述、参数说明、约束**写进 `.py` 的注释里**，让生成算子 kernel 的模型能"读懂"接口语义。

### 1.3.1 标准 PY 骨架

```python
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """<一句话算子功能>。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, <forward_args>) -> torch.Tensor:
        return <调用 torch 接口>


def get_input_groups():
    """逐行读 JSON，返回 list[list]，每个子 list 是一组 forward 实参。"""
    json_path = os.path.join(os.path.dirname(__file__), "{id}_{OpName}.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    # dtype_map 包含所有 NPUKernelBench 使用的 dtype 字符串及别名
    dtype_map = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float64": torch.float64, "fp64": torch.float64,
        "int64": torch.int64, "int32": torch.int32,
        "int16": torch.int16, "int8": torch.int8,
        "uint8": torch.uint8, "uint16": torch.uint16,
        "uint32": torch.uint32, "uint64": torch.uint64,
        "bool": torch.bool,
        "complex64": torch.complex64, "complex128": torch.complex128,
    }

    def make_tensor(dtype_name, shape, rng=None):
        dtype = dtype_map[dtype_name]
        shape = tuple(shape)
        if dtype_name in ("int64", "int32", "int16", "int8"):
            low, high = rng if rng is not None else [0, 9]
            return torch.randint(low, high + 1, shape, dtype=dtype)
        elif dtype_name in ("uint8",):
            low, high = rng if rng is not None else [0, 10]
            return torch.randint(low, high, shape, dtype=torch.int64).to(dtype)
        elif dtype_name in ("uint16", "uint32", "uint64"):
            # CPU torch 不支持这些 dtype，用 int64 生成后转换
            low, high = rng if rng is not None else [0, 10]
            return torch.randint(low, high, shape, dtype=torch.int64).to(dtype)
        elif dtype_name == "bool":
            return torch.randint(0, 2, shape, dtype=dtype)
        else:
            return torch.randn(shape, dtype=dtype)

    input_groups = []
    for case in cases:
        args = []
        for inp in case["inputs"]:
            if inp["type"] == "tensor":
                args.append(make_tensor(inp["dtype"], inp["shape"], inp.get("range")))
            elif inp["type"] == "tensor_list":
                # tensor_list 用 "shapes" 字段（同 dtype），或 "tensors" 数组（异构）
                if "shapes" in inp:
                    args.append([make_tensor(inp.get("dtype", "float16"), s,
                                             inp.get("range"))
                                 for s in inp["shapes"]])
                else:
                    args.append([make_tensor(t["dtype"], t["shape"], t.get("range"))
                                 for t in inp.get("tensors", [])])
            else:  # attr
                args.append(inp["value"])
        input_groups.append(args)
    return input_groups


def get_init_inputs():
    """返回 []（空列表）表示 Model 无初始化参数。
    
    NPUKernelBench 标准：所有参数（包括 eps、alpha、scale 等）都作为
    forward 实参传入，__init__ 不接收参数。\"\"\"
    return []
```

### 1.3.2 在哪里加注释、加什么注释（重点）

**注释要放在 4 个位置**，目的是把 torch 文档"翻译"成生成器最易消化的形式：

#### 位置 A · `Model` 类的 docstring —— 算子整体语义

抄 torch 官网第一段描述，**用中文复述一遍**，并标注官网链接。

```python
class Model(nn.Module):
    """对输入张量沿指定维度做层归一化。

    官方文档：https://pytorch.org/docs/stable/generated/torch.nn.functional.layer_norm.html

    数学公式：
        y = (x - E[x]) / sqrt(Var[x] + eps) * γ + β
    其中 E[x]、Var[x] 沿 normalized_shape 维度计算，γ/β 是可学习仿射参数。

    NPU 实现要点：
    - 最后一维（normalized_shape 总长度）必须 ≤ 4096（硬件 cube 单元限制）；
    - eps 建议取 1e-5，过小会数值溢出；
    - 不带 weight/bias 的版本等价于纯 RMS 归一化。
    """
```

#### 位置 B · `forward` 参数列表 —— 每个参数的含义、约束

```python
def forward(self,
            x: torch.Tensor,                 # 输入张量，shape [..., *normalized_shape]
            normalized_shape: list,          # 归一化维度，list[int]，必须等于 x 末尾几维
            weight: torch.Tensor = None,     # 仿射 γ，shape == normalized_shape，可省
            bias: torch.Tensor = None) -> torch.Tensor:  # 仿射 β，shape == normalized_shape，可省
    ...
```

#### 位置 C · `__init__` —— NPUKernelBench 标准：空 `__init__`

**NPUKernelBench 标准**：`__init__` 不接受参数，所有可配置值（eps、alpha、scale 等）都作为 `forward` 的实参传入。这是为了让下游 harness 能逐 case 传入不同值。

```python
def __init__(self):
    """
    NPUKernelBench 标准：无初始化参数。
    所有配置值（eps、alpha、scale 等）通过 forward 实参传入，
    这样每个 case 可以独立控制参数取值。
    """
    super(Model, self).__init__()
```

> **仅在极少数自定义场景下**，如果有真正模型级共享的可学习参数（如权重矩阵），才在 `__init__` 中初始化。绝大多数 NPUKernelBench 算子都是空 `__init__`。

#### 位置 D · 复杂逻辑前 —— 用注释把公式拆成可读步骤

```python
def forward(self, x, gamma, beta):
    # 1. 沿最后一维求均值 E[x]
    mean = x.mean(dim=-1, keepdim=True)
    # 2. 求方差 Var[x]（无偏估计用 unbiased=True，layer_norm 默认 False）
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    # 3. 标准化：(x - mean) / sqrt(var + eps)
    x_norm = (x - mean) / torch.sqrt(var + self.eps)
    # 4. 仿射：x_norm * gamma + beta
    return x_norm * gamma + beta
```

### 1.3.3 注释验证 —— 用小算子拼装比对 torch API 结果

**关键实践**：如果 `forward` 直接调 torch API，注释里的解释是否正确？建议**用小算子（基础 torch 算子）拼一份"等价实现"放进注释，并验证它和直接调 torch API 输出一致**。

下面这段代码可以单独跑，验证注释里的拆解等价于 `F.layer_norm`：

```python
# === 注释验证脚本（不放进交付的 .py，仅本地跑一次确认注释正确）===
import torch
import torch.nn.functional as F

torch.manual_seed(0)
x = torch.randn(2, 3, 8)
gamma = torch.randn(8)
beta = torch.randn(8)
eps = 1e-5

# (A) 直接调 torch API
ref = F.layer_norm(x, [8], weight=gamma, bias=beta, eps=eps)

# (B) 注释里拆解的小算子实现
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, unbiased=False, keepdim=True)
my = (x - mean) / torch.sqrt(var + eps) * gamma + beta

assert torch.allclose(ref, my, atol=1e-6), "注释里的拆解和 torch API 不一致！"
print("注释验证通过")
```

**通过验证后**，把等价实现留在注释里，`forward` 主体可以**保留 torch API 调用**（更稳），也可以**替换成小算子拼装**（让生成器更容易理解每一步）。两种都可以，看哪个对模型更友好。

---

## 1.4 情况 ③：组合算子 —— 用小算子拼装

**识别**：torch 没有直接对应的 API，但算子语义能用几个基础 op 拼出来（例如"加然后取 topk 再 softmax"这类自定义融合算子）。

**写法**：

1. `forward` 主体直接写小算子拼装的代码。
2. 注释里**逐步标注每一步在做什么**，对应数学公式哪一项。
3. **必须用 1.3.3 的方法验证**：和一份独立的 numpy 或手算参考比对，确保拼装正确。

```python
class Model(nn.Module):
    """自定义融合算子：x 经过 linear，做 GELU 激活，再 rescale。

    数学公式：
        y = gelu(x @ W^T + b) * scale
    其中 gelu 用 tanh 近似：
        gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    官方 GELU 文档：https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html
    """

    def __init__(self, in_features: int, out_features: int, scale: float):
        super().__init__()
        self.weight = torch.randn(out_features, in_features)
        self.bias = torch.zeros(out_features)
        self.scale = scale

    def forward(self, x):
        # 1. 线性变换：x @ W^T + b
        h = x @ self.weight.t() + self.bias
        # 2. GELU 激活（tanh 近似）
        c = torch.sqrt(torch.tensor(2.0 / torch.pi))
        gelu = 0.5 * h * (1 + torch.tanh(c * (h + 0.044715 * h ** 3)))
        # 3. 缩放
        return gelu * self.scale
```

---

## 1.5 自检脚本

每个 `.py` 写完都跑一遍，确保和下游 harness 兼容：

```bash
#!/usr/bin/env bash
# 用法: bash check_op.sh 10_LayerNorm.py
set -e
PY=$1
DIR=$(dirname "$PY")
python -c "
import importlib.util as u, os
spec = u.spec_from_file_location('m', '$PY')
m = u.module_from_spec(spec); spec.loader.exec_module(m)

groups = m.get_input_groups()
inits  = m.get_init_inputs()
# 空 list 表示无 init 参数，补为等长空组
if not inits:
    inits = [[] for _ in groups]
assert len(groups) == len(inits), f'len mismatch: groups={len(groups)} inits={len(inits)}'

json_path = os.path.join('$DIR', os.path.basename('$PY').replace('.py', '.json'))
with open(json_path) as f:
    n_lines = sum(1 for line in f if line.strip())
assert len(groups) == n_lines, f'JSON has {n_lines} lines but groups has {len(groups)}'

init_args = inits[0]
model = m.Model(*init_args) if init_args else m.Model()
out = model(*groups[0])
print(f'OK: {len(groups)} cases, output shape={tuple(out.shape) if hasattr(out, \"shape\") else type(out)}')
"
```

检查项：

- [ ] `.py` 和 `.json` 同名（含 id 前缀）、在同一目录。
- [ ] `len(get_input_groups()) == JSON 行数`，`get_init_inputs()` 返回 `[]`（标准模式）或等长 list。
- [ ] `Model(*get_init_inputs()[0])(*get_input_groups()[0])` 能跑通、返回 tensor。
- [ ] 顶部**没有** `import torch_npu` / `import framework`（NPU 框架包，纯 CPU 跑不了）。
- [ ] 注释里的"小算子拼装实现"和 torch API 输出一致（用 1.3.3 的脚本验证过）。

---

---

## 1.6 特殊模式

### 1.6.1 多输出（Tuple Return）

部分算子返回多个 tensor（如 GroupNormSwish 返回 `output, mean, rstd`）：

```python
def forward(self, input: torch.Tensor, ...) -> tuple:
    """
    Returns:
        tuple: (output tensor, mean, rstd)
    """
    # ...计算...
    return output, mean_out, rstd_out
```

`.py` 无需特殊处理，`Model.forward` 直接返回 tuple 即可。下游 harness 会自动解包。

### 1.6.2 Backward 算子模式

backward 算子（如 EluBackward、MishBackward）有固定模式：

- 输入含 `grad_output`（上游梯度）+ `self_or_result`（前向输入或结果）
- 多个 float 型 attr 参数（alpha、scale、input_scale 等）
- bool 型 is_result 参数区分"输入是前向结果"还是"输入是前向输入"
- 通常在 forward 中将 fp16 升为 fp32 计算再转回

```python
def forward(self, grad_output, alpha, scale, input_scale, is_result, self_or_result):
    orig_dtype = grad_output.dtype
    if orig_dtype == torch.float16:
        grad_output = grad_output.float()
        self_or_result = self_or_result.float()

    mask = self_or_result <= 0
    if is_result:
        factor = torch.where(mask, input_scale * (self_or_result + alpha * scale), scale)
    else:
        tmp = torch.exp(self_or_result * input_scale)
        factor = torch.where(mask, input_scale * alpha * scale * tmp, scale)
    result = grad_output * factor

    if orig_dtype == torch.float16:
        result = result.to(orig_dtype)
    return result
```

### 1.6.3 tensor_list 类型

`type: "tensor_list"` 的 JSON 字段有两种格式：

**格式 1 — 同 dtype 多 shape**（推荐，如 Cat）：
```json
{"name": "tensors", "type": "tensor_list", "required": true, "dtype": "float16", "shapes": [[128], [128], [128]]}
```

**格式 2 — 异构 tensor 数组**（如多个不同 dtype/shape 的 tensor）：
```json
{"name": "tensors", "type": "tensor_list", "required": true, "tensors": [
  {"dtype": "float16", "shape": [128]},
  {"dtype": "int64", "shape": [64]}
]}
```

`.py` 中读取时先检查 `"shapes"` 字段，不存在则回退到 `"tensors"` 数组。

### 1.6.4 多 dtype 同 case（不同 Tensor 不同 dtype）

如 GtTensor：`x` 是 float16，`y` 是 int32。JSON 中每个 tensor 独立声明 dtype：

```json
{"inputs": [
  {"name": "x", "type": "tensor", "dtype": "float16", "shape": [128]},
  {"name": "y", "type": "tensor", "dtype": "int32", "shape": [128]}
]}
```

`.py` 中对每个 tensor 独立调用 `make_tensor`，不共享 dtype。

### 1.6.5 uint dtype 兼容

CPU torch 不支持 `uint16`/`uint32`/`uint64`。如需覆盖这些 dtype，forward 中显式 cast 为 float32：

```python
def forward(self, x, y):
    if x.dtype in (torch.uint16, torch.uint32, torch.uint64):
        x = x.to(torch.float32)
    if y.dtype in (torch.uint16, torch.uint32, torch.uint64):
        y = y.to(torch.float32)
    return torch.gt(x, y)
```

### 1.6.6 `_all_case.json` 备份

部分 level0 算子有 `{id}_{OpName}_all_case.json` 文件（如 `10_Relu_all_case.json`）。
这是 case-simplifier 精简前原始全量 case 的备份，供 Phase 6 全量验证恢复使用。
**新生成文件不需要创建此备份**。

---

## 1.7 算子描述文件常见坑（更新）

| 症状 | 原因 | 修复 |
|------|------|------|
| `len(groups) != len(inits)` 且 inits 非空 | 改了 JSON 没同步改 `get_init_inputs` | `get_init_inputs` 改成返回 `[]`（标准模式）或动态读 JSON 行数 |
| `Model()` 报 `missing required argument` | `get_init_inputs` 返回 `[]` 但 `__init__` 有必填参数 | `__init__` 改为空，参数移到 forward |
| `forward()` 报参数过多/过少 | JSON 里 `inputs` 数组字段数和 `forward` 签名不一致 | 对齐顺序和数量 |
| 整数 dtype 在 CPU 报错 `randn not supported for int` | 没走 `make_tensor` 的 int 分支 | 用骨架里的 `make_tensor` |
| uint16/32/64 在 CPU 报错 | CPU torch 不支持这些 dtype | forward 中 cast 为 float32，或 JSON 中避免使用 |
| tensor_list 读取失败 | JSON 用了 `shapes` 但 `.py` 读的是 `tensors` | `.py` 需同时处理两种格式 |
| 生成器输出的 kernel 语义偏了 | 注释不够详细，模型没理解 | 补 torch 官网描述 + 小算子拆解注释 |
| 注释里的拆解实现和 torch API 输出不一致 | 注释写错了 | 跑 1.3.3 验证脚本修正 |

---

# 第二部分 · Case 文件（.json）

> 核心思路：**case 不是越多越好，要覆盖语义边界 + 性能区间**。

## 2.1 文件格式：JSON Lines

**整体不是合法 JSON 数组**，而是**每行一个独立 JSON 对象**。

### 格式 A：inputs 数组格式（NPUKernelBench，推荐）

```json
{"inputs": [{"name": "x", "type": "tensor", "required": true, "dtype": "float16", "shape": [1, 128, 4096]}, {"name": "normalized_shape", "type": "attr", "required": true, "dtype": "list", "value": [4096]}, {"name": "weight", "type": "tensor", "required": false, "dtype": "float16", "shape": [4096]}]}
```

### 格式 B：扁平 kv 格式（CANN 旧格式）

```json
{"case_id": "0", "input_shape": "[48, 128, 256]", "dtype": "float16"}
```

> **新写文件统一用格式 A（inputs 数组）**。格式 B 仅出现在从 CANN 仓直接复用的场景，保留原样不转换。

### 格式 A 单个 input 字段

| 字段 | 必填 | 含义 |
|------|:---:|------|
| `name` | ✅ | 参数名，需与 `Model.forward` 参数名对得上 |
| `type` | ✅ | `tensor` / `tensor_list` / `attr` |
| `required` | ✅ | 是否必传（`false` 时可以省略或给 `null`） |
| `dtype` | tensor 必填 | `float32` / `float16` / `bfloat16` / `int8` / `int16` / `int32` / `int64` / `uint8` / `bool` / `complex64` / `complex128`；attr 则填 `int` / `float` / `list` / `str` |
| `shape` | tensor 必填 | 列表，如 `[1, 128, 4096]`；标量空 tensor 用 `[]` |
| `value` | attr 必填 | attr 的实际值，如 `[4096]` / `0.01` / `true` |
| `range` | 可选 | 整数 tensor 取值区间 `[low, high]`，默认 `[0, 9]` |
| `tensors` | tensor_list 可选 | 异构子 tensor 列表（格式 2），每项含 `dtype`/`shape`/`range` |
| `shapes` | tensor_list 可选 | 同 dtype 多 shape 列表（格式 1，推荐），与 `tensors` 二选一 |

---

## 2.2 Case 数量与覆盖矩阵（按算子类型）

不同算子类型的测试重点不同，按如下矩阵确定 case：

### 算子分类与覆盖要求

| 算子类型 | 典型算子 | dtype | shape 维度 | attr | 特殊 |
|----------|---------|-------|-----------|------|------|
| **elementwise** | abs, cos, gelu, relu, sigmoid, tanh, add, mul, sub, div, neg, exp, log, silu, sqrt | float16(主力), float32, bfloat16 | 1D/2D/3D/4D，含非对齐(5/7/13/17/31) + 标量[] + 超大(>2^18) | 枚举全量 | inf/nan (isinf/isnan类) |
| **reduction** | sum, mean, max, min, softmax, logsoftmax, argmax, argmin | float16(主力), float32, bfloat16 | 1D-4D，含非对齐 + 大最后一维 + 标量[] | keepdim T/F, dim 正/负/0/-1 | 空维度 |
| **normalization** | layernorm, groupnorm, batchnorm, rmsnorm | float16(主力), float32, bfloat16 | 2D-4D | eps(1e-5, 1e-3, 0.1), 带/不带 weight/bias | 末维≥4096，末维非对齐 |
| **index/gather** | index_select, gather, index_put, scatter | float16/32 + int32/64 索引 | 1D-3D | dim 正/负/0/-1 | range 边界(0, shape[dim]-1)，重复索引 |
| **comparison/logical** | eq, ne, gt, lt, ge, le, logical_and/or/not | float16/32, int32/64, bool | 1D-4D + 广播shape | — | 广播场景 (不同rank) |
| **matmul** | matmul, bmm, linear | float16/32, bfloat16 | 2D + 3D(batch) + 大矩阵(>4096) | transpose T/F | K维非对齐 |
| **creation** | eye, fill, zeros, ones, arange | float16/32, bfloat16, int, bool | 小/中/大 | 创建参数 | 边界(n=0, m=0) |
| **manipulation** | permute, cat, split, pad, repeat, reshape | float16/32 | 按需 | 各模式全量 | dim边界 |
| **composite** | swiglu, adamw, 自定义融合 | 按子算子取并集 | 取最宽泛 | 按需 | 按需 |

### 推荐 case 数

| 算子复杂度 | case 数 |
|-----------|--------|
| 简单 elementwise（abs、cos、isinf） | 15–30 |
| 带 reduction（layernorm、softmax） | 20–40 |
| 矩阵乘 / conv | 15–25 |
| 多输入 + 复杂 attr | 20–35 |

> 下游清洗默认全 case 跑评测，case 过多拖慢批跑；过少（<10）训练样本不够。

---

## 2.3 维度覆盖原则

每个算子至少覆盖这几种 shape 维度：

| 类别 | 示例 | 为什么必测 |
|------|------|----------|
| 标量 / 空 tensor | `[]`、`[1]` | 测标量分支 |
| 小 1D（非对齐） | `[5]`、`[7]`、`[13]` | 测 tail 元素处理 |
| 中等 1D | `[128]`、`[1024]`、`[4096]` | 主力区间 |
| 2D | `[16, 2048]`、`[128, 768]` | 矩阵 / batch |
| 3D | `[1, 128, 4096]`、`[4, 64, 64]` | 通道 / 序列 |
| 4D | `[1, 3, 224, 224]`、`[16, 64, 56, 56]` | 图像 / conv |
| 超大维度 | `[1, 4096, 4096]`、`[16, 1024, 4096]` | 测大 shape 性能 |

**强烈推荐的快捷铺法**（参考 `10_LayerNorm.json`）：固定几个典型 LLM hidden_size（`2048 / 3584 / 4096 / 5120 / 6144 / 8192`），× 几个 seq_len（`128 / 512 / 2048`），× 几个 batch（`1 / 4 / 8 / 16`），快速构成几十个有意义的组合。

---

## 2.4 dtype 覆盖

按算子支持情况至少覆盖：

- **浮点**：`float16`（**必含**，NPU 主力 dtype）、`float32`、`bfloat16`（按算子支持，如 LayerNorm/Softmax 支持，但部分老算子不支持）。
- **整数**：算子语义允许时补 `int32` / `int64` / `uint8` / `int8`（如 `one_hot`、`index_select`、比较类）。
- **bool**：逻辑算子（`gt`、`ne`）输出是 bool；输入是 bool 的算子（`logical_and`）要测。
- **complex**：仅在算子明确支持时才加（NPU kernel 大多不支持 complex，慎用）。

**不要**把不支持的 dtype 硬塞进去（如 `cos` 对 int 无意义），下游会全 fail。

### dtype 最小覆盖要求

| 算子类 | 最少 dtype | 说明 |
|--------|----------|------|
| 浮点计算类 | float16(≥50%), float32(≥2case), bfloat16(≥2case) | bf16 不支持可跳过 |
| 整数语义类 | int32, int64 | 如 bitwise、index |
| 比较/逻辑类 | float16, int32, bool | 各至少 2 case |
| 创建类 | float16, int32 | — |

---

## 2.5 Attr 取值覆盖

- **数值类 attr**（`alpha`、`min_val`、`negative_slope`）：至少 3 档（`0`、小正数、大正数；语义允许时补负数）。
- **list 类 attr**（`normalized_shape`、`kernel_size`、`padding`）：覆盖单元素 list 和多元素 list。
- **bool / 枚举类 attr**（`keepdim`、`descending`）：两个取值都测。
- **str 枚举类 attr**（`approximate`='none'/'tanh'）：每个合法值都测。

### attr 最小覆盖表

| attr 类型 | 最少取值数 | 必须包含 |
|-----------|----------|---------|
| bool | 2 | True + False |
| int/float | 3 | 0, 正数, 负数(语义允许) |
| enum(str) | 全部 | 每个合法值 |
| list | 2 | 单元素 + 多元素 |
| dim(int) | 4 | 0, 正数, -1, 负数 |

---

## 2.6 边界 case（按算子类型选加）

| 边界类型 | 适用算子 | 示例 |
|----------|---------|------|
| keepdim=True vs False | reduction | dim=0, dim=-1 各一份 |
| 空 dim / 标量输入 | 通用 | shape=[]，shape=[1] |
| 负 dim | index, reduction, cat | dim=-1, dim=-2 |
| 输入含 inf/nan | isinf, isnan, 除法类 | value 中含 inf/-inf/nan |
| required=false 省略/给 null | 带可选参数的算子 | 如 LayerNorm 不带 bias |
| 广播 (broadcasting) | elementwise | [128,1] + [128], [1,256] + [128,256] |
| 重复索引 | index_select, gather | range 内同一值多次 |
| 极端大 shape | 通用 | numel > 2^20 |
| 极端小 shape | 通用 | shape=[1], shape=[2] |
| 单元素 shape | 通用 | shape=[1,1], shape=[1,1,1] |

---

## 2.7 一个完整的 case 设计示例（LayerNorm）

按上面原则铺出来，LayerNorm 的 case 大概长这样（节选）：

```json
{"inputs": [{"name": "x", "type": "tensor", "required": true, "dtype": "float32", "shape": [1, 128, 4096]}, {"name": "normalized_shape", "type": "attr", "required": true, "dtype": "list", "value": [4096]}, {"name": "weight", "type": "tensor", "required": false, "dtype": "float32", "shape": [4096]}, {"name": "bias", "type": "tensor", "required": false, "dtype": "float32", "shape": [4096]}]}
{"inputs": [{"name": "x", "type": "tensor", "required": true, "dtype": "float16", "shape": [1, 256, 4096]}, {"name": "normalized_shape", "type": "attr", "required": true, "dtype": "list", "value": [4096]}, {"name": "weight", "type": "tensor", "required": false, "dtype": "float16", "shape": [4096]}, {"name": "bias", "type": "tensor", "required": false, "dtype": "float16", "shape": [4096]}]}
{"inputs": [{"name": "x", "type": "tensor", "required": true, "dtype": "bfloat16", "shape": [1, 1024, 4096]}, ...]}
{"inputs": [{"name": "x", "type": "tensor", "required": true, "dtype": "float16", "shape": [1, 2048, 4096]}, ...]}
{"inputs": [{"name": "x", "type": "tensor", "required": true, "dtype": "float16", "shape": [4, 128, 4096]}, ...]}
{"inputs": [{"name": "x", "type": "tensor", "required": true, "dtype": "float16", "shape": [16, 1024, 4096]}, ...]}
{"inputs": [{"name": "x", "type": "tensor", "required": true, "dtype": "float16", "shape": [1, 128, 4096]}, {"name": "normalized_shape", "type": "attr", "required": true, "dtype": "list", "value": [4096]}, {"name": "weight", "type": "tensor", "required": false, "dtype": "float16", "shape": [4096]}]}
```

最后一行故意不带 `bias`，对应 `required: false` 的 case。

---

## 2.8 Case 文件常见坑

| 症状 | 原因 | 修复 |
|------|------|------|
| 下游报 `KeyError: 'inputs'` | 用了 CANN 扁平 kv 格式但 `.py` 是 inputs 数组骨架 | 两种格式要一致，新写统一用 inputs 数组 |
| 整行 case 都 fail | dtype 不被算子支持 | 删掉该 dtype 的 case |
| 多数 case pass 但个别 fail | 某 case shape 落在算子不支持的区间（如 LayerNorm 末维 > 8192） | 删该 case 或拆分 |
| JSON 末尾多了空行 | 编辑器自动加 | 不影响（脚本会过滤），但洁癖建议删 |
| `forward()` 报参数过多/过少 | JSON inputs 数组字段数和 `Model.forward` 参数不一致 | 对齐两者 |

---

# 第三部分 · 完整性评估标准

> 此部分供 `op-case-generator` skill 的评估模式使用，也可人工对照检查。

## 3.1 评估维度与权重

| 维度 | 权重 | 说明 |
|------|:---:|------|
| 结构完整性 | 一票否决 | import/Mode/函数都存在、命名一致、行数对齐 |
| 可执行性 | 一票否决 | import→Model()→forward() 全链路通 |
| dtype 覆盖 | 25% | 覆盖算子类型要求的 dtype |
| shape 覆盖 | 25% | 覆盖要求的维度数 + 非对齐 + 超大 shape |
| attr 覆盖 | 15% | 覆盖所有合法 attr 取值 |
| 边界 case | 20% | 覆盖算子类型要求的关键边界 |
| 注释质量 | 15% | docstring、参数说明、公式拆解 |

## 3.2 评估等级

| 等级 | 条件 |
|------|------|
| **PASS** | 一票否决项通过 + 总分 ≥ 80 |
| **WARN** | 一票否决项通过 + 总分 60–79 |
| **FAIL** | 任一否决项不通过，或总分 < 60 |

## 3.3 逐维度检查清单

### 结构完整性（一票否决）

- [ ] 文件存在且同名（`.py` 和 `.json` 在同目录，去掉路径后文件名完全一致）
- [ ] `.py` 可 import 通过（无 SyntaxError/ImportError）
- [ ] `.py` 包含 `class Model(nn.Module)`
- [ ] `.py` 包含 `get_input_groups()` 函数
- [ ] `.py` 包含 `get_init_inputs()` 函数
- [ ] `len(get_input_groups()) == JSON 行数`，`get_init_inputs()` 返回 `[]`（标准模式）或等长 list

### 可执行性（一票否决）

- [ ] `Model(*get_init_inputs()[0])(*get_input_groups()[0])` 不报错
- [ ] 返回值为 `torch.Tensor`（或 tuple of Tensor）
- [ ] 输出不含 NaN/Inf（除非算子语义本身就是测 NaN/Inf 的，如 isinf/isnan）

### dtype 覆盖

- [ ] `float16` 在 case 中占比 ≥ 30%（NPU 主力 dtype）
- [ ] `float32` 至少 2 个 case
- [ ] 如算子支持 `bfloat16`，至少 2 个 case
- [ ] 整数/布尔类算子有对应 dtype

### shape 覆盖

- [ ] 至少 3 种不同维度数（1D/2D/3D/4D 中取 3）
- [ ] 至少 1 个非对齐 shape（如 `[13]`、`[17, 31]`）
- [ ] 至少 1 个超大 shape（numel > 2^18）
- [ ] 至少 1 个标量或极小 shape（`[]` 或 `[1]` 或 `[2]`）

### attr 覆盖

- [ ] bool attr：True 和 False 都出现
- [ ] 数值 attr：至少 3 个不同值
- [ ] enum/str attr：所有合法值都出现
- [ ] list attr：不同长度
- [ ] dim attr：正数、负数、0 都出现

### 边界 case

- [ ] `required: false` 的参数在至少 1 个 case 中被省略
- [ ] 含广播的 shape 组合（多输入算子适用）
- [ ] 含重复索引 / 边界索引（index 类算子适用）

### 注释质量

- [ ] `Model` 类有 docstring（含算子语义说明）
- [ ] `forward` 参数有行内注释（含义、约束）
- [ ] 复杂逻辑有步骤拆解注释
- [ ] docstring 包含官网链接或数学公式

---

# 附录 · 三种来源对照速查

| 维度 | CANN 仓（情况①） | torch API 新写（情况②） | 组合拼装（情况③） |
|------|----------------|---------------------|----------------|
| 典型路径 | `AscendOpGenAgent/benchmarks/CANN/level1/` | 自己写 | 自己写 |
| `.py` 命名 | `{id}_{Op}.py` | `{id}_{Op}.py` | `{id}_{Op}.py` |
| `.json` 命名 | `{Op}.json`（**无 id 前缀**） | `{id}_{Op}.json` | `{id}_{Op}.json` |
| JSON 结构 | 扁平 kv | `{"inputs": [...]}` 数组 | `{"inputs": [...]}` 数组 |
| 是否 stub NPU 依赖 | 是（保留） | 否 | 否 |
| 注释重点 | 不需要写（已经成型） | 抄 torch 官网描述 + 小算子拆解 | 标注每步语义 + 数学公式 |
| 推荐操作 | 直接用或补 case | 全新写，注释要做 1.3.3 验证 | 全新写，必须做 1.3.3 验证 |
