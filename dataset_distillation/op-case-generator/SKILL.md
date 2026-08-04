---
name: op-case-generator
description: >
  生成或评估 AscendC/KernelBench 算子描述文件(.py)和测试用例文件(.json)。
  支持两种模式：生成模式(从算子语义生成 .py + .json 对)和评估模式
  (检查已有文件对的完整性、覆盖率、注释质量)。
argument-hint: >
  生成模式: op_name, op_class, torch_api, 输出目录。
  评估模式: .py 文件路径（自动发现同名 .json）。
  可选: --mode generate|evaluate, --format npu|cann。
---

# 算子描述文件与 Case 生成/评估 Skill

<role>
你是算子描述文件和测试用例的生成与评估专家。你能从算子语义出发生成
KernelBench 格式的 `.py` + `.json` 文件对，也能检查已有文件的完整性与覆盖率。
</role>

## 参考文档

本 skill 的行为规范以以下文档为准：

- **指导文档**: `/home/t00893162/auto_data_clean/算子描述文件和case生成指导.md`
  — 算子描述文件(.py)和 case 文件(.json)的完整规范、注释策略、覆盖矩阵、评估标准。
- **格式参考**: `@references/kernelbench-format.md`
  — JSONL 字段 schema 的快速参考（本 skill 自带副本）。

行动前先读取指导文档，以其中的决策树、骨架模板、覆盖矩阵为准。
指导文档中的"第三部分·完整性评估标准"是评估模式的核心判定依据。

---

## 模式判定

按用户输入自动判定：

1. 用户提供了算子名/语义/torch API 且要求"生成"或"创建" → **生成模式**
2. 用户提供了已有 `.py` 路径且要求"检查"/"评估"/"review" → **评估模式**
3. 用户只给了算子名，未明确模式 → 询问用户

---

# 生成模式

## 输入

| 参数 | 必填 | 说明 |
|------|:---:|------|
| `op_name` | ✅ | 算子标识，如 `25_GELU`（含 id 前缀）或 `GELU` |
| `op_class` | ✅ | 算子类型：`elementwise` / `reduction` / `normalization` / `index` / `comparison` / `matmul` / `creation` / `manipulation` / `composite` / `backward` |
| `torch_api` | ✅ | 参考 torch API，如 `torch.nn.functional.gelu` |
| `output_dir` | ✅ | 输出目录（.py 和 .json 写到这里） |
| `forward_args` | 否 | `Model.forward` 的参数描述（不提供则从 torch API 推断） |
| `supported_dtypes` | 否 | 算子支持的 dtype 列表，默认从 `op_class` 推断（见指导文档 §2.4） |
| `extra_notes` | 否 | 算子特有的 NPU 限制、uint 兼容要求、多输出说明等 |

## 生成流程

### Step 1: 信息收集

1. 读取指导文档的决策树（§1.1）和覆盖矩阵（§2.2）
2. 根据 `op_class` 确定：
   - 需要覆盖的 dtype（指导文档 §2.4）
   - 需要的 shape 维度种类（指导文档 §2.3）
   - 需要的 attr 取值数量（指导文档 §2.5）
   - 需要的边界 case（指导文档 §2.6）
   - 推荐 case 数量（指导文档 §2.2）
3. 查阅 torch 官网文档（`https://pytorch.org/docs/stable/`）获取 API 详细语义

### Step 2: 检查 CANN 仓（必做）

搜索 `/home/t00893162/AscendOpGenAgent/benchmarks/CANN/level{1,2,3,4}/` 下是否存在同名算子文件。

- **命中** → 复制 CANN 的 `.py` 和 `.json` 到 `output_dir`，按指导文档 §1.2 做适配（重命名 JSON、修文件名引用、补 case）。
- **未命中** → 进入 Step 3 新写。

### Step 3: 生成 .py 文件

按指导文档 §1.3 的骨架模板，生成包含以下内容的 `.py`：

1. **Imports 区**：仅 `os`, `json`, `torch`, `torch.nn`，必要时加 `torch.nn.functional as F`，禁止 NPU 依赖
2. **`class Model(nn.Module)`**：
   - docstring 含算子语义（中文）、数学公式、官网链接、NPU 实现要点
   - **`__init__` 为空**（NPUKernelBench 标准），所有参数放在 `forward` 中
   - `forward` 参数 + 行内注释（含义、约束、dtype 限制）
   - 复杂逻辑有步骤拆解注释（指导文档 §1.3.2 位置 D）
   - 若返回多个 tensor，返回 tuple 并注明各元素含义
3. **`get_input_groups()`**：读同名 JSON，按 inputs 数组生成每组输入
   - `dtype_map` 含全部 dtype 及别名（fp32/float32, fp16/float16, bf16/bfloat16, fp64/float64）
   - `make_tensor` 正确处理 int/bool/complex/uint 分支（指导文档 §1.3.1 骨架）
   - 按 `name` 匹配参数（不按位置硬编码），可选的 `required: false` 参数用条件判断
   - `tensor_list`：同时支持 `"shapes"` 字段（同 dtype）和 `"tensors"` 数组（异构）
4. **`get_init_inputs()`**：直接 `return []`（NPUKernelBench 标准模式）

**特殊算子类型额外要求**：
- **Backward 算子**：按指导文档 §1.6.2 模式，forward 中 fp16→fp32→fp16，注释标注梯度计算逻辑
- **多输出算子**：返回 tuple，注释标注每个输出的含义
- **uint dtype 算子**：forward 中显式 cast uint→float32（CPU torch 兼容），按指导文档 §1.6.5

### Step 4: 生成 .json 文件

按覆盖矩阵铺 case，每条一行 JSON：

1. **dtype 维度**：float16(占 40-50%) + float32(≥2) + bfloat16(≥2，如支持)
2. **shape 维度**：至少 3 种维度数 + 非对齐 + 超大(numel>2^18) + 标量
3. **attr 维度**：所有合法值都需要覆盖（指导文档 §2.5 最小覆盖表）
4. **边界**：required=false 的省略、广播、极端值（指导文档 §2.6）

case 数量控制在指导文档推荐范围内。

### Step 5: 自检

运行验证脚本：

```bash
python3 <skill-path>/scripts/validate_op.py \
    --py <output_dir>/<op_name>.py \
    --mode runtime
```

验证不通过 → 根据输出的 JSON 修复问题，最多重试 2 次。

### Step 6: 输出摘要

展示生成结果：
- 文件路径
- case 数量
- dtype 分布
- shape 维度分布
- 自检结果

---

# 评估模式

## 输入

| 参数 | 必填 | 说明 |
|------|:---:|------|
| `py_path` | ✅ | 算子描述文件路径（自动发现同名 .json） |
| `op_class` | 否 | 算子类型，不提供则从 `Model.forward` 推断 |

## 评估流程

### Step 1: 静态检查

```bash
python3 <skill-path>/scripts/validate_op.py \
    --py <py_path> \
    --mode static
```

检查内容：
- 文件存在、命名一致
- `.py` 可 import
- `Model`、`get_input_groups`、`get_init_inputs` 存在
- 三者行数对齐

静态检查失败 → 报告 FAIL，**不再进行后续步骤**。

### Step 2: 运行时检查

```bash
python3 <skill-path>/scripts/validate_op.py \
    --py <py_path> \
    --mode runtime
```

检查内容：
- 每组 case 的 `Model(*init)(*inputs)` 可执行
- 输出为 Tensor（或 tuple of Tensor）
- 输出不含 NaN/Inf（isinf/isnan 类算子除外）

运行时检查失败 → 逐 case 报告失败原因。

### Step 3: 覆盖率分析

按评估标准的 5 个维度逐项检查（指导文档 §3.3）：

#### dtype 覆盖 (25%)
- float16 占比 ≥ 30%? (10分)
- float32 ≥ 2 case? (5分)
- bfloat16 ≥ 2 case（如适用）? (5分)
- 整数/bool 类型有覆盖（如适用）? (5分)

#### shape 覆盖 (25%)
- ≥ 3 种维度数? (8分)
- 含非对齐 shape? (5分)
- 含超大 shape (numel > 2^18)? (5分)
- 含标量/极小 shape? (7分)

#### attr 覆盖 (15%)
- bool attr: True/False 都出现? (5分)
- 数值/enum attr: 全部取值? (5分)
- dim: 正/负/0? (5分)

#### 边界 case (20%)
- required=false 被省略? (8分)
- 广播 shape 组合? (6分)
- 重复/边界索引? (6分)

#### 注释质量 (15%)
- Model docstring 含语义说明? (5分)
- forward 参数有注释? (5分)
- 复杂逻辑有步骤拆解? (5分)

### Step 4: 输出评估报告

评估报告为 JSON 格式，包含：

```json
{
  "status": "PASS|WARN|FAIL",
  "score": 85,
  "static_check": {"passed": true, "issues": []},
  "runtime_check": {"passed": true, "total": 30, "passed_cases": 30, "failed_cases": 0, "failures": []},
  "coverage": {
    "dtype": {"score": 25, "max": 25, "details": [...]},
    "shape": {"score": 20, "max": 25, "details": [...]},
    "attr": {"score": 12, "max": 15, "details": [...]},
    "boundary": {"score": 14, "max": 20, "details": [...]},
    "annotation": {"score": 15, "max": 15, "details": [...]}
  },
  "suggestions": [
    "缺少 bfloat16 case：当前 0 个，建议至少 2 个",
    "缺少非对齐 shape：建议添加 [13]、[17, 31] 等"
  ]
}
```

评估等级（指导文档 §3.2）：
- PASS: 一票否决通过 + 总分 ≥ 80
- WARN: 一票否决通过 + 总分 60–79
- FAIL: 一票否决不通过，或总分 < 60

**向用户展示评估摘要**：等级、总分、各维度得分、关键缺失项 Top 5。

---

## 关键约束

| 约束 | 说明 |
|------|------|
| 指导文档优先 | 所有规则以 `/home/t00893162/auto_data_clean/算子描述文件和case生成指导.md` 为准 |
| 禁止 NPU 依赖 | 新写的 `.py` 不能 import torch_npu/framework |
| 命名一致 | `.py` 和 `.json` 必须同名同目录 |
| 三数对齐 | `len(groups) == len(inits) == JSON行数` |
| 验证必做 | 生成后和评估时都要跑 `validate_op.py` |
| CANN 优先 | 生成前必须先查 CANN 仓 |
| 格式默认 | 新写统一用 inputs 数组格式（格式 A） |

---

## 自检脚本

位于 `<skill-path>/scripts/validate_op.py`。

用法：

```bash
# 静态检查（不需 torch，纯 AST 分析）
python3 validate_op.py --py /path/to/op.py --mode static

# 运行时检查（需 torch 环境）
python3 validate_op.py --py /path/to/op.py --mode runtime

# 覆盖率分析（需 torch 环境）
python3 validate_op.py --py /path/to/op.py --mode coverage --op-class elementwise

# 全量检查
python3 validate_op.py --py /path/to/op.py --mode all --op-class elementwise
```
