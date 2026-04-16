# 单轮数据集转多轮对话数据集

本项目用于将包含思维链（`<think>...</think>`）的单轮对话数据集，自动转换为标准的多轮对话格式。

## 核心功能

- **智能拆分思维链**：根据思维内容的语义逻辑，自动识别 3~6 个自然断点，将长思维链拆分为多个连续的问答轮次
- **自动生成追问问题**：每个思维阶段都会配有一个合理的 `user` 追问问题，体现从宏观分析到微观实现的渐进过程
- **严格保持内容一致**：所有 `assistant` 回答均来自原始数据集的 verbatim 切片，确保不丢失、不篡改任何原文
- **代码输出完整保留**：`</think>` 后的代码/文件内容作为最终轮次完整输出

## 文件说明

| 文件 | 说明 |
|---|---|
| `dataset.json` | 原始单轮数据集（输入） |
| `dataset_multi_round.json` | 转换后的多轮数据集（输出） |
| `.claude/skills/convert/scripts/convert_to_multi_round.py` | 核心转换脚本 |
| `.claude/skills/convert/SKILL.md` | Claude Code Skill 定义文件，支持通过自然语言触发转换 |
| `dataset_to_multi_round_skill.md` | 完整的设计规范文档（供人阅读） |

## 使用方法

### 方式一：直接运行脚本

```bash
python3 .claude/skills/convert/scripts/convert_to_multi_round.py
```

脚本默认读取当前目录下的 `dataset.json`，输出到 `dataset_multi_round.json`。

如需处理其他路径的数据，通过命令行参数传入：

```bash
python3 .claude/skills/convert/scripts/convert_to_multi_round.py \
  --input_path "/path/to/your/dataset.json" \
  --output_path "/path/to/output.json"
```

#### 大数据集处理（推荐）

当数据集条目数较多（如 > 1000）时，建议启用**分块处理 + 断点续跑 + 多文件输出**，以避免单次运行时间过长或中途失败导致全部重来：

```bash
python3 .claude/skills/convert/scripts/convert_to_multi_round.py \
  --input_path "/path/to/large_dataset.json" \
  --output_path "/path/to/output.json" \
  --chunk_size 500 \
  --resume \
  --num_parts 4
```

参数说明：
- `--chunk_size 500`：每处理 500 条自动保存一次 checkpoint，粒度越细容错性越高
- `--resume`：如果之前中断过，自动从 checkpoint 继续处理
- `--num_parts 4`：最终结果拆分为 `output.part001.json` ~ `output.part004.json`，便于后续分批加载

运行结束后，checkpoint 文件会自动清理。

### 方式二：通过 Claude Code Skill 触发

如果你已将 `SKILL.md` 注册到 Claude Code 的 skill 目录中，可以直接用自然语言触发。支持的说法包括：

```
把数据集转成多轮
```

```
单轮数据集转多轮
```

```
拆分思维链为多轮
```

**带参数调用**：在自然语言中直接说明 `input_path` 和 `output_path` 即可，例如：

```
把数据集转成多轮，input_path 是 /path/to/dataset.json，output_path 是 /path/to/output.json
```

```
单轮数据集转多轮，输出到 /path/to/output.json
```

Claude 会自动解析参数，并调用 `convert_to_multi_round.py --input_path ... --output_path ...` 完成转换，最后汇报结果统计。

如果不指定参数，将使用默认值：
- `input_path`：`/home/l00899543/gen_sft_datasets/dataset.json`
- `output_path`：`/home/l00899543/gen_sft_datasets/dataset_multi_round.json`

## 技术原理

转换采用**"脚本 + LLM Skill"**的混合方案：

1. **LLM 负责智能分析**：读取每条数据的 `think_content`，分析逻辑结构，返回断点标记（`split_marker`）和每轮的追问问题（`user_question`）
2. **Python 负责精确切分**：根据 LLM 返回的断点在原文中精确定位，做 verbatim 文本切片，并组装成标准的 `messages` 格式
3. **自动校验**：拼接所有 assistant 回答，与原始输出做长度和内容比对，校验不通过时自动回退到原始单轮格式并标记 `needs_review`

## 输出格式

转换后的数据集为 JSON 数组，每个元素结构如下：

```json
{
  "system": "原始 system 提示词",
  "messages": [
    {"role": "user", "content": "初始问题"},
    {"role": "assistant", "content": "第一阶段思维内容（原文）"},
    {"role": "user", "content": "第二阶段追问问题"},
    {"role": "assistant", "content": "第二阶段思维内容（原文）"},
    ...
    {"role": "user", "content": "引出代码的问题"},
    {"role": "assistant", "content": "代码/文件内容（原文）"}
  ],
  "needs_review": false,
  "phases": ["任务理解", "硬件分析", "Tiling设计", "代码实现"]
}
```

## 典型转换效果

以 72 条 Ascend C 算子开发数据集为例：

- 总条目数：72
- 成功转换率：100%
- 消息轮次范围：8 ~ 14 条（平均 11.2 条）
- 思维阶段范围：3 ~ 6 个（平均 4.6 个）
- 需要复核数：0

## 常见问题

### 1. API 调用失败或返回解析错误

脚本依赖环境变量中的 API 配置。如果运行时报错，请检查：

```bash
# 当前环境是否已配置
env | grep ANTHROPIC
```

脚本会自动尝试读取 `~/.claude/settings.json` 中的 `ANTHROPIC_AUTH_TOKEN` 和 `ANTHROPIC_BASE_URL`。如果环境变量和配置文件都没有，转换将无法进行。

### 2. 某条数据被标记为 `needs_review: true`

这通常是因为：
- LLM 分析返回的格式异常
- 自动校验发现内容长度差异过大（可能截断）
- 找不到有效的 `split_marker`

被标记的数据会回退为原始单轮格式，不会丢失数据。你可以单独取出这些数据，手工调整或重新运行转换。

### 3. 处理大数据集时中断或耗时太久

对于条目数超过几千的数据集，转换可能需要较长时间，且容易因 API rate limit 或会话超时而中断。推荐做法：

1. **启用分块 + 断点续跑**：加上 `--chunk_size 500 --resume`，即使中断也能从中断点继续，不会丢失已处理的结果。
2. **拆分输出文件**：加上 `--num_parts 4`（或更多），将结果拆分为多个小文件，避免单文件过大导致后续加载困难。
3. **降低并发**：如果 API 频繁返回 rate limit，可编辑脚本中的 `MAX_WORKERS = 8`，适当调小并发数。

### 4. 如何修改拆分阶段的数量或风格

编辑 `convert_to_multi_round.py` 中的 `ANALYZE_PROMPT`，调整以下部分：

- `"划分为 3~6 个自然阶段"`：可修改阶段数量范围
- `"典型阶段参考"`：可增减或替换阶段类型描述
- `"问题构造原则"`：可调整追问问题的风格要求

修改后重新运行脚本即可。

## 依赖

- Python 3.10+
- `anthropic` Python SDK
- `httpx`

安装依赖：

```bash
pip install anthropic httpx
```
