---
name: dataset-to-multi-round
description: 将单轮对话数据集转换为多轮对话格式，自动拆分思维链并构造合理的追问问题
version: "1.0"
tools:
  write: true
  edit: true
  bash: true
  skill: true
  read: true
  
triggers:
  - 把数据集转成多轮
  - 单轮数据集转多轮
  - 转换数据集为多轮对话
  - dataset to multi round
  - 拆分思维链为多轮
args:
  - name: input_path
    description: 输入的单轮数据集 JSON 文件路径
    required: false
    default: "/home/l00899543/gen_sft_datasets/dataset.json"
  - name: output_path
    description: 输出的多轮数据集 JSON 文件路径
    required: false
    default: "/home/l00899543/gen_sft_datasets/dataset_multi_round.json"
---

# 单轮数据集转多轮对话数据集

## 任务目标

将单轮对话数据集转换为多轮对话形式。转换后的数据集需要：

1. 保留原始数据集中的 `system`、`instruction`、`input` 等字段
2. 将 `output` 中的思维链（`<think>...</think>`）合理拆分为多个连续的问答轮次
3. 拆分后的每一轮回答必须与原始数据集中的对应内容保持一致
4. 每一轮的问题需要结合当前思维阶段和最终目标自行构造，要求自然、合理
5. 思维链结束后，最后一轮或多轮引出并呈现实际的代码/文件输出

## 执行方式

本 Skill 的核心执行逻辑已经封装在脚本 `/home/l00899543/gen_sft_datasets/.claude/skills/convert/scripts/convert_to_multi_round.py` 中。

当被用户触发时，你应该：

1. **确认输入/输出路径**
   - 如果用户未指定，使用默认路径：
     - 输入：`/home/l00899543/gen_sft_datasets/dataset.json`
     - 输出：`/home/l00899543/gen_sft_datasets/dataset_multi_round.json`

2. **检查脚本存在性**
   - 确认 `/home/l00899543/gen_sft_datasets/.claude/skills/convert/scripts/convert_to_multi_round.py` 存在
   - 如果不存在，你需要根据本 Skill 的规范重新生成该脚本

3. **运行转换脚本**
   - 在 Bash 中执行，并将路径参数传入：
     ```bash
     python3 /home/l00899543/gen_sft_datasets/.claude/skills/convert/scripts/convert_to_multi_round.py \
       --input_path "{input_path}" \
       --output_path "{output_path}"
     ```
   - 对于大数据集（条目数 > 1000），建议启用 **分块处理 + 断点续跑** 以及 **自动拆分为多个输出文件**，避免单次运行过久或失败：
     ```bash
     python3 /home/l00899543/gen_sft_datasets/.claude/skills/convert/scripts/convert_to_multi_round.py \
       --input_path "{input_path}" \
       --output_path "{output_path}" \
       --chunk_size 500 \
       --resume \
       --num_parts 4
     ```
   - 脚本会自动读取环境变量中的 API 配置（`ANTHROPIC_AUTH_TOKEN` 和 `ANTHROPIC_BASE_URL`）

4. **验证输出质量**
   - 读取生成的 `dataset_multi_round.json`
   - 统计总条目数、消息轮次范围、阶段数、格式错误数
   - 随机抽查 1~2 条数据，确认 user/assistant 交替正确，且 assistant 回答拼接后与原始 output 一致

5. **向用户汇报结果**
   - 报告成功转换的条数、平均轮次、阶段分布
   - 如有 `needs_review` 标记的数据，指出数量和索引
   - 如果使用了 `--num_parts` 拆分输出，列出所有生成的文件路径

## 大数据集处理参数说明

针对条目数较多（> 1000）的数据集，脚本提供以下参数以避免运行过久或中途失败：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--chunk_size` | `500` | 每处理多少条保存一次 checkpoint。数值越小，断点粒度越细，容错性越高。 |
| `--resume` | `False` | 若存在 checkpoint，则从上次中断的位置继续处理，无需从头开始。 |
| `--checkpoint_path` | 自动生成 | checkpoint 文件路径，默认与 `--output_path` 同名并追加 `.checkpoint.json`。 |
| `--num_parts` | `1` | 将最终结果拆分为多少个 JSON 文件输出（例如 `--num_parts 4` 会生成 `output.part001.json` ~ `output.part004.json`）。 |

## 脚本不存在时的兜底实现

如果 `convert_to_multi_round.py` 不存在，你需要重建它。脚本的核心逻辑应遵循以下设计：

### 第一阶段：LLM 分析思维结构

对每条数据的 `think_content`（去除 `<think>` 标签后的思维链），向 LLM 发送分析请求：

- **任务**：将思维链划分为 3~6 个自然阶段
- **输出**：JSON 数组，每个元素包含 `phase_title`、`split_marker`、`user_question`
- **约束**：`split_marker` 必须取自原文，用于脚本精确定位切分点

### 第二阶段：脚本精确切分与组装

1. 根据 `split_marker` 在原文中定位，将 `think_content` 切分为多个 `think_pieces`
2. 组装 `messages` 列表：
   - 首条：`user` 的 `initial_query`
   - 中间：`assistant` 的 think piece → `user` 的追问问题，交替进行
   - 尾条：`user` 引出代码的问题 → `assistant` 的 `code_content` 完整原文
3. 做长度校验，确保内容未被截断
4. 失败时回退到原始单轮格式，并标记 `needs_review: true`

### 核心 Prompt 模板（脚本内嵌使用）

**System Prompt（分析阶段）**：

```
你是一位专业的数据集工程师。请阅读下面的 think_content，将其划分为 3~6 个自然阶段。

划分原则：
- 每个阶段必须是完整、自洽的推理步骤
- 断点应在段落结束、主题转换或决策完成处
- 不能在公式推导、条件判断、或分析过程的中间切断

输出格式：JSON 数组，每个元素包含：
- phase_title: 阶段主题
- split_marker: 阶段结束位置附近的原文句子（15~40字，有区分度，用于脚本定位）
- user_question: 引导进入下一阶段的问题（渐进深入、自然多样、目标导向）

最后一个阶段不需要 split_marker，但必须有 user_question（用于引出代码输出）。
```

## 关键约束

1. **内容一致性**：拆分后的所有 assistant 回答拼接起来必须与原始 `output` 等价
2. **问题合理性**：问题必须与对应阶段的思维内容高度相关
3. **分段灵活性**：不同条目分段数量可以不同，但必须符合逻辑
4. **代码完整性**：代码部分不得截断、修改或遗漏

## 输出格式示例

```json
{
  "system": "...",
  "messages": [
    {"role": "user", "content": "初始问题"},
    {"role": "assistant", "content": "第一阶段思维内容"},
    {"role": "user", "content": "第二阶段追问"},
    {"role": "assistant", "content": "第二阶段思维内容"},
    {"role": "user", "content": "请给出完整代码"},
    {"role": "assistant", "content": "代码内容"}
  ],
  "needs_review": false,
  "phases": ["任务理解", "硬件分析", "Tiling设计", "代码实现"]
}
```
