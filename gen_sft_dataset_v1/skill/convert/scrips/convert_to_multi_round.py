#!/usr/bin/env python3
"""
单轮数据集 -> 多轮对话数据集 转换脚本
核心策略：LLM 只负责分析断点和生成问题；文本切片和组装由脚本完成。

改进点（应对大数据集）：
- 分块处理（chunk processing），每处理完一块就保存 checkpoint
- 支持断点续跑（--resume），中断后可以从上次进度继续
- 支持将结果自动拆分为多个输出文件（--num_parts）
"""
import argparse
import json
import math
import os
import re
import time
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
from anthropic import Anthropic, APIError, RateLimitError

# ==================== 默认配置 ====================
DEFAULT_INPUT_PATH = "/home/l00899543/gen_sft_datasets/dataset.json"
DEFAULT_OUTPUT_PATH = "/home/l00899543/gen_sft_datasets/dataset_multi_round.json"
DEFAULT_CHUNK_SIZE = 500


def _load_api_config():
    try:
        with open("/root/.claude/settings.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("env", {})
    except Exception:
        return {}


_api_env = _load_api_config()
API_KEY = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or _api_env.get("ANTHROPIC_AUTH_TOKEN", "")
BASE_URL = os.environ.get("ANTHROPIC_BASE_URL") or _api_env.get("ANTHROPIC_BASE_URL", "")
MODEL = "claude-sonnet-4-6"

# 并发数
MAX_WORKERS = 8
LIMIT = None

# LLM 可用性探测标志：None=未探测, True=可用, False=不可用（跳过所有 LLM 调用）
_llm_available = None

# ==================== Prompts ====================
ANALYZE_PROMPT = """你是一位专业的数据集工程师。你的任务是对一段思维链进行结构化分析。

请阅读下面的 `think_content`（模型在生成代码前的思考过程），将其划分为 3~6 个自然阶段。

## 划分原则
- 每个阶段必须是思维链中一个完整、自洽的推理步骤
- 断点应出现在段落结束、主题转换或决策完成的位置
- 绝对不能在一个公式推导、条件判断、或分析过程的中间切断
- 不同条目思维结构不同，请结合具体内容灵活判断

## 典型阶段参考（灵活运用）
1. 任务理解与算子分析
2. 硬件与约束分析
3. Tiling 策略设计
4. 单核处理逻辑
5. 多核并行与调度
6. 代码实现规划

## 输出格式
必须且仅输出一个合法的 JSON 数组，每个元素包含：
- `phase_title`: 该阶段的主题（如 "Tiling策略设计"）
- `split_marker`: 该阶段结束位置附近的一句原文（15~40 个字符），用于脚本在原文中定位切分点。请尽量选择有区分度、不容易在其他地方重复出现的句子。
- `user_question`: 引导进入下一阶段的问题。要求：
  * 渐进深入，从宏观到微观
  * 引用或承接前面已讨论的结论
  * 目标导向，始终指向最终生成完整代码
  * 自然多样，像有经验的工程师在确认方案

示例输出：
[
  {
    "phase_title": "任务理解与计算单元选择",
    "split_marker": "这些计算都是元素级别的，不包含复杂的矩阵乘法。",
    "user_question": "首先，请你帮我梳理一下这个算子的核心功能和数学原理，以及实现时应该选用哪种计算单元？"
  },
  {
    "phase_title": "硬件约束与数据划分",
    "split_marker": "我会定义两个变量 valueN 和 valueM 来表示这个逻辑二维矩阵的维度。",
    "user_question": "好的，明白了。接下来请你结合给定的输入 shape 和硬件平台信息，分析一下主要约束，并确定逻辑上的数据划分方式。"
  }
]

注意：
- 最后一个阶段不需要 `split_marker`（因为不需要再切分），但必须有 `phase_title` 和 `user_question`（这个问题将引出最终的代码输出）
- 请确保 `split_marker` 尽量取自原文的完整句子或语义完整的片段，以便脚本能精确定位
"""

USER_ANALYZE_TEMPLATE = """请分析以下思维链：

### initial_query
{initial_query}

### think_content
{think_content}

请严格按照上述要求输出 JSON 数组。"""


# ==================== 辅助函数 ====================

def extract_think_and_code(output: str) -> tuple[str, str]:
    think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        code_content = output[think_match.end():].strip()
    else:
        think_content = output.strip()
        code_content = ""
    return think_content, code_content


def create_client() -> Anthropic:
    http_client = httpx.Client(verify=False)
    return Anthropic(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)


def call_llm_for_analysis(client: Anthropic, system: str, user_prompt: str) -> list[dict] | None:
    global _llm_available
    if _llm_available is False:
        return None

    max_retries = 5
    base_delay = 2.0
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                temperature=0.3,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = response.content[0].text.strip()
            for prefix in ["```json", "```"]:
                if raw_text.startswith(prefix):
                    raw_text = raw_text[len(prefix):].strip()
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()
            parsed = json.loads(raw_text)
            if isinstance(parsed, list) and len(parsed) >= 2:
                _llm_available = True
                return parsed
            return None
        except (APIError, RateLimitError) as e:
            err_msg = str(e).lower()
            if "engine is currently overloaded" in err_msg or "rate_limit" in err_msg:
                _llm_available = False
                print("  [API unavailable] Engine overloaded/rate limited. Skipping LLM for remaining items.")
                return None
            delay = base_delay * (2 ** attempt) + (attempt * 0.5)
            print(f"  [API error attempt {attempt + 1}/{max_retries}] {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
        except Exception as e:
            print(f"  [API/parse error] {e}")
            return None
    print(f"  [API error] All {max_retries} retries exhausted.")
    _llm_available = False
    return None


def find_split_index(text: str, marker: str, start: int = 0) -> int:
    """
    在 text 中从 start 位置开始查找 marker，返回 marker 结束后的索引。
    优先精确匹配，否则使用近似匹配。
    """
    idx = text.find(marker, start)
    if idx != -1:
        return idx + len(marker)
    # 模糊匹配：找最相似的子串
    marker_len = len(marker)
    best_ratio = 0.0
    best_idx = -1
    # 滑动窗口搜索（限制搜索范围以提高效率）
    step = max(1, marker_len // 4)
    search_end = len(text) - marker_len + 1
    for i in range(start, search_end, step):
        chunk = text[i:i + marker_len]
        ratio = difflib.SequenceMatcher(None, marker, chunk).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
    if best_ratio > 0.7 and best_idx != -1:
        return best_idx + marker_len
    return -1


def split_think_content(think_content: str, phases: list[dict]) -> list[str]:
    """
    根据 phases 中的 split_marker 切分 think_content。
    返回每个 phase 对应的文本片段列表。
    """
    splits = []
    start = 0
    for i, phase in enumerate(phases):
        marker = phase.get("split_marker", "")
        if not marker or i == len(phases) - 1:
            # 最后一个阶段直接取剩余全部
            splits.append(think_content[start:].strip())
            break
        end = find_split_index(think_content, marker, start)
        if end == -1 or end <= start:
            # 找不到 marker，fallback：平均切分
            remaining_phases = len(phases) - i
            approx_len = (len(think_content) - start) // remaining_phases
            end = start + approx_len
        splits.append(think_content[start:end].strip())
        start = end
    return splits


def build_messages(system: str, initial_query: str, phases: list[dict], think_pieces: list[str], code_content: str) -> list[dict]:
    messages = [{"role": "user", "content": initial_query}]
    n = len(phases)
    for i in range(n):
        piece = think_pieces[i] if i < len(think_pieces) else ""
        messages.append({"role": "assistant", "content": piece})
        if i < n - 1:
            # 中间 phases：用当前 phase 的 user_question 作为追问，引导进入下一个 think_piece
            next_q = phases[i].get("user_question", "请继续。")
            messages.append({"role": "user", "content": next_q})
        elif i == n - 1:
            # 最后一个 think_piece 之后，用最后一个 phase 的 user_question 引出代码
            final_q = phases[-1].get("user_question", "请给出完整的代码实现。")
            messages.append({"role": "user", "content": final_q})
    # 最后一轮 assistant 永远是 code_content
    messages.append({"role": "assistant", "content": code_content})
    return messages


def validate_messages(messages: list[dict], original_output: str) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    expected_role = "user"
    for msg in messages:
        if msg.get("role") != expected_role:
            return False
        expected_role = "assistant" if expected_role == "user" else "user"

    assistant_parts = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
    combined = "\n\n".join(assistant_parts).strip()

    norm_original = re.sub(r'<think>.*?</think>', '', original_output, flags=re.DOTALL).strip()
    # 允许一定差异：去除空行后比较长度比例
    len_orig = len(norm_original.replace(" ", "").replace("\n", ""))
    len_comb = len(combined.replace(" ", "").replace("\n", ""))
    ratio = min(len_orig, len_comb) / max(len_orig, 1)
    if ratio < 0.85:
        print(f"  [Validation warning] length ratio={ratio:.2f}, might be truncated.")
        return False
    return True


def _get_first_meaningful_line(piece: str) -> str:
    """获取第一个非空行的内容（小写）。"""
    for line in piece.split('\n'):
        stripped = line.strip().lower()
        if stripped:
            return stripped
    return ""


def _keyword_matches(text: str, keywords: list[str]) -> bool:
    """匹配关键词，纯英文使用词边界，中文或混合使用子串匹配。"""
    for k in keywords:
        if re.match(r'^[a-zA-Z]+$', k):
            if re.search(rf'\b{re.escape(k)}\b', text):
                return True
        else:
            if k in text:
                return True
    return False


def _infer_phase_title(piece: str, piece_index: int, total_pieces: int) -> str:
    """根据文本内容推断阶段主题，优先参考段落开头的 markdown 标题或第一行。"""
    # 第一段几乎总是任务理解
    if piece_index == 0:
        return "任务理解与算子分析"

    first_line = _get_first_meaningful_line(piece)

    # 强信号：第一行直接包含主题关键词
    if _keyword_matches(first_line, ["单核内部", "每个核", "一个核心", "kernel侧", "执行流程", "循环结构", "for (", "compute", "内层循环", "模板实例", "pipeline", "pipe", "数据搬入", "数据搬出"]):
        return "单核处理逻辑设计"
    if _keyword_matches(first_line, ["多核并行", "blockdim", "并行调度", "同步", "核间", "block_idx"]):
        return "多核并行与调度分析"
    if _keyword_matches(first_line, ["tiling策略", "分块", "数据划分", "切分策略", "tilingkey", "分块信息", "tilingdata", "ub空间", "数据块", "tile"]):
        return "Tiling 与数据划分策略"
    if _keyword_matches(first_line, ["host侧", "op_host", "def.cpp", "kernel_launch", "启动参数", "blockdim设置"]):
        return "Host 侧与启动配置"
    if _keyword_matches(first_line, ["硬件资源分析", "数据对齐", "ub_size", "vector_core", "ai core", "约束分析", "内存占用", "硬件规格"]):
        return "硬件约束与平台分析"

    # 匹配 markdown 标题
    header_match = re.search(r'^(?:\s*)#+\s*\d*\.?\s*(.+)', piece, re.MULTILINE)
    if not header_match:
        header_match = re.search(r'\*\s*\*\*\s*(\d+\.\s*.+?)\*\*', piece)
    if header_match:
        header = header_match.group(1).lower()
        if _keyword_matches(header, ["kernel", "单核", "循环", "执行流程", "处理逻辑", "pipeline", "数据搬入"]):
            return "单核处理逻辑设计"
        if _keyword_matches(header, ["多核", "并行", "调度", "数据搬入搬出"]):
            return "多核并行与调度分析"
        if _keyword_matches(header, ["tiling", "分块", "数据划分", "切分策略", "tilingdata", "ub分析"]):
            return "Tiling 与数据划分策略"
        if _keyword_matches(header, ["host", "启动"]):
            return "Host 侧与启动配置"
        if _keyword_matches(header, ["约束", "硬件", "内存", "资源分析"]):
            return "硬件约束与平台分析"

    preview = piece[:400].lower()
    if _keyword_matches(preview, ["tiling策略", "分块维度", "ub空间分析", "切分策略", "n_axis", "d_axis", "数据块"]):
        return "Tiling 与数据划分策略"
    if _keyword_matches(preview, ["kernel侧执行", "单核内部", "循环结构", "for (", "compute", "内层循环", "pipe_barrier"]):
        return "单核处理逻辑设计"
    if _keyword_matches(preview, ["多核并行", "blockdim", "数据搬入搬出", "同步", "核间"]):
        return "多核并行与调度分析"
    if _keyword_matches(preview, ["硬件资源分析", "数据对齐", "ub_size", "vector_core"]):
        return "硬件约束与平台分析"
    if _keyword_matches(preview, ["host侧", "op_host", "def.cpp", "kernel_launch", "启动参数"]):
        return "Host 侧与启动配置"

    return "代码实现与细节规划" if piece_index == total_pieces - 1 else "任务理解与算子分析"


def _generate_user_question(next_piece: str, is_code_question: bool = False) -> str:
    """根据下一段 assistant 内容生成匹配的 user 追问，优先参考第一行/标题。"""
    if is_code_question:
        return "好的，思路已经很清晰了。请你给出完整的代码实现。"

    first_line = _get_first_meaningful_line(next_piece)

    # 强信号：第一行直接包含主题关键词
    if _keyword_matches(first_line, ["单核内部", "每个核", "一个核心", "kernel侧", "执行流程", "循环结构", "for (", "compute", "内层循环", "pipeline", "pipe", "数据搬入", "数据搬出", "模板实例"]):
        return "接下来，请你详细描述一下单核内的处理逻辑和循环结构。"
    if _keyword_matches(first_line, ["host侧", "op_host", "def.cpp", "kernel_launch", "启动参数", "blockdim设置"]):
        return "请你进一步说明 Host 侧的实现，包括 Tiling 计算和 Kernel 启动参数的设置。"
    if _keyword_matches(first_line, ["多核并行", "blockdim", "并行调度", "同步", "核间", "block_idx", "数据搬入搬出"]):
        return "再进一步，请你分析一下多核并行的调度方式，以及数据搬入搬出的流程。"
    if _keyword_matches(first_line, ["tiling策略", "分块", "数据划分", "切分策略", "tilingkey", "分块信息", "tilingdata", "ub空间", "数据块", "tile", "n_axis", "d_axis"]):
        return "明白了。基于前面的分析，请你设计一下数据划分（Tiling）策略，包括 UB 和 L1 的使用方式。"
    if _keyword_matches(first_line, ["硬件资源分析", "数据对齐", "ub_size", "vector_core", "ai core", "约束分析", "内存占用", "硬件规格"]):
        return "好的，了解了。接下来请你结合给定的输入 shape 和硬件平台信息，分析一下主要的约束条件。"
    if _keyword_matches(first_line, ["数学原理", "公式", "计算逻辑", "功能描述", "输入输出", "前向计算", "反向传播", "mean", "rstd", "算子功能"]):
        return "首先，请你帮我梳理一下这个算子的核心功能、数学原理以及实现时需要注意的关键点。"

    # 匹配 markdown 标题
    header_match = re.search(r'^(?:\s*)#+\s*\d*\.?\s*(.+)', next_piece, re.MULTILINE)
    if not header_match:
        header_match = re.search(r'\*\s*\*\*\s*(\d+\.\s*.+?)\*\*', next_piece)
    if header_match:
        header = header_match.group(1).lower()
        if _keyword_matches(header, ["kernel", "单核", "循环", "执行流程", "处理逻辑", "pipeline", "数据搬入"]):
            return "接下来，请你详细描述一下单核内的处理逻辑和循环结构。"
        if _keyword_matches(header, ["host", "启动"]):
            return "请你进一步说明 Host 侧的实现，包括 Tiling 计算和 Kernel 启动参数的设置。"
        if _keyword_matches(header, ["多核", "并行", "调度", "数据搬入搬出"]):
            return "再进一步，请你分析一下多核并行的调度方式，以及数据搬入搬出的流程。"
        if _keyword_matches(header, ["tiling", "分块", "数据划分", "切分策略", "tilingdata", "ub分析"]):
            return "明白了。基于前面的分析，请你设计一下数据划分（Tiling）策略，包括 UB 和 L1 的使用方式。"
        if _keyword_matches(header, ["约束", "硬件", "内存", "资源分析"]):
            return "好的，了解了。接下来请你结合给定的输入 shape 和硬件平台信息，分析一下主要的约束条件。"

    preview = next_piece[:400].lower()
    if _keyword_matches(preview, ["kernel侧执行", "基于 `tilingkey`", "循环结构", "for (", "compute", "内层循环"]):
        return "接下来，请你详细描述一下单核内的处理逻辑和循环结构。"
    if _keyword_matches(preview, ["host侧", "op_host", "def.cpp", "kernel_launch", "启动参数"]):
        return "请你进一步说明 Host 侧的实现，包括 Tiling 计算和 Kernel 启动参数的设置。"
    if _keyword_matches(preview, ["多核并行", "blockdim", "数据搬入搬出", "同步", "核间"]):
        return "再进一步，请你分析一下多核并行的调度方式，以及数据搬入搬出的流程。"
    if _keyword_matches(preview, ["tiling策略", "分块维度", "ub空间分析", "切分策略", "n_axis", "d_axis"]):
        return "明白了。基于前面的分析，请你设计一下数据划分（Tiling）策略，包括 UB 和 L1 的使用方式。"
    if _keyword_matches(preview, ["硬件资源分析", "数据对齐", "ub_size", "vector_core", "ai core"]):
        return "好的，了解了。接下来请你结合给定的输入 shape 和硬件平台信息，分析一下主要的约束条件。"
    if _keyword_matches(preview, ["数学原理", "公式", "计算逻辑", "功能描述", "输入输出", "前向计算", "反向传播", "mean", "rstd"]):
        return "首先，请你帮我梳理一下这个算子的核心功能、数学原理以及实现时需要注意的关键点。"

    return "请你继续深入分析下一步的实现细节。"


def heuristic_split_think_content(think_content: str) -> tuple[list[dict], list[str]] | tuple[None, None]:
    """
    当 LLM 不可用时，使用启发式规则将 think_content 切分为 3~6 个阶段。
    直接按段落边界分组，避免 mid-sentence 切断；并根据下一段内容生成匹配的 user_question。
    返回 (phases, think_pieces)。
    """
    paragraphs = [p.strip() for p in think_content.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        return None, None

    total_len = sum(len(p) for p in paragraphs)
    target_phases = min(6, max(3, total_len // 800))
    target_len_per_phase = total_len / target_phases

    groups = []
    current_group = []
    current_len = 0

    for p in paragraphs:
        current_group.append(p)
        current_len += len(p)
        if current_len >= target_len_per_phase and len(groups) < target_phases - 1:
            groups.append("\n\n".join(current_group))
            current_group = []
            current_len = 0
    if current_group:
        if groups:
            groups[-1] += "\n\n" + "\n\n".join(current_group)
        else:
            groups.append("\n\n".join(current_group))

    if len(groups) < 2:
        return None, None
    while len(groups) > 6:
        groups[-2] += "\n\n" + groups[-1]
        groups.pop()

    phases = []
    for i, group_text in enumerate(groups):
        phases.append({
            "phase_title": _infer_phase_title(group_text, i, len(groups)),
            "split_marker": "",  # 启发式模式直接返回 pieces，不依赖 marker 定位
            "user_question": _generate_user_question(
                groups[i + 1] if i + 1 < len(groups) else "",
                is_code_question=(i == len(groups) - 1),
            ),
        })

    return phases, groups


def build_fallback_item(item: dict) -> dict:
    initial_query = (item.get("instruction", "") + "\n" + item.get("input", "")).strip()
    return {
        "system": item.get("system", ""),
        "messages": [
            {"role": "user", "content": initial_query},
            {"role": "assistant", "content": item.get("output", "")}
        ],
        "needs_review": True,
        "review_reason": "LLM analysis failed or validation did not pass"
    }


def process_one_item(args: tuple[int, dict, Anthropic]) -> tuple[int, dict]:
    idx, item, client = args
    print(f"\n[{idx}] Processing...")

    system = item.get("system", "")
    initial_query = (item.get("instruction", "") + "\n" + item.get("input", "")).strip()
    output = item.get("output", "")

    think_content, code_content = extract_think_and_code(output)

    if len(think_content) < 200:
        print("  [Skip] think_content too short, using fallback.")
        return idx, build_fallback_item(item)

    user_prompt = USER_ANALYZE_TEMPLATE.format(
        initial_query=initial_query,
        think_content=think_content,
    )

    phases = call_llm_for_analysis(client, ANALYZE_PROMPT, user_prompt)
    think_pieces = None
    source = "LLM"
    if not phases:
        print("  [Heuristic fallback] LLM unavailable, trying local split...")
        phases, think_pieces = heuristic_split_think_content(think_content)
        if not phases:
            print("  [Fallback] Heuristic split also failed.")
            return idx, build_fallback_item(item)
        source = "heuristic"

    if think_pieces is None:
        think_pieces = split_think_content(think_content, phases)
    messages = build_messages(system, initial_query, phases, think_pieces, code_content)

    if validate_messages(messages, output):
        print(f"  [OK] Converted successfully with {len(phases)} phases ({source}).")
        return idx, {
            "system": system,
            "messages": messages,
            "needs_review": False,
            "phases": [p.get("phase_title", "") for p in phases]
        }
    else:
        print("  [Fallback] Validation failed.")
        return idx, build_fallback_item(item)


# ==================== 大文件处理：分块 + checkpoint ====================

def _get_checkpoint_path(output_path: str, custom_path: str | None) -> str:
    if custom_path:
        return custom_path
    base, ext = os.path.splitext(output_path)
    return f"{base}.checkpoint.json"


def _load_checkpoint(checkpoint_path: str) -> dict:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_checkpoint(checkpoint_path: str, results_map: dict):
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results_map, f, ensure_ascii=False, indent=2)


def _split_output(results: list[dict], output_path: str, num_parts: int) -> list[str]:
    """将结果拆分为 num_parts 个文件，返回保存的文件路径列表。"""
    if num_parts <= 1:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return [output_path]

    total = len(results)
    per_file = math.ceil(total / num_parts)
    base, ext = os.path.splitext(output_path)
    saved_paths = []
    for i in range(num_parts):
        start = i * per_file
        end = min((i + 1) * per_file, total)
        if start >= total:
            break
        part_path = f"{base}.part{i + 1:03d}{ext}"
        os.makedirs(os.path.dirname(part_path) or ".", exist_ok=True)
        with open(part_path, "w", encoding="utf-8") as f:
            json.dump(results[start:end], f, ensure_ascii=False, indent=2)
        saved_paths.append(part_path)
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="单轮数据集 -> 多轮对话数据集 转换脚本")
    parser.add_argument("--input_path", default=DEFAULT_INPUT_PATH, help="输入的单轮数据集 JSON 文件路径")
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH, help="输出的多轮数据集 JSON 文件路径")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="每处理多少条数据保存一次 checkpoint（默认 500）")
    parser.add_argument("--resume", action="store_true", help="如果存在 checkpoint，则从上次进度继续处理")
    parser.add_argument("--checkpoint_path", default=None, help="checkpoint 文件路径（默认与 output_path 同名加 .checkpoint.json）")
    parser.add_argument("--num_parts", type=int, default=1, help="将最终结果拆分为多少个文件输出（默认 1，不拆分）")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    chunk_size = args.chunk_size
    resume = args.resume
    checkpoint_path = _get_checkpoint_path(output_path, args.checkpoint_path)
    num_parts = args.num_parts

    if not API_KEY:
        print("错误：未找到 API Key。请设置环境变量 ANTHROPIC_API_KEY 或 ANTHROPIC_AUTH_TOKEN")
        return

    client = create_client()

    print(f"Reading from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if LIMIT:
        dataset = dataset[:LIMIT]

    total_items = len(dataset)
    print(f"Total items to process: {total_items}, max_workers={MAX_WORKERS}, chunk_size={chunk_size}")

    # 加载 checkpoint
    checkpoint_data = _load_checkpoint(checkpoint_path) if resume else {}
    results_map = {int(k): v for k, v in checkpoint_data.items()}
    processed_count = len(results_map)
    if processed_count > 0:
        print(f"Resuming from checkpoint: {processed_count}/{total_items} items already processed.")

    num_chunks = math.ceil(total_items / chunk_size)
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, total_items)
        chunk_indices = list(range(chunk_start + 1, chunk_end + 1))

        # 过滤掉已经处理过的
        pending_indices = [idx for idx in chunk_indices if idx not in results_map]
        if not pending_indices:
            print(f"[Chunk {chunk_idx + 1}/{num_chunks}] items {chunk_start + 1}-{chunk_end} already done, skipping.")
            continue

        print(f"\n[Chunk {chunk_idx + 1}/{num_chunks}] Processing items {min(pending_indices)}-{max(pending_indices)}...")
        pending_items = [(idx, dataset[idx - 1], client) for idx in pending_indices]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_one_item, item): item[0] for item in pending_items}
            for future in as_completed(futures):
                idx, result = future.result()
                results_map[idx] = result

        # 每完成一个 chunk 就保存 checkpoint
        _save_checkpoint(checkpoint_path, results_map)
        print(f"[Chunk {chunk_idx + 1}/{num_chunks}] Checkpoint saved. Progress: {len(results_map)}/{total_items}")

    # 组装最终结果
    results = [results_map[i + 1] for i in range(total_items)]
    success_count = sum(1 for r in results if not r.get("needs_review", True))
    fallback_count = len(results) - success_count

    print(f"\nDone. Success: {success_count}, Fallback: {fallback_count}")

    saved_paths = _split_output(results, output_path, num_parts)
    for p in saved_paths:
        print(f"Saved to {p}")

    # 删除 checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Cleaned up checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
