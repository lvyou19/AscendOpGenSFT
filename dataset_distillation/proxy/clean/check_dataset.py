#!/usr/bin/env python3
"""第三步：对导出的训练数据做结构规则校验 + 二元分流。

针对 export_training.py 产出的 *.raw.json 做检查并按结果分流到 pass/ 和 fail/。

设计原则：脚本只做能 100% 判定的结构规则；不能 100% 判定的语义信号
只扫描并回显到报告里（标 needs_manual_review），让用户自行决定是否手动剔除。

硬规则（违反即 FAIL，100% 可判定）：
    R1  messages 至少 3 条
    R2  第一条 role == "system"，且 content 非空
    R3  第二条 role == "user"
    R4  user 之后必须紧跟 assistant（不允许连续 user、user→tool、user 结尾）
    R5  不允许连续两条 assistant
    R6  最后一条 role == "assistant"
    R7  第一条 system 的 content 不能是 compaction marker
        （[--- context compaction boundary (Claude Code /compact) ---]）
    R9  同算子分组级别：每个算子必须有且只有 1 个 subagent 文件
        （文件名 <base>__sub[N]_<type>.raw.json）。多了（多次委派，可能首次失败
        重试）或少了（没委派或失败没产出）都判失败，整组（主+子）→ fail/。

注：用例是否实际通过不由文本规则判定——由 R-eval（batch_evaluate.py 实际跑
evaluate_ascendc.sh 重编译 + verification）唯一决定。eval_report.json 里 status
非 PASS 都强制判 FAIL（含 NO_TEST/未记录）。

软规则（命中给 WARN，标 needs_manual_review，样本仍走 pass/）：
    这些规则的关键词可能出现在合法上下文里（如"修复了 error"），所以
    不能 100% 判定失败。命中后只在报告里回显。
    S1  最后一条 assistant 命中失败关键词（traceback/exception/exit code 等）
    S2  任意 tool result 命中 token / context 超限信号（max_tokens/context length 等）
    S3  任意 Agent/Task tool_call 对应的 tool result 命中 subagent 失败信号

分流（--split <dir>）：
    PASS（硬规则全过）→ mv 到 <dir>/pass/
    FAIL（任意硬规则违反）→ mv 到 <dir>/fail/

用法：
    # 只检查
    python3 scripts/check_dataset.py work/cleaned/ -r
    # 检查 + 分流（推荐）
    python3 scripts/check_dataset.py work/cleaned/ -r --split work/cleaned -o work/check_report.json
    # 关闭软规则（只跑硬规则）
    python3 scripts/check_dataset.py work/cleaned/ -r --no-heuristic --split work/cleaned
    # 试运行（不实际 move 文件）
    python3 scripts/check_dataset.py work/cleaned/ -r --split work/cleaned --dry-run

退出码：所有文件全 PASS = 0；任意文件 FAIL = 1。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

COMPACTION_MARKER = "[--- context compaction boundary (Claude Code /compact) ---]"

# S1: 最后一条 assistant 的失败信号（强信号，但仍可能误伤——"修复了 traceback"）
FAILURE_HINTS = re.compile(
    r"(?:traceback|exception\s+occurred|fatal\s+error|"
    r"<error>|<tool_use_error>|exit code\s*[1-9]\d*|"
    r"errno\s*[1-9]\d*|command not found)",
    re.I,
)

# S2: token / context 超限信号
TOKEN_LIMIT_HINTS = re.compile(
    r"(?:max_tokens|maximum\s+(?:context|tokens?)|context\s+length|"
    r"prompt\s+too\s+long|输入过长|超过\s*(?:token|上下文)|"
    r"context\s+window|exceeds?\s+(?:the\s+)?(?:token|context))",
    re.I,
)

# S3: subagent 调用 result 的失败信号
SUBAGENT_FAIL_HINTS = re.compile(
    r"(?:agent\s+(?:failed|error)|task\s+failed|"
    r"sub-?agent\s+(?:returned?\s+no|did\s+not\s+(?:return|complete))|"
    r"<tool_use_error>|no\s+result\s+(?:from|returned))",
    re.I,
)

SUBAGENT_TOOL_NAMES = {"Agent", "Task"}


def _flatten_text(value: Any) -> str:
    """content / tool_calls 里取可用于关键词扫描的纯文本。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for blk in value:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict):
                t = blk.get("text") or blk.get("thinking") or ""
                if isinstance(t, str):
                    parts.append(t)
                args = (blk.get("function") or {}).get("arguments")
                if isinstance(args, str):
                    parts.append(args)
        return "\n".join(parts)
    return str(value)


def _scan_soft_signals(messages: List[Dict[str, Any]]) -> List[str]:
    """扫 S1-S3 软规则，返回 warnings 列表（每条带 S 编号前缀）。"""
    warns: List[str] = []
    if not messages:
        return warns

    # S1: 最后一条 assistant 命中失败关键词
    last = messages[-1] if isinstance(messages[-1], dict) else {}
    if last.get("role") == "assistant":
        text = _flatten_text(last.get("content"))
        for tc in last.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                text += "\n" + _flatten_text(fn.get("arguments"))
        m = FAILURE_HINTS.search(text)
        if m:
            warns.append(f"S1 最后一条 assistant 命中失败关键词 {m.group(0)!r}")

    # S2 / S3: 扫所有 tool result
    # 先把 subagent 的 tool_call_id 收集起来，之后看到对应 role=tool 消息时
    # 额外用 SUBAGENT_FAIL_HINTS 扫一遍。
    pending_subagent_call_ids: Set[str] = set()
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                if fn.get("name") in SUBAGENT_TOOL_NAMES:
                    cid = tc.get("id")
                    if cid:
                        pending_subagent_call_ids.add(cid)
        elif role == "tool":
            content_text = _flatten_text(msg.get("content"))
            # S2: 任何 tool result 命中 token 超限信号
            mt = TOKEN_LIMIT_HINTS.search(content_text)
            if mt:
                warns.append(f"S2 tool result 命中 token 超限信号 {mt.group(0)!r}（位置 {i}）")
            # S3: subagent 调用的 result 命中失败信号
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id and tool_call_id in pending_subagent_call_ids:
                ms = SUBAGENT_FAIL_HINTS.search(content_text)
                if ms:
                    warns.append(
                        f"S3 subagent 调用 result 命中失败信号 {ms.group(0)!r}（位置 {i}）"
                    )
                pending_subagent_call_ids.discard(tool_call_id)

    return warns



def check_sample(sample: Dict[str, Any], use_heuristic: bool
                 ) -> Tuple[List[str], List[str]]:
    """对一条样本返回 (errors, warnings)。

    errors   = 硬规则违反（R1-R7 结构），命中任意一条 → FAIL
    warnings = 软规则命中（S1-S3），样本仍 PASS 但标 needs_manual_review

    注：用例是否实际通过由 R-eval（batch_evaluate.py 评测脚本）判定，
    不再靠 assistant content 关键字识别。
    """
    errors: List[str] = []
    warnings: List[str] = []

    messages = sample.get("messages")
    if not isinstance(messages, list):
        return [f"messages 字段缺失或不是 list（实际类型 {type(messages).__name__}）"], []

    n = len(messages)
    if n < 3:
        return [f"R1 消息数 {n} < 3，无法构成最小训练样本（system+user+assistant）"], []

    # R2 第一条必须是 system 且非空 / R7 不能是 compaction marker
    first = messages[0] if isinstance(messages[0], dict) else {}
    if first.get("role") != "system":
        errors.append(f"R2 第一条 role={first.get('role')!r}，应为 'system'")
    else:
        content = _flatten_text(first.get("content")).strip()
        if not content:
            errors.append("R2 第一条 system 的 content 为空")
        elif COMPACTION_MARKER in content:
            errors.append("R7 第一条 system 的 content 是 compaction marker（/compact 残留）")

    # R3 第二条必须是 user
    second = messages[1] if isinstance(messages[1], dict) else {}
    if second.get("role") != "user":
        errors.append(f"R3 第二条 role={second.get('role')!r}，应为 'user'")

    # R4 user 之后必须紧跟 assistant / R5 不允许连续 assistant
    for i in range(n - 1):
        cur = messages[i] if isinstance(messages[i], dict) else {}
        nxt = messages[i + 1] if isinstance(messages[i + 1], dict) else {}
        cur_role = cur.get("role")
        nxt_role = nxt.get("role")
        if cur_role == "user" and nxt_role != "assistant":
            errors.append(
                f"R4 位置 {i} 是 user，但 {i + 1} 是 {nxt_role!r}（应为 assistant）"
            )
        if cur_role == "assistant" and nxt_role == "assistant":
            errors.append(f"R5 位置 {i}/{i + 1} 出现连续两条 assistant")

    # R6 最后一条必须是 assistant
    last = messages[-1] if isinstance(messages[-1], dict) else {}
    if last.get("role") != "assistant":
        errors.append(f"R6 最后一条 role={last.get('role')!r}，应为 'assistant'")


    # 软规则
    if use_heuristic:
        warnings.extend(_scan_soft_signals(messages))

    return errors, warnings


def _extract_base(filename: str) -> str:
    """从样本文件名抽算子 base name。

    export_training.py 的实际命名（见其 1404 行）：
      单 sub:  <base>__sub_<type>.raw.json      （suffix='sub'，无数字）
      多 sub:  <base>__subN_<type>.raw.json     （N=1,2,...）

    所以：
      0_add.raw.json                       → '0_add'
      0_add__sub_explore.raw.json          → '0_add'   ← 单子，无数字
      0_add__sub1_explore.raw.json         → '0_add'   ← 多子第 1 个
      0_add__sub10_plan.raw.json           → '0_add'
    其他形态 → 文件 stem（独立成组，不影响别人）。
    """
    # 先剥 .raw.json / .clean.json 后缀
    m = re.match(r'^(.+?)\.(?:raw|clean)\.json$', filename)
    if not m:
        return Path(filename).stem
    stem = m.group(1)
    # 切掉 __sub<可选数字>_ 后缀
    sep = re.search(r'__sub\d*_', stem)
    return stem[:sep.start()] if sep else stem



def _apply_eval_override(
    report_passed: List[Dict[str, Any]],
    report_failed: List[Dict[str, Any]],
    eval_report: Dict[str, Any],
) -> Tuple[int, int]:
    """根据 batch_evaluate.py 产出的报告判定（用例是否通过由评测脚本唯一决定）。

    用户规则：pass/ 算子必须是脚本验证通过的。
      - eval 里 status=PASS → 不影响
      - eval 里 status=FAIL/ERROR/TIMEOUT → 强制整组 FAIL
      - eval 里 status=NO_TEST 或该算子没记录 → 强制整组 FAIL（没评测就是没验证）

    返回：(overridden, blocked) 两个计数。
      overridden：原本 PASS 被推翻为 FAIL 的样本数
      blocked：没评测记录（NO_TEST/缺失）被强制 FAIL 的样本数
    """
    if not eval_report or "ops" not in eval_report:
        return 0, 0
    overridden = 0
    blocked = 0
    # 收集所有需要判定的 base（即所有出现过的算子组）
    all_bases = set()
    for e in report_passed + report_failed:
        all_bases.add(_extract_base(Path(e["source"]).name))

    for base in all_bases:
        op_eval = eval_report["ops"].get(base)
        if op_eval is None:
            # 该算子没评测记录 → 强制 FAIL
            verdict = "NO_TEST"
            detail = "（无评测记录）"
        else:
            verdict = op_eval.get("status", "NO_TEST")
            if verdict == "PASS":
                continue  # PASS 不影响
            if verdict == "FAIL":
                p = op_eval.get("passed", 0)
                f = op_eval.get("failed", 0)
                detail = f"（实际跑测试 {p}/{p + f} passed）"
            elif verdict == "ERROR":
                snippet = (op_eval.get("error_snippet", "") or "")[:80].replace("\n", " ")
                detail = f"（实际跑测试崩溃: {snippet}）"
            elif verdict == "TIMEOUT":
                detail = f"（实际跑测试超时 {op_eval.get('duration_s', 0)}s）"
            else:  # NO_TEST
                detail = f"（{op_eval.get('error_snippet', '无 test 脚本')[:60]}）"

        # 该 base 下所有 PASS 标 FAIL
        for e in list(report_passed):
            if _extract_base(Path(e["source"]).name) != base:
                continue
            e["status"] = "FAIL"
            e["errors"].append(
                f"R-eval 实际评测 {verdict}{detail}，强制判失败"
                f"（pass/ 必须是脚本验证通过的）"
            )
            e["needs_manual_review"] = False
            report_passed.remove(e)
            report_failed.append(e)
            if verdict == "NO_TEST":
                blocked += 1
            else:
                overridden += 1
    return overridden, blocked


def _apply_grouping(
    report_passed: List[Dict[str, Any]],
    report_failed: List[Dict[str, Any]],
    eval_report: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, int, int]:
    """同 base 的样本作为一组，原地修改 entries：
       - R9: subagent 数量必须正好是 1（多了少了都 FAIL）
       - R-eval: 实际评测非 PASS（FAIL/ERROR/TIMEOUT/NO_TEST）→ 整组 FAIL
       - 任意成员 FAIL → 整组 FAIL（含从 passed 移到 failed）
       - 任意成员 needs_manual_review → 整组 needs_manual_review

    返回：(r9_demoted, eval_overridden, eval_blocked, group_demoted) 四种计数。
    """
    r9_demoted = 0
    eval_overridden = 0
    eval_blocked = 0
    group_demoted = 0

    all_entries = report_passed + report_failed
    base_to_entries: Dict[str, List[Dict[str, Any]]] = {}
    for e in all_entries:
        b = _extract_base(Path(e["source"]).name)
        base_to_entries.setdefault(b, []).append(e)

    # 1. R9: subagent 数量必须正好是 1（用户规则：有且只有 1 个 subagent）
    for base, entries in base_to_entries.items():
        sub_count = sum(1 for e in entries
                        if '__sub' in Path(e["source"]).stem)
        if sub_count == 1:
            continue
        for e in entries:
            if e["status"] != "PASS":
                continue
            e["status"] = "FAIL"
            e["errors"].append(
                f"R9 算子 {base!r} 的 subagent 数量为 {sub_count}，应为 1"
            )
            e["needs_manual_review"] = False
            report_passed.remove(e)
            report_failed.append(e)
            r9_demoted += 1

    # 1.5 R-eval: 实际评测非 PASS（FAIL/ERROR/TIMEOUT/NO_TEST/缺失）→ 整组 FAIL
    if eval_report:
        eval_overridden, eval_blocked = _apply_eval_override(
            report_passed, report_failed, eval_report
        )

    # 3. R-group: 任意成员 FAIL → 整组 FAIL（含从 passed 移到 failed）
    for b, entries in base_to_entries.items():
        fail_names = [Path(m["source"]).name for m in entries
                      if m["status"] == "FAIL"]
        if not fail_names:
            continue
        for e in entries:
            if e["status"] != "PASS":
                continue
            e["status"] = "FAIL"
            e["errors"].append(
                f"R-group 同算子组 {b!r} 内有成员 FAIL"
                f"（{', '.join(fail_names)}），整组判失败"
            )
            e["needs_manual_review"] = False
            report_passed.remove(e)
            report_failed.append(e)
            group_demoted += 1

    # 4. needs_manual_review 传播（只在剩下的 PASS 成员间）
    base_to_pass: Dict[str, List[Dict[str, Any]]] = {}
    for e in report_passed:
        b = _extract_base(Path(e["source"]).name)
        base_to_pass.setdefault(b, []).append(e)

    for b, entries in base_to_pass.items():
        review_members = [e for e in entries if e.get("needs_manual_review")]
        clean_members = [e for e in entries if not e.get("needs_manual_review")]
        if not review_members or not clean_members:
            continue
        review_summary = []
        for m in review_members:
            fname = Path(m["source"]).name
            for w in m["warnings"]:
                review_summary.append(f"{fname}: {w}")
        reason = (f"S-group 同算子组 {b!r} 内有可疑成员"
                  f"（{'；'.join(review_summary)}）")
        for e in clean_members:
            e["warnings"].append(reason)
            e["needs_manual_review"] = True

    return r9_demoted, eval_overridden, eval_blocked, group_demoted


def load_samples(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """加载一个文件返回 [(来源标识, sample), ...]。

    支持两种形态：
      - 单个 dict（带 messages 字段）
      - list[dict] / JSONL：每个元素是一条样本
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        samples = []
        with path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{ln} JSON 解析失败：{e}")
                samples.append((f"{path.name}#L{ln}", obj))
        return samples

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} JSON 解析失败：{e}")

    if isinstance(data, dict):
        return [(path.name, data)]
    if isinstance(data, list):
        return [(f"{path.name}[{i}]", s) for i, s in enumerate(data)]
    raise ValueError(f"{path} 顶层既不是 dict 也不是 list")


def _is_under(path: Path, ancestor: Path) -> bool:
    """path 是否位于 ancestor 目录下（含多层子目录）。"""
    try:
        path.resolve().relative_to(ancestor.resolve())
        return True
    except ValueError:
        return False


def collect_files(targets: List[str], recursive: bool,
                  exclude_dirs: List[Path] = None) -> List[Path]:
    """收集所有待检查的文件。exclude_dirs 下的文件会被跳过。"""
    excludes = exclude_dirs or []
    files: List[Path] = []
    for t in targets:
        p = Path(t)
        if not p.exists():
            print(f"[warn] 跳过不存在的路径：{p}", file=sys.stderr)
            continue
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            candidates = sorted(p.rglob("*.json")) if recursive else sorted(p.glob("*.json"))
            candidates += sorted(p.rglob("*.jsonl")) if recursive else sorted(p.glob("*.jsonl"))
            for c in candidates:
                if any(_is_under(c, ex) for ex in excludes):
                    continue
                files.append(c)
        else:
            print(f"[warn] 跳过非文件非目录的路径：{p}", file=sys.stderr)
    return sorted(set(files))


def _split_sample(src_file: Path, split_dir: Path, status: str, dry_run: bool) -> Path:
    """把样本文件 move 到 split_dir/pass/ 或 split_dir/fail/。返回目标路径。"""
    subdir = split_dir / ("pass" if status == "PASS" else "fail")
    subdir.mkdir(parents=True, exist_ok=True)
    dst = subdir / src_file.name
    if dry_run:
        print(f"[dry] mv {src_file} -> {dst}")
    else:
        if dst.exists():
            print(f"[warn] 目标已存在，覆盖：{dst}", file=sys.stderr)
            dst.unlink()
        shutil.move(str(src_file), str(dst))
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("inputs", nargs="+", help="待检查的 .json / .jsonl 文件或目录")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="对目录递归扫描 *.json / *.jsonl")
    ap.add_argument("--no-heuristic", dest="heuristic", action="store_false",
                    help="关闭软规则（S 系列扫描），只跑硬规则")
    ap.add_argument("--no-group-by-base", dest="group_by_base",
                    action="store_false",
                    help="关闭同算子分组判定（默认开）。默认主 agent 和它的 "
                         "sub agent（<base>__sub<i>_<type>.raw.json）作为一组："
                         "组内任意 FAIL 整组 FAIL、任意 needs_manual_review "
                         "整组标。加这个参数后每个文件独立判定。")
    ap.add_argument("--split", metavar="DIR",
                    help="检测完后把样本文件 mv 到 DIR/pass/（PASS）或 DIR/fail/（FAIL）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印会做什么，不实际 move（与 --split 配合）")
    ap.add_argument("--eval-report", metavar="JSON",
                    help="batch_evaluate.py 产出的 JSON 报告路径。"
                         "强制规则：pass/ 必须是脚本验证通过的。eval 里 status=PASS 才放行，"
                         "FAIL/ERROR/TIMEOUT/NO_TEST/未记录都强制判 FAIL。")
    ap.add_argument("-o", "--output", help="把详细报告写入该 JSON 文件")
    ap.add_argument("--fail-fast", action="store_true",
                    help="遇到第一条 ERROR 立即终止（默认会扫完所有样本）")
    ap.set_defaults(heuristic=True, group_by_base=True)
    args = ap.parse_args()

    split_dir = Path(args.split) if args.split else None
    # 输入和 split 同目录时，跳过前次分流产生的 pass/ fail/ 子目录
    exclude_dirs: List[Path] = []
    if split_dir:
        exclude_dirs = [split_dir / "pass", split_dir / "fail"]

    files = collect_files(args.inputs, args.recursive, exclude_dirs)
    if not files:
        print("[check] 没有可检查的文件", file=sys.stderr)
        return 2

    report_passed: List[Dict[str, Any]] = []
    report_failed: List[Dict[str, Any]] = []
    n_pass = n_fail = n_warn = 0
    needs_review_entries: List[Dict[str, Any]] = []

    for f in files:
        try:
            samples = load_samples(f)
        except ValueError as e:
            print(f"[FAIL] {f}: {e}", file=sys.stderr)
            report_failed.append({"file": str(f), "source": str(f),
                                  "status": "FAIL",
                                  "errors": [str(e)], "warnings": [],
                                  "needs_manual_review": False})
            n_fail += 1
            if args.fail_fast:
                break
            continue

        for src, sample in samples:
            errors, warnings = check_sample(sample, args.heuristic)
            needs_review = bool(warnings)
            if errors:
                status = "FAIL"
                n_fail += 1
            else:
                status = "PASS"
                n_pass += 1
                if warnings:
                    n_warn += 1
            entry = {
                "file": str(f),
                "source": src,
                "status": status,
                "errors": errors,
                "warnings": warnings,
                "needs_manual_review": needs_review,
            }
            if errors:
                report_failed.append(entry)
                print(f"[FAIL] {src}")
                for e in errors:
                    print(f"        - {e}")
            else:
                report_passed.append(entry)
                if warnings:
                    print(f"[WARN] {src}")
                    for w in warnings:
                        print(f"        - {w}")
                    needs_review_entries.append(entry)
                else:
                    print(f"[ OK ] {src}")
            if errors and args.fail_fast:
                break

    # 同算子分组判定：默认开，把同 base 的主+子 agent 作为一组
    # 读 eval 报告（可选）
    eval_report_data: Optional[Dict[str, Any]] = None
    if args.eval_report:
        eval_path = Path(args.eval_report)
        if not eval_path.is_file():
            print(f"[warn] eval 报告不存在: {eval_path}（跳过 R-eval）", file=sys.stderr)
        else:
            try:
                eval_report_data = json.loads(eval_path.read_text(encoding="utf-8"))
                print(f"[check] 读入 eval 报告: {eval_path}")
            except json.JSONDecodeError as e:
                print(f"[warn] eval 报告 JSON 解析失败: {e}（跳过 R-eval）", file=sys.stderr)

    if args.group_by_base:
        r9_demoted, eval_overridden, eval_blocked, group_demoted = _apply_grouping(
            report_passed, report_failed, eval_report_data
        )
        # 重算计数（_apply_grouping 可能改变了 report_passed/failed 的成员）
        n_pass = len(report_passed)
        n_fail = len(report_failed)
        n_warn = sum(1 for e in report_passed if e.get("needs_manual_review"))
        needs_review_entries = [e for e in report_passed if e.get("needs_manual_review")]
        if r9_demoted > 0:
            print(f"[check] R9 subagent 数量异常：{r9_demoted} 个样本被降级为 FAIL")
        if eval_overridden > 0:
            print(f"[check] R-eval 实际评测失败：{eval_overridden} 个样本推翻 R8 判定为 FAIL")
        if eval_blocked > 0:
            print(f"[check] R-eval 未评测算子：{eval_blocked} 个样本强制判 FAIL（pass/ 必须是脚本验证过的）")
        if group_demoted > 0:
            print(f"[check] 同算子分组：{group_demoted} 个 PASS 因组内有 FAIL 被降级为 FAIL")

    # 分流：用 set 去重文件路径（一文件多样本时只 move 一次）
    if split_dir:
        files_to_pass: Set[Path] = {Path(e["file"]) for e in report_passed}
        files_to_fail: Set[Path] = {Path(e["file"]) for e in report_failed}
        for f in files_to_pass:
            _split_sample(f, split_dir, "PASS", args.dry_run)
        for f in files_to_fail:
            _split_sample(f, split_dir, "FAIL", args.dry_run)
        # 更新报告里的 file 字段指向新路径
        pass_dir = split_dir / "pass"
        fail_dir = split_dir / "fail"
        for e in report_passed:
            e["file"] = str(pass_dir / Path(e["file"]).name)
        for e in report_failed:
            e["file"] = str(fail_dir / Path(e["file"]).name)
        for e in needs_review_entries:
            e["file"] = str(pass_dir / Path(e["file"]).name)

    print()
    print(f"[check] 总结：PASS={n_pass}  FAIL={n_fail}（其中带 WARN 的 PASS={n_warn}）")
    if split_dir:
        tag = "（dry-run，未实际 move）" if args.dry_run else ""
        print(f"[check] 已分流：pass/ {n_pass} 个，fail/ {n_fail} 个{tag}")
    if needs_review_entries:
        print(f"[check] ⚠ 需人工复查的 PASS 样本（{len(needs_review_entries)} 个）:")
        for e in needs_review_entries:
            print(f"  - {e['file']}  ← {'; '.join(e['warnings'])}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 报告里没有内部字段需要清理（_last_text 已废弃）
        def _strip(entry):
            return entry
        out_path.write_text(
            json.dumps(
                {"summary": {"pass": n_pass, "fail": n_fail, "warn_pass": n_warn,
                             "needs_manual_review": len(needs_review_entries)},
                 "passed": [_strip(e) for e in report_passed],
                 "failed": [_strip(e) for e in report_failed]},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[check] 详细报告 -> {out_path}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
