#!/usr/bin/env python3
"""第五步：把清洗好的单条样本合并成一个数据集文件。

输入：一个或多个目录（递归扫描）/ 文件，里面是 export_training.py 产出的
       *.clean.json（推荐）或 *.raw.json、*.jsonl。
输出：单个 JSON 数组，形如 [{"messages": [...]}, {"messages": [...]}, ...]。
       这是规范、可直接 json.load 的形态，与 dataset/one_shot_success_merged.json
       的"逗号拼接无外层括号"形态不同——后者只是历史遗留，新流程统一用标准数组。

样本字段处理：
    默认只保留 messages（与现有 one_shot_success_merged.json 一致）；
    --keep-meta 时额外保留 meta / tools / session_id。
    可用 --strip-system 移除每条样本第一条 system（部分训练 pipeline 自带 system prompt，
    不希望重复）。

用法：
    # 默认：合并所有 *.clean.json，每条只留 messages
    python3 scripts/merge_dataset.py path/to/cleaned/ -o merged.json

    # 递归 + 只挑 *.clean.json + 保留 meta/tools
    python3 scripts/merge_dataset.py path/to/cleaned/ -r \\
        --pattern '*.clean.json' --keep-meta -o merged.json

    # 多个来源合并
    python3 scripts/merge_dataset.py pathA pathB pathC -r -o merged.json

    # 输出 JSONL（流式友好，单行一个 dict）
    python3 scripts/merge_dataset.py path/to/cleaned/ -r -o merged.jsonl --format jsonl

退出码：成功 = 0；任意样本缺 messages 字段 = 1（除非 --allow-empty）。
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def iter_files(paths: List[str], recursive: bool, pattern: str,
               exclude_patterns: Optional[List[str]] = None) -> List[Path]:
    excludes = exclude_patterns or []
    out: List[Path] = []
    for t in paths:
        p = Path(t)
        if not p.exists():
            print(f"[warn] 跳过不存在的路径：{p}", file=sys.stderr)
            continue
        if p.is_file():
            candidates = [p]
        elif p.is_dir():
            it = p.rglob(pattern) if recursive else p.glob(pattern)
            candidates = sorted(it)
        else:
            print(f"[warn] 跳过非文件非目录的路径：{p}", file=sys.stderr)
            continue
        for c in candidates:
            if any(fnmatch.fnmatch(c.name, ex) for ex in excludes):
                continue
            out.append(c)
    # 去重并按路径排序，保证合并顺序稳定、可复现
    return sorted(set(out))


def load_samples_from_file(path: Path) -> List[Dict[str, Any]]:
    """加载一个文件，返回 [sample, ...]。

    支持三种形态：
      - dict（含 messages）→ 视作单条
      - list[dict] → 多条
      - JSONL → 每行一条
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[warn] {path}:{ln} JSON 解析失败，跳过该行：{e}",
                          file=sys.stderr)
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
        return out

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[warn] {path} JSON 解析失败，跳过该文件：{e}", file=sys.stderr)
        return []

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    print(f"[warn] {path} 顶层既不是 dict 也不是 list，跳过", file=sys.stderr)
    return []


def project_sample(sample: Dict[str, Any],
                   keep_meta: bool, strip_system: bool) -> Optional[Dict[str, Any]]:
    """按规则裁剪一条样本。返回 None 表示这条样本无效（应丢弃或报错）。"""
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return None

    if strip_system and messages and (messages[0] or {}).get("role") == "system":
        messages = messages[1:]

    if keep_meta:
        out = dict(sample)
        out["messages"] = messages
        return out
    # 默认只保留 messages，与现有 one_shot_success_merged.json 对齐
    return {"messages": messages}


def write_json_array(path: Path, samples: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, samples: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False))
            f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("inputs", nargs="+", help="待合并的 .json / .jsonl 文件或目录")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="对目录递归扫描匹配的文件")
    ap.add_argument("--pattern", default="*.clean.json",
                    help="目录匹配的文件名 glob（默认 *.clean.json）")
    ap.add_argument("--exclude-pattern", action="append", default=[],
                    metavar="GLOB",
                    help="排除匹配该 glob 的文件名，可重复。"
                         "例如 --exclude-pattern '*__sub_*' 只合并 main 样本")
    ap.add_argument("-o", "--output", required=True,
                    help="合并结果输出路径")
    ap.add_argument("--format", choices=("auto", "json", "jsonl"), default="auto",
                    help="输出格式；auto 按输出文件后缀决定（默认 auto）")
    ap.add_argument("--keep-meta", action="store_true",
                    help="额外保留 meta / tools / session_id 字段（默认只保留 messages）")
    ap.add_argument("--strip-system", action="store_true",
                    help="移除每条样本第一条 system 消息（部分训练 pipeline 自带 system）")
    ap.add_argument("--allow-empty", action="store_true",
                    help="遇到 messages 缺失/类型错误的样本时丢弃并继续，而不是失败")
    args = ap.parse_args()

    files = iter_files(args.inputs, args.recursive, args.pattern,
                       args.exclude_pattern or None)
    if not files:
        print(f"[merge] 没有匹配到任何文件（pattern={args.pattern!r}）", file=sys.stderr)
        return 2

    print(f"[merge] 待处理文件数：{len(files)}")

    all_samples: List[Dict[str, Any]] = []
    n_invalid = 0
    for f in files:
        samples = load_samples_from_file(f)
        for s in samples:
            projected = project_sample(s, args.keep_meta, args.strip_system)
            if projected is None:
                n_invalid += 1
                continue
            all_samples.append(projected)
        print(f"  + {f.name}: 读入 {len(samples)} 条")

    if n_invalid > 0 and not args.allow_empty:
        print(f"[merge] 有 {n_invalid} 条样本缺 messages 或类型错误，"
              f"加 --allow-empty 可丢弃并继续，或先回到第四步修复", file=sys.stderr)
        return 1

    if not all_samples:
        print("[merge] 没有可用样本", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    fmt = args.format
    if fmt == "auto":
        fmt = "jsonl" if out_path.suffix.lower() == ".jsonl" else "json"

    if fmt == "jsonl":
        write_jsonl(out_path, all_samples)
    else:
        write_json_array(out_path, all_samples)

    total_msgs = sum(len(s.get("messages", [])) for s in all_samples)
    print()
    print(f"[merge] 输出：{out_path}")
    print(f"[merge] 样本数：{len(all_samples)}  总消息数：{total_msgs}  "
          f"格式：{fmt}")
    if n_invalid > 0:
        print(f"[merge] 已丢弃无效样本：{n_invalid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
