#!/usr/bin/env python3
"""
统计算子数据集分布。

约定的数据布局：
  <root>/<来源文件夹>/<编号>_<算子名>[.raw].json            # 主 agent
  <root>/<来源文件夹>/<编号>_<算子名>__sub_<agent>[.raw].json  # 配对 subagent

每个算子 = 1 主 agent + 1 配对 subagent，算 1 条数据。
不同「来源文件夹」是不同人收集的，文件夹名字不固定 —— 脚本自动把 <root> 下的
一级子目录当作来源，无需硬编码名字。跨来源可能出现相同算子。

用法：
  python3 stat_operators.py <root>                 # 默认按语义算子名归并
  python3 stat_operators.py <root> --by prefixed   # 按 "编号_算子名" 区分（不归并）
  python3 stat_operators.py <root> --csv out.csv   # 额外导出 CSV

分类统计：自动按 <root> 下的【顶层目录】把数据分成 level_1 / level_2 / CANN 等几类，
每类单独出一份统计，最后给总计。顶层目录名即类别名。
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict


SUB_MARKER = "__sub_"


def strip_suffix(name):
    for suf in (".raw.json", ".json"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def parse_file(rel_path):
    """返回 (类别, 来源, 算子原名(含编号), 是否subagent)。

    类别  = rel_path 的【顶层目录】（如 level_1 / level_2 / CANN）。
    来源  = 类别下的一级子目录（不同人收集的文件夹）；若文件直接在类别目录下
           （如 CANN/xxx.json）则来源同类别名。
    """
    parts = rel_path.split(os.sep)
    category = parts[0] if len(parts) > 1 else "."
    if len(parts) >= 3:
        source = parts[1]          # <category>/<source>/file
    elif len(parts) == 2:
        source = parts[0]          # <category>/file（如 CANN 直接放文件）
    else:
        source = "."
    base = strip_suffix(os.path.basename(rel_path))
    is_sub = SUB_MARKER in base
    op_raw = base.split(SUB_MARKER)[0]
    return category, source, op_raw, is_sub


def semantic_name(op_raw):
    """去掉前导 '编号_' 得到语义算子名；无编号则原样返回。"""
    m = re.match(r"\d+_(.+)", op_raw)
    return m.group(1) if m else op_raw


def collect(root, exclude_suffixes=None):
    """扫描 root 下所有 json，返回 {(category, source, op_raw): {'main':n,'sub':n}}。

    exclude_suffixes: 目录名（小写后）以其中任一后缀结尾则整树跳过
                      （如 'failed' 排除 level_1_failed/）。大小写不敏感。
    """
    suffixes = [s.lower() for s in (exclude_suffixes or [])]
    stats = defaultdict(lambda: {"main": 0, "sub": 0})
    for dirpath, dirnames, filenames in os.walk(root):
        # 原地裁剪要排除的子目录，os.walk 不再下钻
        if suffixes:
            dirnames[:] = [d for d in dirnames
                           if not any(d.lower().endswith(s) for s in suffixes)]
        for fn in filenames:
            if not fn.endswith(".json"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            category, source, op_raw, is_sub = parse_file(rel)
            stats[(category, source, op_raw)]["sub" if is_sub else "main"] += 1
    return stats


def report_section(title, sub_stats, by):
    """对一个类别（或全部）的 stats 子集打印统计；sub_stats: {(source, op_raw): counts}。
    返回 (总主数, 总子数)。"""
    key_fn = semantic_name if by == "semantic" else (lambda x: x)
    by_op = defaultdict(list)  # op_key -> [(source, op_raw, counts)]
    for (source, op_raw), c in sub_stats.items():
        by_op[key_fn(op_raw)].append((source, op_raw, c))

    sources = sorted({s for (s, _) in sub_stats})
    total_main = sum(c["main"] for c in sub_stats.values())
    total_sub = sum(c["sub"] for c in sub_stats.values())

    print("=" * 70)
    print(f"【{title}】")
    print(f"  来源文件夹({len(sources)}): {', '.join(sources)}")
    print("=" * 70)

    header = f"{'算子':<24}{'条数':<6}来源明细 (来源:编号 主m/子s)"
    print(header)
    print("-" * max(len(header), 60))
    warnings = []
    for op_key in sorted(by_op):
        entries = sorted(by_op[op_key])
        n = sum(e[2]["main"] for e in entries)
        detail = "  ".join(
            f"{src}:{raw}({c['main']}m/{c['sub']}s)" for src, raw, c in entries
        )
        dup = "  <重复>" if len({e[0] for e in entries}) > 1 else ""
        print(f"{op_key:<24}{n:<6}{detail}{dup}")
        for src, raw, c in entries:
            if c["main"] != c["sub"]:
                warnings.append(f"{src}/{raw}: 主={c['main']} 子={c['sub']} 未成对")

    print()
    print(f"  算子种类数: {len(by_op)}")
    print(f"  数据集总条数(主 agent): {total_main}")
    print(f"  配对 subagent 文件数: {total_sub}")
    print(f"  数据集总数(主 agent + subagent): {total_main + total_sub}")
    dup_ops = [k for k, v in by_op.items() if len({e[0] for e in v}) > 1]
    if dup_ops:
        print(f"  跨来源重复算子({len(dup_ops)}): {', '.join(sorted(dup_ops))}")
    if warnings:
        print("  [配对告警]")
        for w in warnings:
            print("    " + w)
    print()
    return total_main, total_sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".", help="要统计的根目录")
    ap.add_argument("--by", choices=["semantic", "prefixed"], default="semantic",
                    help="算子归并方式：semantic=去编号归并(默认)，prefixed=按编号_算子名区分")
    ap.add_argument("--csv", default=None, help="导出明细 CSV 路径")
    ap.add_argument("--exclude-suffix", action="append", default=[],
                    metavar="SUFFIX",
                    help="排除名字（小写后）以该后缀结尾的目录，整树跳过。可重复。"
                         "如 --exclude-suffix failed 排除 level_1_failed/。")
    ap.add_argument("--no-category-split", dest="category_split",
                    action="store_false",
                    help="关闭按顶层目录(level_1/level_2/CANN)分类统计，只出总表。")
    ap.set_defaults(category_split=True)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"目录不存在: {args.root}")

    stats = collect(args.root, exclude_suffixes=args.exclude_suffix)
    if not stats:
        sys.exit(f"未找到 json 数据: {args.root}")

    print(f"根目录: {os.path.abspath(args.root)}")
    print(f"归并方式: {args.by}\n")

    # 分类统计：按顶层目录(category)拆分
    if args.category_split:
        by_cat = defaultdict(dict)  # category -> {(source, op_raw): counts}
        for (category, source, op_raw), c in stats.items():
            by_cat[category][(source, op_raw)] = c
        grand_main = grand_sub = 0
        for category in sorted(by_cat):
            tm, ts = report_section(category, by_cat[category], args.by)
            grand_main += tm
            grand_sub += ts
        # 全部类别合计
        print("#" * 70)
        print("【全部类别合计】")
        print(f"  类别数: {len(by_cat)}（{', '.join(sorted(by_cat))}）")
        print(f"  数据集总条数(主 agent): {grand_main}")
        print(f"  配对 subagent 文件数: {grand_sub}")
        print(f"  数据集总数(主 agent + subagent): {grand_main + grand_sub}")
        print("#" * 70)
    else:
        # 不分类：把所有数据并成一个 section（忽略 category 维度）
        flat = defaultdict(lambda: {"main": 0, "sub": 0})
        for (category, source, op_raw), c in stats.items():
            cur = flat[(source, op_raw)]
            cur["main"] += c["main"]
            cur["sub"] += c["sub"]
        report_section("全部", flat, args.by)

    # ---- CSV（始终带 category 列）----
    if args.csv:
        key_fn = semantic_name if args.by == "semantic" else (lambda x: x)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["类别", "算子(归并)", "来源", "编号_算子名", "主agent数", "subagent数"])
            for (category, source, op_raw), c in sorted(stats.items()):
                w.writerow([category, key_fn(op_raw), source, op_raw,
                            c["main"], c["sub"]])
        print(f"明细已导出: {args.csv}")


if __name__ == "__main__":
    main()
