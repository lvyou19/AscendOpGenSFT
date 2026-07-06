#!/usr/bin/env python3
"""SFT 数据集构建四步流水线编排器。

一次拉起，顺序跑完四步；每步跑完打印报告，暂停等终端确认（y 继续 / n 中止）。
四步：
  1. 检查（2_check_dataset.py）：只检查不删除；打印不合格文件路径+数量。
     确认后把【合格数据】复制到 01_passed/（原数据不动）。
  2. 统计（stat_operators.py）：统计 01_passed/ 的算子分布，落 01_stat.txt / 01_stat.csv。
  3. 增强（3_batch_datasets_augment.py）：对 01_passed/ 做 copies 倍增强 -> 02_augmented/。
  4. 合并（4_combine_into_one_jsonl.py）：把 02_augmented/ 合并为 03_combined.jsonl。

排除所有名字（小写后）以 --exclude-suffix（默认 failed）结尾的目录。

用法：
  python3 run_pipeline.py                          # 用默认数据根，交互式
  python3 run_pipeline.py --data-root <DIR>        # 指定数据根
  python3 run_pipeline.py --copies 5               # 改增强倍数
  python3 run_pipeline.py --exclude-suffix failed --exclude-suffix falied
  python3 run_pipeline.py --yes                    # 全自动，不暂停
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CHECK = SCRIPT_DIR / "2_check_dataset.py"
STAT = SCRIPT_DIR / "stat_operators.py"
AUGMENT = SCRIPT_DIR / "3_batch_datasets_augment.py"
COMBINE = SCRIPT_DIR / "4_combine_into_one_jsonl.py"

DEFAULT_DATA_ROOT = "/home/l00899543/SFT_DATASETS/第二批"


def run(cmd, capture=False):
    """跑子进程。capture=True 时同时回显并返回 stdout 文本。"""
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    if capture:
        proc = subprocess.run([str(c) for c in cmd], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(proc.stdout, end="")
        return proc.returncode, proc.stdout
    proc = subprocess.run([str(c) for c in cmd])
    return proc.returncode, ""


def confirm(step_name, auto_yes):
    if auto_yes:
        print(f"\n[pipeline] --yes 已开启，自动继续：{step_name}")
        return True
    while True:
        ans = input(f"\n[pipeline] {step_name} 完成。继续下一步？[y/n] ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            print("[pipeline] 已按用户要求中止。")
            return False
        print("请输入 y 或 n。")


def abort_if_failed(rc, step):
    if rc != 0:
        print(f"\n[pipeline] ✗ 步骤「{step}」返回非零退出码 {rc}，流水线终止。",
              file=sys.stderr)
        sys.exit(rc)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help=f"原始数据根目录。默认 {DEFAULT_DATA_ROOT}")
    ap.add_argument("--out", default=None,
                    help="流水线输出目录。默认 <data-root>_pipeline")
    ap.add_argument("--copies", type=int, default=10,
                    help="增强倍数（保留原始 + N 倍）。默认 10")
    ap.add_argument("--exclude-suffix", action="append", default=None,
                    metavar="SUFFIX",
                    help="排除以该后缀结尾的目录，可重复。默认 ['failed']")
    ap.add_argument("--seed", type=int, default=0, help="增强随机种子")
    ap.add_argument("--yes", action="store_true", help="跳过所有确认，全自动")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        sys.exit(f"数据根目录不存在: {data_root}")
    out_dir = Path(args.out) if args.out else data_root.parent / (data_root.name + "_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude = args.exclude_suffix if args.exclude_suffix else ["failed"]
    excl_args = []
    for s in exclude:
        excl_args += ["--exclude-suffix", s]

    check_report = out_dir / "00_check_report.json"
    passed_dir = out_dir / "01_passed"
    stat_txt = out_dir / "01_stat.txt"
    stat_csv = out_dir / "01_stat.csv"
    augmented_dir = out_dir / "02_augmented"
    combined = out_dir / "03_combined.jsonl"

    print("=" * 64)
    print("SFT 数据集构建流水线")
    print(f"  数据根   : {data_root}")
    print(f"  输出目录 : {out_dir}")
    print(f"  排除后缀 : {exclude}")
    print(f"  增强倍数 : {args.copies}")
    print(f"  模式     : {'全自动(--yes)' if args.yes else '交互式确认'}")
    print("=" * 64)

    # ---------- 步骤 1a：检查（只报告，不复制） ----------
    print("\n########## 步骤 1/4：检查数据合格性（只报告，不删除） ##########")
    rc, _ = run([sys.executable, CHECK, str(data_root), "-r",
                 *excl_args, "-o", str(check_report)])
    # 注意：2_check 在「发现不合格文件」时返回 1（这是正常的、预期内的结果），
    # 只有 rc>=2（无文件可检 / 崩溃）才视为真正失败。
    if rc >= 2:
        abort_if_failed(rc, "检查")

    # 读报告，回显不合格清单 + 数量
    summary = {}
    if check_report.exists():
        rep = json.loads(check_report.read_text(encoding="utf-8"))
        summary = rep.get("summary", {})
        n_pass = summary.get("pass", 0)
        fail_files = summary.get("fail_files", [])
        print("\n[pipeline] 检查报告摘要：")
        print(f"  合格(PASS)   : {n_pass}")
        print(f"  不合格(FAIL) : {summary.get('fail_file_count', len(fail_files))}")
        print(f"  需人工复查   : {summary.get('needs_manual_review', 0)}")
        if fail_files:
            print("  不合格文件清单：")
            for ff in fail_files:
                print(f"    - {ff}")
        print(f"\n  详细报告: {check_report}")

    if not confirm("步骤1 检查（确认后将把合格数据复制到 01_passed/）", args.yes):
        sys.exit(0)

    # ---------- 步骤 1b：复制合格数据 ----------
    print("\n########## 步骤 1b：复制合格数据到 01_passed/ ##########")
    rc, _ = run([sys.executable, CHECK, str(data_root), "-r",
                 *excl_args, "--copy-passed", str(passed_dir),
                 "--copy-base", str(data_root)])
    if rc >= 2:
        abort_if_failed(rc, "复制合格数据")
    n_copied = len(list(passed_dir.rglob("*.json")))
    print(f"\n[pipeline] 已复制 {n_copied} 个合格文件到 {passed_dir}")
    if not confirm("步骤1 复制合格数据", args.yes):
        sys.exit(0)

    # ---------- 步骤 2：统计 ----------
    print("\n########## 步骤 2/4：统计算子信息 ##########")
    rc, stat_out = run([sys.executable, STAT, str(passed_dir),
                        *excl_args, "--csv", str(stat_csv)], capture=True)
    abort_if_failed(rc, "统计")
    stat_txt.write_text(stat_out, encoding="utf-8")
    print(f"\n[pipeline] 统计文本 -> {stat_txt}")
    print(f"[pipeline] 统计明细 -> {stat_csv}")
    if not confirm("步骤2 统计", args.yes):
        sys.exit(0)

    # ---------- 步骤 3：增强 ----------
    print("\n########## 步骤 3/4：数据增强 ##########")
    rc, _ = run([sys.executable, AUGMENT, str(passed_dir),
                 "--out-dir", str(augmented_dir),
                 "--copies", str(args.copies), "--seed", str(args.seed),
                 "--verify", *excl_args])
    abort_if_failed(rc, "增强")
    n_aug = len(list(augmented_dir.rglob("*.json")))
    print(f"\n[pipeline] 增强产物 {n_aug} 个 -> {augmented_dir}")
    if not confirm("步骤3 增强", args.yes):
        sys.exit(0)

    # ---------- 步骤 4：合并 ----------
    print("\n########## 步骤 4/4：合并为单个 JSONL ##########")
    rc, _ = run([sys.executable, COMBINE, str(augmented_dir), "--out", str(combined)])
    abort_if_failed(rc, "合并")

    print("\n" + "=" * 64)
    print("[pipeline] ✓ 全部完成")
    print(f"  合格数据   : {passed_dir}")
    print(f"  统计       : {stat_txt} / {stat_csv}")
    print(f"  增强产物   : {augmented_dir}")
    print(f"  训练文件   : {combined}")
    print("=" * 64)


if __name__ == "__main__":
    main()
