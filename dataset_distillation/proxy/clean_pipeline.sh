#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# clean_pipeline.sh — 一键清洗 trace.db → SFT 训练数据
#
# 用法：
#   bash clean_pipeline.sh <trace目录> <算子产出目录> <输出目录>
#
# 环境变量：
#   ASCEND_RT_VISIBLE_DEVICES  NPU 卡号（batch_evaluate 评测用）
#   EVAL_NPU_LIST              评测用 NPU 列表（如 "10,11"，按轮询分配）
#   CLEAN_JOBS                 并发数（默认 4）
#   CLEAN_TIMEOUT              单算子评测超时秒数（默认 1800）
#   SKIP_EVAL=1                跳过真机评测，check_dataset 退回 R8 文本判定
#   TRACE_SPLIT_DIR            自动拆分 trace.db 的输出目录（默认 <trace目录>/split_by_op）
# ----------------------------------------------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLEAN_SCRIPTS="${SCRIPT_DIR}/clean"

TRACE_DIR="${1:?用法: bash clean_pipeline.sh <trace目录> <算子产出目录> <输出目录>}"
OPS_DIR="${2:?缺少算子产出目录}"
OUT_DIR="${3:?缺少输出目录}"

JOBS="${CLEAN_JOBS:-4}"
# 单算子评测超时：clean build + 全量用例实测可达 400-600s，300s 会误判 TIMEOUT。
EVAL_TIMEOUT="${CLEAN_TIMEOUT:-1800}"
WORK_DIR=$(mktemp -d
)
mkdir -p "$OUT_DIR/pass" "$OUT_DIR/fail"

# ── ⓪ 自动拆分（目录里只有 trace.db 时）──
has_op_db=0
for f in "$TRACE_DIR"/*.db; do
    [[ -f "$f" ]] || continue
    [[ "$(basename "$f")" == "trace.db" ]] && continue
    has_op_db=1
    break
done
if [[ $has_op_db -eq 0 && -f "$TRACE_DIR/trace.db" ]]; then
    SPLIT_SCRIPT="${SCRIPT_DIR}/claude-code-proxy/scripts/split_trace_by_op.py"
    SPLIT_DIR="${TRACE_SPLIT_DIR:-$TRACE_DIR/split_by_op}"
    echo "[clean] 未发现 per-op db，自动拆分 trace.db -> $SPLIT_DIR"
    if [[ ! -f "$SPLIT_SCRIPT" ]]; then
        echo "[clean] 错误：缺少拆分脚本 $SPLIT_SCRIPT" >&2
        exit 1
    fi
    if ! python3 "$SPLIT_SCRIPT" --trace-db "$TRACE_DIR/trace.db" \
            --out-dir "$SPLIT_DIR" --keep-source >> "$OUT_DIR/clean.log" 2>&1; then
        echo "[clean] 错误：拆分 trace.db 失败，见 $OUT_DIR/clean.log" >&2
        exit 1
    fi
    TRACE_DIR="$SPLIT_DIR"
fi

# 统计算子数
db_count=$(ls "$TRACE_DIR"/*.db 2>/dev/null | grep -v "trace.db" | wc -l)
if [[ "$db_count" -eq 0 ]]; then
    echo "[clean] 错误：$TRACE_DIR 下没有可用的 per-op db（拆分后仍为空？）" >&2
    exit 1
fi

echo "[clean] 清洗启动（$db_count 个算子，NPU=${ASCEND_RT_VISIBLE_DEVICES:-未指定}，jobs=$JOBS）"

# ── ① export_training ──
export_ok=0
for db in "$TRACE_DIR"/*.db; do
    [[ -f "$db" ]] || continue
    [[ "$(basename "$db")" == "trace.db" ]] && continue
    op_name="$(basename "$db" .db)"
    python3 "$CLEAN_SCRIPTS/export_training.py" \
        --db "$db" --out "$WORK_DIR/raw/$op_name/" \
        --export-subagents --name-by opsname >> "$OUT_DIR/clean.log" 2>&1 \
        && export_ok=$((export_ok + 1))
done
echo "[clean] ① export_training...        ✅ $export_ok/$db_count"

# ── ② prepare_for_training ──
python3 "$CLEAN_SCRIPTS/prepare_for_training.py" "$WORK_DIR/raw/" >> "$OUT_DIR/clean.log" 2>&1
echo "[clean] ② prepare...                ✅"

# ── ③ batch_evaluate ──
# SKIP_EVAL=1 跳过真机评测（例如当前环境缺少编译算子所需的 CANN），
# 此时 ④ 不传 --eval-report，check_dataset 退回 R8 文本判定（样本未经真机验证）。
# 默认 fail-closed：batch_evaluate 失败/报告缺失时中止，绝不让“没评测”变成 pass。
eval_args=()
if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
    echo "[clean] ③ batch_evaluate...         ⏭  跳过（SKIP_EVAL=1，④ 走 R8 文本回退）"
else
    eval_script_args=()
    if [[ -n "${EVAL_ASCENDC_SH:-}" ]]; then
        eval_script_args=(--eval-script "$EVAL_ASCENDC_SH")
    fi
    eval_ok=1
    EVAL_NPU_LIST="${EVAL_NPU_LIST:-}" python3 "$CLEAN_SCRIPTS/batch_evaluate.py" \
        "$OPS_DIR" -o "$OUT_DIR/eval_report.json" \
        -j "$JOBS" --timeout "$EVAL_TIMEOUT" "${eval_script_args[@]}" >> "$OUT_DIR/clean.log" 2>&1 \
        || eval_ok=0
    if [[ $eval_ok -eq 1 ]]; then
        python3 -c "
import json,sys
d=json.load(open('$OUT_DIR/eval_report.json'))
assert isinstance(d.get('ops'), dict) and d['ops'], 'eval_report.ops 为空'
" >> "$OUT_DIR/clean.log" 2>&1 || eval_ok=0
    fi
    if [[ $eval_ok -eq 0 ]]; then
        echo "[clean] ③ batch_evaluate...         ✗ 失败或报告缺失（见 clean.log）" >&2
        if [[ "${ALLOW_R8_FALLBACK:-0}" == "1" ]]; then
            echo "[clean] ③ ALLOW_R8_FALLBACK=1：退回 R8 文本判定（样本未经真机验证）" >&2
            eval_args=()
        else
            echo "[clean] ③ 拒绝继续：pass/ 必须是真机评测通过的样本。" >&2
            echo "[clean]    如需退回文本判定，请显式设置 SKIP_EVAL=1 或 ALLOW_R8_FALLBACK=1。" >&2
            exit 1
        fi
    else
        eval_summary=$(python3 -c "
import json; d=json.load(open('$OUT_DIR/eval_report.json'))
s=d.get('summary',d) if isinstance(d,dict) else {}
print(f\"PASS={s.get('pass',0)} FAIL={s.get('fail',0)} ERROR={s.get('error',0)}\")
" 2>/dev/null || echo "读取失败")
        echo "[clean] ③ batch_evaluate...         ✅ $eval_summary"
        eval_args=(--eval-report "$OUT_DIR/eval_report.json")
    fi
fi

# ── ④ check_dataset ──
python3 "$CLEAN_SCRIPTS/check_dataset.py" \
    "$WORK_DIR/raw/" --recursive \
    --split "$OUT_DIR/" \
    "${eval_args[@]}" \
    -o "$OUT_DIR/check_report.json" >> "$OUT_DIR/clean.log" 2>&1
check_rc=$?
if [[ $check_rc -ge 2 ]]; then
    echo "[clean] ④ check_dataset...          ✗ 脚本错误（rc=$check_rc，见 clean.log）" >&2
    exit $check_rc
fi
pass_count=$(ls "$OUT_DIR/pass/"*.json 2>/dev/null | wc -l)
fail_count=$(ls "$OUT_DIR/fail/"*.json 2>/dev/null | wc -l)
echo "[clean] ④ check_dataset...          ✅ pass=$pass_count, fail=$fail_count"

# ── 汇总 ──
echo "[clean] 清洗完成：PASS=$pass_count, FAIL=$fail_count（合格数据在 $OUT_DIR/pass/）"

rm -rf "$WORK_DIR"
