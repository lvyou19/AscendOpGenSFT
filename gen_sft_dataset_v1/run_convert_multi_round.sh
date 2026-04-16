#!/bin/bash
# 调度 dataset-to-multi-round 转换任务
# 默认自动同意所有权限，适合无人值守运行
#
# 用法:
#   bash run_convert_multi_round.sh --input-path /path/to/dataset.json --output-path /path/to/output.json

set -euo pipefail

# ── 默认值 ──
INPUT_PATH="/home/l00899543/gen_sft_datasets/dataset.json"
OUTPUT_PATH="/home/l00899543/gen_sft_datasets/dataset_multi_round.json"
PROJECT_DIR="/home/l00899543/gen_sft_datasets"

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-path)   INPUT_PATH="$2"; shift 2 ;;
        --output-path)  OUTPUT_PATH="$2"; shift 2 ;;
        -h|--help)
            echo "用法: bash run_convert_multi_round.sh --input-path <path> --output-path <path>"
            echo ""
            echo "参数:"
            echo "  --input-path   输入的单轮数据集 JSON 文件路径 (默认: ${INPUT_PATH})"
            echo "  --output-path  输出的多轮数据集 JSON 文件路径 (默认: ${OUTPUT_PATH})"
            echo ""
            echo "示例:"
            echo "  bash run_convert_multi_round.sh"
            echo "  bash run_convert_multi_round.sh --input-path /path/to/r1_ops.json --output-path /path/to/r1_multi_ops.json"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 参数校验 ──
if [[ ! -f "$INPUT_PATH" ]]; then
    echo "错误: 输入文件不存在: ${INPUT_PATH}"
    exit 1
fi

# ── 创建输出目录 ──
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"

# ── 切换到项目目录，确保 Claude Code 能发现 skill ──
cd "$PROJECT_DIR"

START_TIME=$(date +%s)

# ── 生成调试日志路径 ──
DEBUG_LOG="${OUTPUT_DIR}/claude_debug_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================"
echo "启动数据集转换任务"
echo "输入: ${INPUT_PATH}"
echo "输出: ${OUTPUT_PATH}"
echo "调试日志: ${DEBUG_LOG}"
echo "================================================================"
echo ""

PROMPT="把数据集转成多轮，input_path 是 ${INPUT_PATH}，output_path 是 ${OUTPUT_PATH}"

if claude -p "$PROMPT" \
    --debug-file "$DEBUG_LOG" \
    --allowedTools 'Bash(*)' 'Read(*)' 'Write(*)' 'Edit(*)' 'Glob(*)' 'Grep(*)' 'Skill(*)'; then

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo ""
    echo "================================================================"
    echo "✅ 转换任务完成 (耗时 ${ELAPSED}s)"
    echo "输出文件: ${OUTPUT_PATH}"
    echo "================================================================"
else
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo ""
    echo "================================================================"
    echo "❌ 转换任务失败 (耗时 ${ELAPSED}s)"
    echo "================================================================"
    exit 1
fi
