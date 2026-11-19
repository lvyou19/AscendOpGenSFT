#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# trace_claude.sh — 一键启动带轨迹采集的交互式 claude
#
# 用法：
#   bash script/trace_claude.sh                     # 起采集 → 进交互 claude → 退出自动收摊
#   bash script/trace_claude.sh --keep              # 退出后保留 proxy（继续看轨迹 UI）
#   bash script/trace_claude.sh -- -p "hi"          # -- 之后的参数原样透传给 claude
#   bash script/trace_claude.sh --print-base-url    # 仅打印将使用的上游地址（排查用）
#
# 设计要点：
#   - 与批跑（run_benchmark_ascendc.sh）共用同一套 proxy 启停（proxy/trace_proxy.sh）：
#     端口自动探测、认证自动判定（ANTHROPIC_API_KEY 有则覆盖 / 无则透传 claude 自带 token）、
#     PREFERRED_PROVIDER 自动注入
#   - 不改任何配置文件：settings.json（全局/项目/local）与 proxy/.env 全程只读；
#     唯一"添加"的是 --settings 注入的 claude→proxy 指向，随进程退出消失
#   - 唯一搬运的配置是上游地址（proxy 必须知道转发到哪），按优先级取：
#       shell 环境变量 ANTHROPIC_BASE_URL
#       → 项目 .claude/settings.local.json → 项目 .claude/settings.json
#       → 全局 ~/.claude/settings.json 里的 env.ANTHROPIC_BASE_URL
#     key / 模型名不查不碰：claude 发请求时自己带
#   - 轨迹落 cc_traces/session_<时间戳>/trace.db（与批跑的 batch_ 目录区分）
# ----------------------------------------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

KEEP_PROXY=0
CLAUDE_ARGS=()
PRINT_BASE_URL_ONLY=0

usage() {
    cat <<'USAGE'
用法: bash script/trace_claude.sh [--keep] [--print-base-url] [-- <claude 参数>]

  --keep             退出 claude 后保留 proxy（继续看轨迹 UI，手动 kill -- -<pgid> 停）
  --print-base-url   仅打印将使用的上游地址后退出（排查配置用）
  --                 之后的参数原样透传给 claude（如 -- -p "hi" --model glm-5.2）

配置：零新增。key/上游/模型用你已有的（settings.json 或环境变量或 proxy/.env），
认证模式自动判定，与批跑完全一致。
USAGE
}

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep)
            KEEP_PROXY=1
            shift
            ;;
        --print-base-url)
            PRINT_BASE_URL_ONLY=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        --)
            shift
            CLAUDE_ARGS=("$@")
            break
            ;;
        *)
            echo "未知参数: $1（claude 参数请放在 -- 之后）" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# ── 前置检查 ──
if ! command -v claude >/dev/null 2>&1; then
    echo "错误：未找到 claude 命令" >&2
    exit 1
fi

# ── 查真实上游地址（proxy 转发用；key/模型不查——claude 自己带）──
# 查找顺序（与 claude 的项目配置语义一致：从当前目录向上找最近的 .claude）：
#   shell 环境变量 → 逐层向上最近的 .claude/settings.local.json → settings.json
#   → 全局 ~/.claude/settings.json
resolve_base_url() {
    # 1. shell 环境变量已有 → 不动（proxy 直接继承）
    if [[ -n "${ANTHROPIC_BASE_URL:-}" ]]; then
        return 0
    fi
    command -v python3 >/dev/null 2>&1 || return 0
    local f url
    # 2. 从 $PWD 逐层向上找最近的 .claude（注意不能用 git root——claude 的项目
    #    配置根是最近的 .claude 祖先目录，与 git 仓库根不一定相同）
    local dir="$PWD" candidates=()
    while [[ -n "$dir" && "$dir" != "/" ]]; do
        if [[ -f "$dir/.claude/settings.local.json" || -f "$dir/.claude/settings.json" ]]; then
            candidates+=("$dir/.claude/settings.local.json" "$dir/.claude/settings.json")
            break
        fi
        dir="$(dirname "$dir")"
    done
    # 3. 全局兜底
    candidates+=("${HOME}/.claude/settings.json")
    for f in "${candidates[@]}"; do
        [[ -f "$f" ]] || continue
        url=$(python3 -c "
import json, sys
try:
    d = json.load(open('$f'))
    print(d.get('env', {}).get('ANTHROPIC_BASE_URL', ''))
except Exception:
    pass
" 2>/dev/null || true)
        if [[ -n "$url" ]]; then
            export ANTHROPIC_BASE_URL="$url"
            echo "[trace] 上游地址取自: $f"
            return 0
        fi
    done
    # 4. 都没有 → 不设，proxy 落到自己的 .env / 官方默认
    echo "[trace] 提示：未在任何位置找到 ANTHROPIC_BASE_URL，proxy 将使用其 .env 或官方默认地址" >&2
}

resolve_base_url

if [[ "$PRINT_BASE_URL_ONLY" == 1 ]]; then
    echo "上游地址: ${ANTHROPIC_BASE_URL:-<未设置，proxy 用 .env 或官方默认>}"
    exit 0
fi

# ── 起 proxy（与批跑同一套函数库）──
source "${SCRIPT_DIR}/proxy/trace_proxy.sh"

export PROXY_TRACE_PREFIX=session   # 交互轨迹目录与批跑的 batch_ 区分
export CC_SESSION_SPLIT=1           # 轨迹按会话拆库（sess_<短id>.db），批跑不受影响

# 清理钩子：任何退出路径都收摊（HUP 覆盖直接关终端的场景）
cleanup() {
    local rc=$?
    if [[ "$KEEP_PROXY" == 1 && -n "${PROXY_PGID:-}" ]]; then
        echo ""
        echo "[trace] proxy 保留 (pid=${PROXY_PID}, pgid=${PROXY_PGID})"
        echo "[trace] 轨迹 UI: http://127.0.0.1:${PROXY_PORT_RESOLVED:-?}/trace"
        echo "[trace] 手动停止: kill -- -${PROXY_PGID}"
    else
        trace_proxy_stop
        echo ""
        echo "[trace] 轨迹目录: ${PROXY_TRACE_DIR:-<未启动>}/"
        ls "${PROXY_TRACE_DIR:-/nonexistent}"/sess_*.db 2>/dev/null | sed 's/^/       会话库: /' || true
    fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 143' TERM

trace_proxy_start "" || {
    echo "错误：trace proxy 启动失败" >&2
    exit 1
}

PROXY_PORT="${PROXY_PORT_RESOLVED}"
echo "[trace] 实时轨迹 UI: http://127.0.0.1:${PROXY_PORT}/trace"
echo ""

# ── 前台启动交互 claude ──
# 只注入"指向 proxy"这一个配置；key/模型由 claude 按自己的配置层级合并进来
# 默认 IS_SANDBOX=1 + --dangerously-skip-permissions：免权限确认（内部算子生成环境）
# stdin 原样继承（交互场景=终端；-p/管道场景=对应输入），不做任何重定向
# 运行期间忽略 SIGINT：Ctrl-C 由 claude 前台消化（中断当前生成），脚本不自杀，
# 否则会出现"proxy 已被清理而 claude 还活着"的断链状态
trap '' INT
set +e
IS_SANDBOX=1 claude --dangerously-skip-permissions \
    --settings "{\"env\":{\"ANTHROPIC_BASE_URL\":\"http://127.0.0.1:${PROXY_PORT}\"}}" \
    ${CLAUDE_ARGS[@]+"${CLAUDE_ARGS[@]}"}
CLAUDE_RC=$?
set -e

# claude 退出 → EXIT trap 自动收摊，并透传退出码
exit "$CLAUDE_RC"
