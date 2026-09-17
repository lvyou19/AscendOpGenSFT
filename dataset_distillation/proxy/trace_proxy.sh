#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# trace_proxy.sh — trace proxy 启停函数库（精简版）
#
# 被 run_benchmark_ascendc.sh source 使用，提供：
#   trace_proxy_start [port]   起 proxy 并就绪探测；端口可省略（自动探测空闲端口）
#                              成功后实际端口写入 PROXY_PORT_RESOLVED（调用方回读）
#   trace_proxy_stop           进程组 kill proxy
#
# 认证配置来源（优先级：进程环境变量 > proxy/.env > 默认值）：
#   - 用户已在环境或 .env 配置 → 原样尊重（override 换钥匙 / litellm 翻译老用法均兼容）
#   - 都未配置 PREFERRED_PROVIDER → 注入 anthropic（批跑必须走 passthrough 采集路径）
#   - .env 不再是必需品：无 ANTHROPIC_API_KEY 时 server.py 自动走认证透传
#     （claude 自带的 key 原样到达上游），多窗口多 key 天然按进程隔离
#
# 设计要点：
#   - 只管启停 proxy，不碰 settings.json
#   - 用 setsid 让 proxy 独立成进程组，stop 时进程组 kill 连带 uvicorn worker 一起清
#   - 去掉 --reload（并发场景热重载会丢请求）
#   - 函数幂等，stop 多次调用安全
# ----------------------------------------------------------------------------------------------------------

# 全局状态（start 写入，stop 读取）
PROXY_PID=""
PROXY_PGID=""
PROXY_LOG=""
PROXY_TRACE_DIR=""      # 本次批跑的 trace 目录（带时间戳，避免多次批跑覆盖）
PROXY_PORT_RESOLVED=""  # 本次实际使用的端口（自动探测时由本脚本决定）

# ------------------------------------------------------------------------------
# trace_proxy_port_in_use <port>   端口被监听返回 0，空闲返回 1
# ------------------------------------------------------------------------------
trace_proxy_port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":$port "; then
        return 0
    fi
    return 1
}

# ------------------------------------------------------------------------------
# trace_proxy_pick_port   探测一个空闲端口并输出到 stdout；失败 return 1
# 优先 python3 内核级 bind 探测（比 ss 扫描更准）；无 python3 则 ss+随机段兜底
# ------------------------------------------------------------------------------
trace_proxy_pick_port() {
    if command -v python3 >/dev/null 2>&1; then
        python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()' \
            2>/dev/null && return 0
    fi
    local p _i
    for _i in 1 2 3 4 5; do
        p=$(( (RANDOM % 40000) + 20000 ))
        if ! trace_proxy_port_in_use "$p"; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

# ------------------------------------------------------------------------------
# _trace_proxy_launch <port>   真正起进程 + 就绪探测（成功 return 0）
# 依赖：PROXY_TRACE_DIR 已建好、CC_TRACE_DIR 已 export、SCRIPT_DIR 已定义
# ------------------------------------------------------------------------------
_trace_proxy_launch() {
    local port="$1"
    local proxy_dir="${SCRIPT_DIR}/proxy/claude-code-proxy"

    PROXY_LOG="/tmp/cc-proxy.$$.log"
    local orig_dir="$(pwd)"
    cd "$proxy_dir"
    setsid nohup uvicorn server:app \
        --host 0.0.0.0 --port "$port" \
        > "$PROXY_LOG" 2>&1 &
    PROXY_PID=$!
    disown "$PROXY_PID" 2>/dev/null || true   # 让 bash 放弃跟踪 proxy，避免无参数 wait 卡住
    # setsid 让 proxy 成为新会话 leader，PGID == PID；ps 兜底确认
    PROXY_PGID=$(ps -o pgid= -p "$PROXY_PID" 2>/dev/null | tr -d ' ')
    [[ -z "$PROXY_PGID" ]] && PROXY_PGID="$PROXY_PID"
    cd "$orig_dir"
    echo "[trace] proxy 后台启动 (pid=$PROXY_PID, pgid=$PROXY_PGID)"

    # 就绪探测（轮询 15s）
    local i
    for i in $(seq 1 30); do
        if ! kill -0 "$PROXY_PID" 2>/dev/null; then
            echo "[trace] 错误：proxy 进程已退出，查看日志：$PROXY_LOG" >&2
            tail -20 "$PROXY_LOG" >&2 2>/dev/null || true
            return 1
        fi
        if curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$port/" 2>/dev/null; then
            echo "[trace] proxy ready (port=$port, log=$PROXY_LOG)"
            return 0
        fi
        sleep 0.5
    done
    echo "[trace] 错误：proxy 就绪探测超时（15s），查看日志：$PROXY_LOG" >&2
    tail -20 "$PROXY_LOG" >&2 2>/dev/null || true
    return 1
}

# ------------------------------------------------------------------------------
# trace_proxy_start [port]
# 不传端口 = 自动探测空闲端口（被抢/启动失败自动换端口重试，至多 5 次）
# 传端口   = 固定端口（被占或启动失败直接报错，不跳换 —— 兼容旧行为）
# 成功 return 0 并把实际端口写入 PROXY_PORT_RESOLVED；失败 return 1
# （由调用方 exit，trap 会负责清理）
# 依赖主脚本先定义 SCRIPT_DIR（脚本所在目录）
# ------------------------------------------------------------------------------
trace_proxy_start() {
    local port="${1:-}"
    local explicit=0
    [[ -n "$port" ]] && explicit=1
    local proxy_dir="${SCRIPT_DIR}/proxy/claude-code-proxy"

    echo "[trace] === trace_proxy_start: port=${port:-auto} dir=$proxy_dir ==="

    # --- 1. 前置检查（.env 非必需：透传模式靠进程环境即可）---
    if ! command -v uvicorn >/dev/null 2>&1; then
        echo "[trace] 错误：未找到 uvicorn 命令，请先安装 fastapi/uvicorn" >&2
        return 1
    fi
    if ! command -v curl >/dev/null 2>&1; then
        echo "[trace] 错误：未找到 curl（就绪探测用）" >&2
        return 1
    fi
    if [[ ! -f "$proxy_dir/server.py" ]]; then
        echo "[trace] 错误：$proxy_dir/server.py 不存在" >&2
        return 1
    fi
    if [[ ! -f "$proxy_dir/.env" ]]; then
        echo "[trace] 提示：无 $proxy_dir/.env，proxy 仅用进程环境变量/默认值（认证透传模式无需 .env）"
    fi
    if (( explicit )) && trace_proxy_port_in_use "$port"; then
        echo "[trace] 错误：端口 $port 已被占用" >&2
        return 1
    fi

    # --- 2. 认证路径注入：批跑必须走 anthropic passthrough；用户已配置则尊重 ---
    if [[ -z "${PREFERRED_PROVIDER:-}" ]] && ! grep -qsE '^\s*PREFERRED_PROVIDER\s*=' "$proxy_dir/.env" 2>/dev/null; then
        export PREFERRED_PROVIDER=anthropic
        echo "[trace] 注入 PREFERRED_PROVIDER=anthropic（未检测到用户配置；批跑走采集透传路径）"
    fi

    # --- 3. 生成唯一的 trace 目录（每次启动独立子目录，避免覆盖）---
    # 前缀由调用方可通过 PROXY_TRACE_PREFIX 定制（批跑默认 batch，交互模式用 session）
    local timestamp=$(date +%Y%m%d_%H%M%S)
    PROXY_TRACE_DIR="${proxy_dir}/cc_traces/${PROXY_TRACE_PREFIX:-batch}_${timestamp}"
    mkdir -p "$PROXY_TRACE_DIR"
    export CC_TRACE_DIR="$PROXY_TRACE_DIR"
    echo "[trace] trace 目录: $PROXY_TRACE_DIR"

    # --- 4. 启动（自动端口被抢/失败换端口重试；显式端口不跳换）---
    local attempts=0
    while (( attempts < 5 )); do
        if [[ -z "$port" ]]; then
            if ! port="$(trace_proxy_pick_port)"; then
                echo "[trace] 错误：探测空闲端口失败" >&2
                return 1
            fi
        fi
        if trace_proxy_port_in_use "$port"; then
            if (( explicit )); then
                echo "[trace] 错误：端口 $port 已被占用" >&2
                return 1
            fi
            port=""
            attempts=$((attempts + 1))
            continue
        fi
        if _trace_proxy_launch "$port"; then
            PROXY_PORT_RESOLVED="$port"
            # 透出 server.py 的认证模式 banner（不含 key），确认 override/passthrough 生效
            grep -m1 '^\[auth\]' "$PROXY_LOG" 2>/dev/null || true
            echo "[trace] === trace_proxy_start 完成 ==="
            return 0
        fi
        if (( explicit )); then
            return 1   # 显式端口启动失败不换端口，便于看日志定位
        fi
        port=""
        attempts=$((attempts + 1))
    done
    echo "[trace] 错误：自动端口重试 5 次均失败" >&2
    return 1
}

# ------------------------------------------------------------------------------
# trace_proxy_stop
# 幂等：变量为空或进程不存在则跳过，可被 trap 多次安全调用
# ------------------------------------------------------------------------------
trace_proxy_stop() {
    # 进程组 kill proxy（连带 uvicorn worker）
    if [[ -n "$PROXY_PGID" ]]; then
        kill -- -"$PROXY_PGID" 2>/dev/null \
            || kill "$PROXY_PID" 2>/dev/null \
            || true
        echo "[trace] proxy 已停止 (pid=$PROXY_PID, pgid=$PROXY_PGID)"
        PROXY_PID=""
        PROXY_PGID=""
    fi

    echo "[trace] === trace_proxy_stop 完成 ==="
}
