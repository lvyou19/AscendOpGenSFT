# 快速上手 · 从零部署到跑通

> **一句话**：直接跑批跑脚本，自动完成 ① 批量生成算子 ② 采集轨迹 ③ 清洗成训练数据（端口自动探测，无需传参）。
> 不想采集时加 `--no-trace`。

---

## 第一步：git clone

```bash
git clone -b br_data_distribution_update https://gitcode.com/sijiaali/ascendc-kernelgen-data.git
cd ascendc-kernelgen-data
```

> **注意**：批跑脚本会调 `claude -p` 生成算子，claude 需要加载 ascendc skill。如果你的 skill 是项目级安装的（`.claude/skills/`），必须在这个项目目录下跑批跑脚本；如果是全局安装的（`~/.claude/skills/`），任何目录都行。

clone 下来后 `script/` 目录长这样：

```
script/
├── run_benchmark_ascendc.sh     ← 批跑入口（唯一入口，用户只跑这个）
└── proxy/                        ← 所有支撑脚本（不用手动跑）
    ├── trace_proxy.sh              proxy 启停（端口自动探测）
    ├── clean_pipeline.sh            一键清洗（shell 串联 5 步，不依赖 claude）
    ├── clean/                       4 个清洗 python 脚本
    │   ├── export_training.py
    │   ├── prepare_for_training.py
    │   ├── batch_evaluate.py
    │   └── check_dataset.py
    └── claude-code-proxy/           proxy 程序
        ├── server.py                   FastAPI 主程序
        ├── trace_db.py                 实时分库引擎
        ├── .env.example                配置模板（仅覆盖模式需要）
        └── static/trace.html           轨迹查看页
```

---

## 第二步：配置（按方式 A / B 二选一）

### 方式 A · 声明环境变量（推荐）

声明三个环境变量就行了（就是你平时跑 claude 的那套配置，通常已经配好）：

```bash
export ANTHROPIC_API_KEY=sk-xxx           # ① key
export ANTHROPIC_BASE_URL=https://...     # ② 上游地址
export ANTHROPIC_MODEL=glm-5.2            # ③ 模型名
```

- proxy `.env` 不需要创建；环境变量优先于 `.env` 同名项；
- key 配一处即可：批跑/换 key 用环境变量 `ANTHROPIC_API_KEY`；已配在 claude `settings.json` 里的 token 变量名是 `ANTHROPIC_AUTH_TOKEN`（配了它就什么都不用 export，proxy 自动透传）；
- 换 key 就重新声明 `ANTHROPIC_API_KEY`；多窗口各声明各的 key，天然隔离，不用改任何共享文件；
- **注意**：模型名必须是具体名字（如 `glm-5.2`），不要用 `claude-*`/sonnet/haiku 类名——那类名会被 proxy 按 `BIG_MODEL` 映射重写（默认 gpt-4.1，发到上游就错了）；想用 claude 系名可同时设 `BIG_MODEL`/`SMALL_MODEL` 指到真实模型名。

### 方式 B · 覆盖模式（原方式）

在 proxy 侧集中配 key，转发时替换客户端认证（适合"key 统一放服务端管理"的用法）：

```bash
cd script/proxy/claude-code-proxy/
cp .env.example .env
```

编辑 `.env`（示例：Kimi 上游；**只有 `ANTHROPIC_API_KEY` 必填**，其余按需）：

```bash
ANTHROPIC_API_KEY="sk-xxx"                        # ⭐ 必填：proxy 统一使用的 key（方式 B 的定义）
ANTHROPIC_BASE_URL="https://api.kimi.com/coding/" # 上游；默认官方 api，shell 已 export 同名变量时可省
PREFERRED_PROVIDER="anthropic"                    # 可省：批跑脚本未检测到配置时自动注入 anthropic
BIG_MODEL="k3"                                    # 可省：仅当 claude 用 claude 系模型名（如 claude-sonnet-*）
SMALL_MODEL="k3"                                  #       时才需要映射到真实模型；具体模型名则原样透传
```

> **同名环境变量优先于 `.env`**。`.env` 配好后，临时换 key / 多窗口分 key 不用再改文件：
>
> ```bash
> ANTHROPIC_API_KEY=sk-窗口2的key bash script/run_benchmark_ascendc.sh ...
> ```
>
> 只覆盖你 export 的那一项，`.env` 其余配置（上游、模型映射等）继续生效。

### 两种模式怎么判定

启动时 proxy 会打印一行 banner（如 `[auth] mode=passthrough (auto: ...) | upstream=...`）：

| 场景 | 生效模式 |
|------|----------|
| 配了 `ANTHROPIC_API_KEY`（环境变量或 .env，环境变量优先） | override（覆盖，=原方式） |
| 没配 `ANTHROPIC_API_KEY` | passthrough（透传 claude settings.json 里 `ANTHROPIC_AUTH_TOKEN` 的认证） |
| 想强制指定 | `export CC_AUTH_MODE=override` 或 `passthrough` |

### 其他配置

CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

claude 的 `settings.json` 不需要配 `ANTHROPIC_BASE_URL`（批跑脚本自动注入 proxy 地址，跑完即恢复直连）。

---

## 第三步：跑命令 ⭐

```bash
cd ascendc-kernelgen-data

bash script/run_benchmark_ascendc.sh \
    --benchmark-dir /path/to/NPUKernelBench \
    --level 1 \
    --ids "3,4" \
    --npu-list "4,5" \
    --output ./output/level_1
```

**参数说明**：

| 参数 | 必填 | 含义 |
|------|:---:|------|
| `--benchmark-dir` | ✅ | 算子描述文件根目录 |
| `--level` | ✅ | Level 编号（1/2/3/4） |
| `--ids "3,4"` 或 `--range 1-30` | ✅ | 哪些算子 |
| `--npu-list "4,5"` 或 `--npu 0` | ✅ | 哪些卡（多卡并行 / 单卡） |
| `--output` | ✅ | 算子产出放哪 |
| `--no-trace` | | 禁用轨迹采集（不起 proxy、不清洗，等价旧版无参行为；**默认启用**） |
| `--proxy-port=N` | | 固定 proxy 端口为 N（默认：自动探测空闲端口，多窗口无需错开） |

---

## 多窗口并行（多个 API Key）

当算子量大、一个 key 不够用时，开多个窗口各跑一部分算子——典型场景是**同一上游/模型、多个 key 分摊**（也可以每窗口不同上游）。**端口自动错开，key 各自隔离，无需改任何共享文件**：

**窗口 1**（Kimi key）：
```bash
export ANTHROPIC_API_KEY=sk-kimi
export ANTHROPIC_BASE_URL=https://api.moonshot.cn/anthropic
bash script/run_benchmark_ascendc.sh \
    --benchmark-dir /path/to/NPUKernelBench \
    --level 1 --ids "1-30" --npu-list "0,1,2,3" \
    --output ./output/level_1
```

**窗口 2**（GLM key）：
```bash
export ANTHROPIC_API_KEY=sk-glm
export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
bash script/run_benchmark_ascendc.sh \
    --benchmark-dir /path/to/NPUKernelBench \
    --level 1 --ids "31-60" --npu-list "4,5,6,7" \
    --output ./output/level_1
```

**关键点**：
- 每个窗口 export 自己的 key/上游即可，互不干扰（透传模式下 key 跟着各自进程走）
- 算子 id 不要重叠（窗口1 跑 1-30，窗口2 跑 31-60）
- `--output` 同目录或不同目录都行；同目录算子自动合并，不同目录各自独立
- 旧用法仍兼容：想固定端口传 `--proxy-port=8799`（占用即报错，不自动跳换）；想用共享 .env 的 key 走方式 B

---

## 第四步：看产出（3 类）

### ① 算子 kernel 代码 → `--output` 目录

```
output/level_1/
├── 3_Add/
│   ├── model_new_ascendc.py    ← 生成的 AscendC kernel
│   ├── design/ docs/           （设计 / 开发文档）
│   └── perf_report.md          （性能报告）
├── 4_Abs/
└── batch_report.md             （批量执行报告）
```

### ② 算子轨迹 db（每算子独立） → proxy 目录下

```
script/proxy/claude-code-proxy/cc_traces/batch_20260721_145351/
├── 3_Add.db         ← 算子 3 的完整 API 轨迹
├── 4_Abs.db         ← 算子 4 的完整 API 轨迹
└── trace.db         ← fallback（非算子请求，可忽略）
```

> 每次批跑生成独立目录（`batch_年月日_时分秒`），**重跑不覆盖**。

### ③ SFT 训练集（最终目标） → batch 目录里的 `cleaned/`

```
script/proxy/claude-code-proxy/cc_traces/batch_20260721_145351/cleaned/
├── check_report.json （检测报告：多少 PASS / FAIL）
├── eval_report.json  （NPU 评测报告）
├── pass/             ← ⭐ 通过的样本（SFT 训练数据）
└── fail/             （失败的样本）
```

---

## 跑通后的终端输出

```
[trace] proxy ready (port=51923)
[auth] mode=passthrough (auto: ANTHROPIC_API_KEY 无值) | upstream=https://... | provider=anthropic | ...
[trace] 本次采集端口: 51923
[14:53:54] 所有 worker 已启动，准备 wait
[15:35:07] [NPU 12] ✅ 算子 4: 4_Abs.py 完成 (2473s)
[15:43:47] [NPU 11] ✅ 算子 3: 3_Add.py 完成 (2993s)
[15:43:47] ✅ wait 返回
[15:43:47] 清洗使用 NPU: 11,12
[clean] 清洗启动（2 个算子，NPU=11,12，jobs=4）
[clean] ① export_training...        ✅ 2/2
[clean] ② prepare...                ✅
[clean] ③ batch_evaluate...         ✅ PASS=1 FAIL=0 ERROR=1
[clean] ④ check_dataset...          ✅ pass=0, fail=12
[clean] 清洗完成：PASS=0, FAIL=12（合格数据在 cleaned/pass/）
[15:44:07] 批跑完成
```

> 清洗的详细日志在 `cleaned/clean.log`（python 完整输出）。

---

## 跑不通怎么办

| 问题 | 原因 | 解决 |
|------|------|------|
| `proxy 启动失败` | uvicorn 没装 | `pip install uvicorn fastapi litellm` |
| `[auth] mode=...` 怎么读 | banner 显示实际生效的认证模式/上游/模型映射 | `override`=用 proxy key；`passthrough`=透传你的 token |
| 上游 401 | 哪都没配 key（环境变量/`.env` 无 `ANTHROPIC_API_KEY`，settings.json 也没配 `ANTHROPIC_AUTH_TOKEN`） | `export ANTHROPIC_API_KEY=sk-xxx` 再跑，或配 proxy `.env`，或在 settings.json 配 `ANTHROPIC_AUTH_TOKEN` |
| `CC_AUTH_MODE=...` 启动即报错 | override 没配 key / passthrough 配了非 anthropic provider | 按 banner/报错提示修，或 unset 让它自动判定 |
| 端口探测失败 | 自动重试 5 次均失败 | 极少见；可 `--proxy-port=N` 手动指定 |
| 批跑结束提示"未采集到任何算子轨迹" | claude 没走 proxy（罕见；`--settings` 注入已实测优先于环境变量和 settings.json） | 检查 claude 版本是否过旧；确认终端里 `[trace] 本次采集端口: N` 后手动 `curl http://127.0.0.1:N/` 验证 proxy 在 |
| 卡在 `准备 wait` | 某个算子还没跑完 | 正常等（看 ✅ 日志判断进度） |
| `wait 返回` 后卡住 | 清洗阶段报错 | 看 `cleaned/clean.log` |
| 清洗 0 PASS | 算子精度不达标或 JSON 格式不对 | 看 `cleaned/eval_report.json` 和 `check_report.json` |

### 不重跑算子，只重新清洗

算子轨迹 db 已经在 batch 目录里了，直接调清洗脚本：

```bash
export EVAL_NPU_LIST="4,5"
bash script/proxy/clean_pipeline.sh \
    script/proxy/claude-code-proxy/cc_traces/batch_xxxx/ \
    ./output/level_1 \
    script/proxy/claude-code-proxy/cc_traces/batch_xxxx/cleaned/
```

---

## 附：冒烟测试

跑 2 个小算子验证全流程：

```bash
bash script/run_benchmark_ascendc.sh \
    --benchmark-dir /path/to/NPUKernelBench \
    --level 1 --ids "3,4" --npu-list "4,5" \
    --output ./smoke_test
```

跑通后应该看到：算子目录 ✓ → 算子 db ✓ → `cleaned/pass/` 有合格样本 ✓。
