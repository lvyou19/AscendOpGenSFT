# 使用 CANNBot 框架评测指导

本文档介绍使用CANNBot 进行算子生成任务的完整流程， 包括环境搭建、benchmark准备、结果统计等，CANNBot本身提供算子生成、结果评测的功能，同时会介绍针对基模和SFT之后的模型进行能力评测的方式，本质就是将CANNBot进行评测时候的模型进行更换，同时由于SFT模型和官方提供的api服务在数据格式上有不同，使用SFT模型在流程上会进行一些微调。

## 1. 环境准备

### 1.1 基础容器
一定要准备容器，容器中需要包含
- Python 3.8+
- Ascend CANN 8.0+
- PyTorch 2.0+

所有依赖和版本一定要符合上述要求


### 1.2 安装Claude Code

在容器中安装2.1.153版本的cc，注意SFT模型评测和收集数据集轨迹场景下cc版本一定要用2.1.153
为了防止cc自动更新版本，可以按照下面的流程安装

1. 下载和解压node

注意要根据自己的机器的版本和内核来安装对应的node
```
wget --no-check-certificate https://mirrors.huaweicloud.com/nodejs/v24.14.0/node-v24.14.0-linux-x64.tar.xz
tar -xf node-v24.14.0-linux-x64.tar.xz
```

配置环境变量,建议写入bashrc
export NODE_TLS_REJECT_UNAUTHORIZED=0
export PATH="/xxx/node-v24.14.0-linux-x64/bin:$PATH"


2. 安装cc
执行npm -v检查是否安装成功，安装成功后执行下面的命令

npm config set strict-ssl false
npm config set registry https://registry.npmmirror.com
npm cache clean -f


安装cc
npm install -g @anthropic-ai/claude-code@2.1.153 --verbose


3. 配置cc的代理


vi ~/.claude/settings.json
里面的内容如下,注意两个地方需要修改：
- ANTHROPIC_BASE_URL的内容需要根据后面安装步骤中claude-code-proxy的配置设置，保持一致即可
- "command": "/xxx/cannbot-skills/plugins-community/ascendc-ops-lab-developer/.claude/hooks/run-hook.cmd session-start-ascendc-ops-lab-developer"里的路径要根据自己安装的实际路径替换
```
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8083",
    "ANTHROPIC_AUTH_TOKEN": "sk-1234",
    "CLAUDE_CODE_EFFORT_LEVEL": "max",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0"
  },
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume|clear|compact",
        "hooks": [
          {
            "type": "command",
            "command": "/xxx/cannbot-skills/plugins-community/ascendc-ops-lab-developer/.claude/hooks/run-hook.cmd session-start-ascendc-ops-lab-developer",
            "async": false
          }
        ]
      }
    ]
  },
  "permissions": {
    "allow": [
      "Bash(git status)",
      "Bash(git diff:*)",
      "Bash(git log:*)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(npm install)",
      "Bash(npm run dev)",
      "Bash(npm run build)",
      "Bash(npm run test:*)",
      "Bash(pnpm install)",
      "Bash(pnpm dev)",
      "Bash(pnpm build)",
      "Bash(pnpm test:*)",
      "Bash(yarn install)",
      "Bash(yarn dev)",
      "Bash(yarn build)",
      "Bash(yarn test:*)",
      "Bash(pytest:*)",
      "Bash(python:*)",
      "Bash(pip install:*)",
      "Read(~/.zshrc)",
      "Read(~/.bashrc)",
      "Read(./package.json)",
      "Read(./tsconfig.json)",
      "Read(./pyproject.toml)",
      "Read(./README.md)",
      "Edit(./src/**)",
      "Edit(./app/**)",
      "Edit(./pages/**)",
      "Edit(./components/**)",
      "Edit(./lib/**)",
      "Edit(./tests/**)",
      "Edit(./README.md)"
    ],
    "deny": [
      "Read(./.env)",
      "Read(./.env.*)",
      "Read(./secrets/**)",
      "Read(./config/credentials.json)",
      "Read(~/.ssh/**)",
      "Read(~/.aws/**)",
      "Read(~/.config/gcloud/**)",
      "Read(~/.npmrc)",
      "Bash(curl:*)",
      "Bash(wget:*)",
      "Bash(rm -rf /)",
      "Bash(sudo:*)",
      "Bash(docker login:*)",
      "Bash(kubectl config:*)"
    ]
  },
  "cleanupPeriodDays": 20,
  "includeCoAuthoredBy": false,
  "forceLoginMethod": "console"
}

```


### 1.3安装CANNBot

执行下面的命令，保持版本一致，不要随意更换版本和分支
```
git clone https://gitcode.com/chenshushu2020/cannbot-skills.git -b br_asc_dev
git reset --hard 6cf50e29fabca3c78a7ed45d241d69291db613df
```

安装
```
cd /cannbot-skills/plugins-community/ascendc-ops-lab-developer
bash init.sh global claude  
```

然后安装tilelang-ascend
- 参考https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/README.md#method-3-compile-and-install-from-source 安装
下面的执行步骤和上面链接完全一致，可以直接使用
```
git clone --recursive https://github.com/tile-ai/tilelang-ascend.git
cd tilelang-ascend

bash install_ascend.sh
source set_env.sh

cd examples/gemm
python example_gemm.py
```
成功的话会输出Kernel Output Match!


### 1.4 安装claude-code-proxy
代码仓：
https://github.com/MarkJoson/claude-code-proxy/tree/cc-trajectory-gateway
在CANNBot平级的目录下执行下面命令

```
git clone https://github.com/MarkJoson/claude-code-proxy.git -b cc-trajectory-gateway
```

#### 配置claude-code-proxy
在文件夹中新建一个.env文件，内容如下

```
ANTHROPIC_API_KEY="sk-1234" # Needed if proxying *to* Anthropic
OPENAI_API_KEY="sk-123"

OPENAI_BASE_URL="http://80.48.5.60:8018/v1"
PREFERRED_PROVIDER="openai"

BIG_MODEL="qwen36"
SMALL_MODEL="qwen36"
```

这里需要修改：
- OPENAI_BASE_URL 是本地vllm服务的ip和端口
- BIG_MODEL和SMALL_MODEL是本地vllm服务提供的字段，保持一致

之后source .env使环境变量生效，然后执行下面的命令安装依赖

```
pip install litellm
```


然后在claude-code-proxy目录下执行下面的命令
```
uvicorn server:app --host 0.0.0.0 --port 8083 --reload
```
这里port 8083要和第一步中Claude Code的settings.json配置中的保持一致


## 2. 拉起测评

在/cannbot-skills/plugins-community/ascendc-ops-lab-developer目录下新建一个脚本run.sh
```
bash workflows/scripts/run_benchmark_ascendc.sh \
    --benchmark-dir /home/benchmarks/NPUKernelBench/ \
    --level 1 \
    --range 1-31 \
    --npu-list "10,11,12,13" \
    --output /home/
```

这里需要自己修改每一个字段，benchmark-dir是benchmark的目录，--npu-list是可用的设备，最多设置16卡，--output是测评输出的路径

然后直接bash run.sh就可以拉起测评





