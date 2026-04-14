# AscendC 算子生成试错规避规则（Agent Skill 算子生成执行规范）

## 一、基础参数与环境操作规则
### 1. 参数固化与传递规则
- 核心参数（npu、op_file、output_dir）从用户输入提取并全程固化，禁止手动修改、硬编码
- npu 固定使用 0，所有设备指定操作必须对齐
- op_file / output_dir 保持完整原始路径，保留首尾 /，禁止删减、修改、简写
- 所有参数传递保持全局统一，不随意变更

### 2. 环境配置执行规则
- 统一执行命令：`export ASCEND_RT_VISIBLE_DEVICES=0 && mkdir -p {output_dir}`
- 执行顺序：Read 算子文件 → 环境配置 → Skill 调用，不可颠倒
- mkdir 必须带 -p，确保多级目录自动创建
- 禁止拆分环境命令，避免变量生效异常

## 二、工具调用规则
### 1. Read 工具调用规则
- 作为 Phase0 后第一个操作
- file_path 必须与 op_file 完全一致
- 无需额外校验格式，读取后直接进入环境配置

### 2. Bash 工具调用规则
- 仅执行“NPU 指定 + 目录创建”命令，禁止无关操作
- 报错优先检查路径、设备 ID、命令语法，不直接重复执行

### 3. 专用 Skill 调用前置规则
调用前必须完成：
1. 参数确认
2. Read 读取 op_file
3. 环境配置完成
- Skill 传入的 output_dir 必须与用户指定完全一致

## 三、流程执行规则
### 1. 阶段执行顺序（强制不可乱序）
1. Phase0：参数确认
2. Phase1：Read 读取算子文件
3. Phase2：环境配置
4. Phase3：case-simplifier
5. Phase4：tilelang-designer
6. Phase5：ascendc-translator
7. Phase6：performance-analyzer
8. Phase7：trace-recorder

### 2. 重试与异常处理规则
- 失败先排查：路径正确性、NPU ID、命令语法
- 最多重试 1 次
- 重试失败立即终止，输出明确错误原因

## 四、通用避错规则
- 不擅自修改原始参数、路径、设备号
- 不增加额外假设性校验（NPU 状态、权限检查等）
- 流程结束必须执行 trace-recorder 记录全流程信息