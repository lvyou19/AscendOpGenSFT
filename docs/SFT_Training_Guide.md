# SFT 环境搭建

本文档介绍从零开始使用 MindSpeed-MM 框架对 Qwen3.6-35B-A3B 模型进行 SFT 训练

## 训练环境安装

环境建议使用镜像进行管理，可以自行创建一个基础镜像，包含Python 3.11.10

### CANN 

建议选用CANN 8.5.0 及以上商发版本，本文档选用8.5.2进行安装，注意和驱动版本配套
硬件差异化取包：以 910b/x86/Linux 机型为例，需同步下载下面三种安装包：
- 910b 基础包（Ascend-cann-910b-ops_8.5.2_linux-x86_64.run）
- nnal 组件包（Ascend-cann-nnal_8.5.2_linux-x86_64.run）
- toolkit工具链包（Ascend-cann-toolkit_8.5.2_linux-x86_64.run）
三类 Run 安装包根据实际服务器硬件芯片型号匹配对应安装资源。下载地址：[昇腾社区](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.2)
 

安装后验证：

```
python3 -c "import acl;print(acl.get_soc_name())"
```

###  MindSpeed-MM安装说明

执行下面的安装命令

```
git clone https://gitcode.com/Liziqi77/MindSpeed-MM.git -b agenticl_datasets
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..


git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout 26.0.0_core_r0.12.1
# 安装加速库
pip install -r requirements.txt
pip install -e .
cd ..

cd MindSpeed-MM
pip install -e .
```

建议使用私仓分支，主仓main分支由于版本更新，可能需要修改融合算子的代码

### 其他依赖

由于MindSpeed-MM的安装会自动修改一些依赖，因此安装完成后一定要执行下面的命令

```
pip install transformers==5.2.0 triton-ascend==3.2.0 accelerate==1.2.0
```


### GDN算子安装

GDN算子可大幅加速训练，否则训练速度很慢


```
git clone https://github.com/flashserve/flash-linear-attention-npu.git
# 建议使用Tag为v26.1.0的版本
cd flash-linear-attention-npu
git checkout v26.1.0

# 通过--soc参数精准指定当前设备芯片型号，示例配置--soc=ascend910_93，杜绝芯片型号不匹配导致的编译失败。
bash build.sh --soc=ascend910b --pkg --ops=chunk_bwd_dv_local,chunk_bwd_dqkwg,chunk_gated_delta_rule_bwd_dhu,prepare_wy_repr_bwd_da,prepare_wy_repr_bwd_full,chunk_fwd_o,chunk_gated_delta_rule_fwd_h,recurrent_gated_delta_rule,recompute_wu_fwd,causal_conv1d
# 安装run包，查看该路径下生成的安装包，不一定和实例一样
./build_out/cann-ops-transformer-custom_linux-x86_64.run
```

- 设备型号可以参考："ascend910b","ascend910_93","ascend950","acscend310p","kirinx90","kirin9030","mc62cm12a"

```
# torch_custom是flash-linear-attention-npu-main的子目录
cd torch_custom/fla_npu

bash gen.sh npu_custom.yaml

# gen.sh 会生成以下内容，可以用ls命令校验：
# op_plugin/config/v2r7/**: 配置文件
# torch_npu/csrc/aten/**: ATen 层适配代码
# torch_npu/utils/**: 工具函数

# 编译生成 WHL 安装包
python setup.py bdist_wheel

# 安装 whl 包
pip install dist/fla_npu*.whl --force-reinstall --no-deps

```

接下来，进行算子测试，ml_dtypes ct包仅为测试所用，不影响算子正常使用，按需install即可。


```
cd torch_custom/fla_npu/test
bash test.sh
```

fwd_h、fwd_o、bwd_dv_local的test脚本需要ml_dtypes、ct等库，这两个库是进行精度比较的，不影响算子的正常使用，仅仅阻塞了test，一般来说别的算子能pass，这几个也没大问题
需要注意的是，上述通过是基本要求，并不能完全确定gdn算子安装没有问题，使用过程中出现报错，尤其是device相关报错可以排查GDN算子安装情况或者直接重新安装。


## 训练流程

### 配置文件

整体分为五大块：`parallel`、`data`、`model`、`features`、`training`、`tools`。**这里的每一个参数都很重要，会影响训练的正确性，一定要配置正确！**

#### 2.2.1 parallel —— 并行策略

```yaml
parallel:
  fully_shard_parallel_size: 16 # FSDP2 的分片域大小，等于WORLD_SIZE，16表示参数、梯度和优化器状态将被切分到16张卡上协同计算。
  fsdp_plan: #  (FSDP分片计划)：FSDP2通过精细切分模型的不同模块来消除流水线并行的“空泡”，提升异构模型计算效率。
    apply_modules:  # FSDP2 wrap 的模块层级；可同时指定对多个模块使用FSDP2。其中{*}是通配符，意为对所有blocks和layers应用FSDP2。
      - model.visual
      - model.visual.blocks.{*}
      - model.language_model
      - model.language_model.embed_tokens
      - model.language_model.layers.{*}
      - model.language_model.layers.{*}.linear_attn
      - model.language_model.layers.{*}.mlp.experts
      - lm_head
    hook_modules:  # 注册 forward/backward hook 的目标模块 指定需要在特定模块的前向/后向函数中“钩入”FSDP操作的模块。
      - model.language_model.layers.{*}
    param_dtype: bf16  # 参数计算精度（FSDP2 mixed precision）参数存储采用bf16，跨卡梯度通信用fp32，平衡了显存效率与精度
    reduce_dtype: fp32  # 梯度规约精度
  ulysses_parallel_size: 8  #  (长序列并行)：将超长序列切分到8张卡上并行计算注意力，从而支持32k的超长上下文。
  expert_parallel_size: 1   # MoE 专家并行（这里关闭。ep_plan的apply_modules同样表明计划对模型的MoE层应用专家并行。
  ep_plan:
    apply_modules:
      - model.language_model.layers.{*}.mlp.experts
```

并行组合：`FSDP=16, Ulysses-CP=8, EP=1`。注意 `fully_shard_parallel_size` 与 `ulysses_parallel_size` 同时启用时，FSDP 域内会再切分序列维。

#### 2.2.2 数据相关

```yaml
data:
  dataset_param: 
    dataset_type: huggingface
    attr: # 一个字段映射表，告诉框架如何解析您的数据集JSON文件，例如"prompt": "input"等
      images: null
    preprocess_parameters: # 定义了数据预处理流水线。其中model_name_or_path、image_max_pixels等配置，控制着如何将文本和图片加载、缩放和编码为模型可读的token。
    basic_parameters: # 核心参数 定义了SFT的训练格式和输入输出。
      cutoff_len: 32768 # 最大序列长度。
      template: qwen3_vl # 指定了适配Qwen3-VL模型的对话模板。
      enable_thinking: true # 可能使模型输出推理过程。
      train_on_prompt: false 
      mask_history: false
      overwrite_cache: true
      preprocessing_batch_size: 256
      preprocessing_num_workers: 16
      max_samples: null # 使用多少数据集进行训练
    dataloader_param: 
        pin_memory: true
        shuffle: true # 采集数据随机
        dataloader_mode: sampler # dataloader_mode: sampler：选择使用分布式采样模式进行数据加载。
        drop_last: true
        sampler_type: BaseRandomBatchSampler # sampler_type: BaseRandomBatchSampler：选择基础随机批次采样器，确保数据分布随机。
        num_workers: 8
        collate_param:
          model_name: qwen3vl
          ignore_pad_token_for_loss: true
        enable_preload: true # 在GPU计算当前批次时，预加载下一批次数据。
```

### 2.3 模型配置

```yaml
model:
  model_id: qwen3_5_moe                # 在 ModelHub 中注册的 key（由 plugin 注入）
  model_name_or_path: *HF_MODEL_LOAD_PATH
  trust_remote_code: true
  attn_implementation: flash_attention_2   # 启用 ulysses_cp 时强制 flash_attention_2
  freeze:
    - model.visual                     # 冻结视觉编码器，仅训语言侧
  use_triton_gdn: false                # 关闭 Triton 实现的 GDN
  use_grouped_expert_matmul: true      # 启用 grouped GEMM 加速 MoE 专家计算
  loss_cfg:
    loss_type: default                 # default：以 valid token 数归一；raw：模型自带 loss 直接返回
    router_aux_loss_coef: 0.0          # MoE router 负载均衡 loss 系数（关闭）
```

### 2.4 训练特性

```yaml
features:
  loss_cfg:
    loss_type: default
  recompute: true                       # 激活重计算
  recompute_plan:
    apply_modules:
      - model.language_model.layers.{*}    # 对每层 transformer 都做 recompute
  enable_chunk_loss: false              # chunk loss（lm_head 上分块算 loss，省显存）
  chunkloss_plan:
    apply_module: lm_head
    chunk_size: 1024
  enable_activation_offload: false      # activation 卸载到 host
  activation_offload_plan:
    apply_modules:
     - model.visual.blocks.{*}
     - model.language_model.layers.{*}
```

### 2.5 训练超参

```yaml
training:
  micro_batch_size: 1                  # mbs，per-rank
  gradient_accumulation_steps: 32      # 梯度累积，global_batch = mbs * dp * grad_acc
  seed: 42
  lr: 2.0e-5
  lr_decay_style: cosine
  lr_warmup_ratio: 0.1
  weight_decay: 0.01
  train_iters: 2460
  clip_grad: 1.0
  init_model_with_meta_device: true    # 用 meta device 初始化，避免一次性占满 host 内存
  optimizer: adamw
  adam_fused: true
  save_interval: 615                   # 每 615 step 保存一次
  no_load_optim: true                  # 不加载 optimizer 状态（首次 SFT 不需要）
  no_load_rng: true
  no_save_optim: true                  # 训练后不保存 optimizer 状态
  no_save_rng: true
  load: /home/weight/Qwen3.6-35B-A3B-dcp/      # 转换后的 DCP 权重
  save: /home/weight/Qwen3.6-35B-A3B-ckp/      # 输出目录
  use_deter_comp: false                # 是否启用确定性算子
  plugin:                              # trainer 启动时 import 的插件，触发模型/数据集注册
    - mindspeed_mm/fsdp/models/qwen3_5_moe
    - mindspeed_mm/fsdp/data/datasets/huggingface
```

**下面给出一份适配训练的的ymal文件示例**

```yamll
# 并行策略
parallel:
  fully_shard_parallel_size: 16
  fsdp_plan:
    apply_modules: # 如果要开prefetch的话，请不要随意修改apply_modules的顺序
      - model.visual
      - model.visual.blocks.{*}
      - model.language_model
      - model.language_model.embed_tokens
      - model.language_model.layers.{*}
      - model.language_model.layers.{*}.linear_attn
      - model.language_model.layers.{*}.mlp.experts
      - lm_head
    hook_modules:
      - model.language_model.layers.{*}
    param_dtype: bf16
    reduce_dtype: fp32
  ulysses_parallel_size: 8 # 开启 ulysses-cp 时, 请将 model 的 attn_implementation 设置为 flash_attention_2
  expert_parallel_size: 1
  ep_plan:
    apply_modules:
      - model.language_model.layers.{*}.mlp.experts

### 数据相关配置
data:
  dataset_param:
    dataset_type: huggingface
    #数据集属性
    attr:
      images: null                      # 当前数据集为纯文本 agent trace
      messages: messages                # 对话字段
      tools: tools                      # 工具 schema 字段（OpenAI 风格 list[{"type":"function",...}]）
      role_tag: role                    # 消息中角色字段名
      content_tag: content              # 消息中内容字段名
      user_tag: user                    # 用户角色标识
      assistant_tag: assistant          # 助手角色标识
      system_tag: system                # 系统角色标识
      observation_tag: tool             # 数据中工具响应 role 是 "tool"
      function_tag: function_call       # 占位（OpenAI converter 会按 tool_calls 字段判定）
      # 以下字段如果数据中没有可以保持 null
      prompt: null
      query: null
      response: null
      history: null·
      formatting: openai                # OpenAI ChatCompletion 风格（含 tool_calls + reasoning_content）

    # 数据预处理
    preprocess_parameters:
      model_name_or_path: &HF_MODEL_LOAD_PATH /home/weight/Qwen3.6-35B-A3B/ # 替换为原始hf权重
      use_fast_tokenizer: true
      split_special_tokens: false
      image_max_pixels: 262144
      image_min_pixels: 1024
      video_max_pixels: 16384
      video_min_pixels: 0
      video_fps: 2.0
      video_maxlen: 64

    basic_parameters:
      cutoff_len: 262144
      template: qwen3_6
      enable_thinking: true
      train_on_prompt: false
      mask_history: false
      # tool_format: null
      dataset_dir: /home/SFT/datasets/agentical_ascendc_datasets/4th_sft_datasets/
      dataset:
        - /home/SFT/datasets/agentical_ascendc_datasets/4th_sft_datasets/03_combined.jsonl
      cache_dir: ./cache_dir/
      overwrite_cache: true
      preprocessing_batch_size: 256
      preprocessing_num_workers: 16
      max_samples: null

  # 数据加载
  dataloader_param:
    pin_memory: true
    shuffle: true
    dataloader_mode: sampler
    drop_last: true
    sampler_type: BaseRandomBatchSampler
    num_workers: 8
    collate_param:
      model_name: qwen3vl
      ignore_pad_token_for_loss: true
      pad_to_multiple_of: 16
    enable_preload: true

# 模型配置
model:
  model_id: qwen3_5_moe
  model_name_or_path: *HF_MODEL_LOAD_PATH
  trust_remote_code: true
  attn_implementation: flash_attention_2
  freeze:
    - model.visual
  # 融合算子配置
  gdn_implementation: AscendC
  causal_conv1d_implementation: triton
  use_grouped_expert_matmul: true
  # loss 配置
  loss_cfg:
    loss_type: default   # If you want raw loss in model, loss_type can be set to "raw".
    router_aux_loss_coef: 0.0

# 优化特性配置
features:
  # loss 配置
  loss_cfg:
    loss_type: default   # If you want raw loss in model, loss_type can be set to "raw".
    router_aux_loss_coef: 0.0
  # 重计算配置
  recompute: true
  recompute_plan:
      apply_modules:
        - model.language_model.layers.{*}
        # - model.visual.blocks.{*}
  # chunkloss配置
  enable_chunk_loss: true
  chunkloss_plan:
    apply_module: lm_head
    chunk_size: 1024
  # activation offload 配置
  enable_activation_offload: true
  activation_offload_plan:
    apply_modules:
     - model.visual.blocks.{*}
     - model.language_model.layers.{*}

# 训练配置
training:
  micro_batch_size: 1
  gradient_accumulation_steps: 16
  seed: 42
  lr: 2.0e-5
  lr_decay_style: cosine
  lr_warmup_ratio: 0.1
  weight_decay: 0.01
  train_iters: 940
  clip_grad: 1.0
  init_model_with_meta_device: true
  optimizer: adamw
  adam_fused: true
  save_interval: 235
  no_load_optim: true  # Do not load optimizer state; remove if loading is needed.
  no_load_rng: true  # Do not load RNG state; remove if loading is needed.
  no_save_optim: true  # Do not save optimizer state; remove if saving is needed.
  no_save_rng: true  # Do not save RNG state; remove if saving is needed.
  load: /home/weight/Qwen3.6-35B-A3B-dcp/ # 替换为转换后的dcp权重
  save: /home/weight/Qwen3.6-35B-A3B-agentical-ascendc-ckp-4th/
  use_deter_comp: false
  plugin:
    - mindspeed_mm/fsdp/models/qwen3_5_moe
    - mindspeed_mm/fsdp/data/datasets/huggingface

# 工具配置
tools:
  profile:
    enable: false
    profile_type: static
    ranks: [0]
    static_param:
      level: level1
      with_stack: false
      with_memory: true
      record_shapes: false
      with_cpu: true
      save_path: /home/l00899543/profiling
      start_step: 3
      end_step: 4
      data_simplification: false
      aic_metrics_type: PipeUtilization
  memory_profile:
      enable: false
      start_step: 1
      end_step: 2
      save_path: ./memory_snapshot
      dump_ranks: [0]
      stacks: all
      max_entries: null
      mem_info: false

```

### 训练前准备

在拉起训练前需要将从HF官网下载好权重，然后用下面的脚本将权重转换为DCP格式


```bash
mm-convert Qwen35Converter hf_to_dcp \
--hf_dir ckpt/hf_path/xxxxxxx \
--dcp_dir ckpt/dcp_path/xxxxxxx
```

### 数据集转换

如果是直接使用ascend_agent_cc_dataset下的数据集，则无需转化，如果使用自己的数据集，则根据情况考虑转换格式


### 拉起训练

拉起脚本在MindSpeed-MM/examples/qwen3_6/finetune_qwen3_6_35B_A3B.sh