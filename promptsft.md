# 任务描述
你是基于华为昇腾处理器AscendC编程专家，你需要生成一个训练数据集，包含算子名称和对应算子的实现代码。在当前工程路径下，有4个ops仓库，针对每一个包含ops_host和ops_kernel的kernel，生成一条训练数据，最终把工程路径下所有AscendC kernel算子代码整理成一个SFT训练数据集

# 要求
思维链应展示从问题到代码的完整推理路径，包括关键思考步骤、逻辑分析和推导过程。

要求：
展示解决问题的清晰思路；
包含必要的分析步骤和中间推理；
逐步引导至最终答案；
使用自然、流畅的语言；
展示你作为AscendC专家的专业知识和思考过程；

# 工程路径
root_dir: "/home/l00868164/ops_dataset"
output_dir："/home/l00868164/ops_dataset/output"

# 数据集格式要求
 按照Alpaca 格式构建

# 格式参考
     {
        "instruction": "",
        "input": "帮我生成Ascend C高性能kernel算子，算子名称为XXX",
        "output": "",
        "system": "",
        "history": []
    }
    其中，input的算子名称为算子目录名字，output内容为拼接的源文件和头文件内容

# 工作流程
1.  **目录遍历**：脚本从根目录 (root_dir) 开始，按'分类' -> '算子名称' 的层级结构进行递归遍历。
2.  **文件检查**：对于每个算子目录，检查是否存在必要的文件：Kernel实现目录必须同时包含op_host和op_kernel文件夹,否则跳过该Kernel
3.  **output组装**：
    对每一个算子文件夹，收集op_host和op_kernel的一级目录中的所有 .cpp 和 .h 文件，排除config和op_api文件夹
    完整串联的 C/C++ 源文件和头文件中的代码内容，每个文件内容用 Markdown 代码块包裹，并以 `============================================================================================` 分隔。
4. **数据生成**：
   将组装好的output写成一条Alpaca格式数据集，参考Alpaca格式，保存到json文件，继续遍历其他算子目录
5.  **文件输出**：
    待全部目录遍历完后，将数据集保存到(output_dir) 下，命名格式为 `OpName_ImplTag_Idx.json`。

# 注意
步骤3串联的是文件中的代码内容，不是文件名

# 示例：
算子目录为：
add_layer_norm_grad/
├── docs/
│   └── aclnnAddLayerNormGrad.md
│ 
├── examples/
│   └── xxxx
│
├── op_graph/
│   └── xxxx
│
├── op_host/
│    ├── config/
│    │   └── xxxx
│    │
│    │── add_layer_norm_grad_def.cpp
│    │── add_layer_norm_grad_infershape.cpp
│    │── add_layer_norm_grad_tiling.cpp
│    │── add_layer_norm_grad_tiling.h
│    └── CMakeLists.txt 
│
├── op_kernel/
│    ├── config/
│    │   ├─── xxxx
│    │
│    ├── add_layer_norm_determinstic_compute.h
│    ├── add_layer_norm_grad_cut_d.h
│    ├── add_layer_norm_grad_cut_n.h
│    ├── add_layer_norm_grad_utils.h
│    ├── add_layer_norm_grad.cpp
│    └── CMakeLists.txt 
│
└── tests/
    └── xxxx

其中，docs、examples、op_graph、tests几个目录由于非op_host和op_kernel，不参与output组装。config文件夹排除，不参与output组装，CMakeLists.txt 不包含，因为不是.cpp 和 .h 文件。最终这9个文件参与output组装。
add_layer_norm_grad_def.cpp
add_layer_norm_grad_infershape.cpp
add_layer_norm_grad_tiling.cpp
add_layer_norm_grad_tiling.h
add_layer_norm_determinstic_compute.h
add_layer_norm_grad_cut_d.h
add_layer_norm_grad_cut_n.h
add_layer_norm_grad_utils.h
add_layer_norm_grad.cpp