#### 命令行参数

| 参数 | 简写 | 描述 | 默认值 |
|------|------|------|--------|
| `--root-dir` | `-r` | 算子项目根目录 | `./` |
| `--output-dir` | `-o` | Parquet 文件的输出目录 | `./` |
| `--parquet-filename` | `-p` | 输出的 Parquet 文件名 | `train.parquet` |
| `--verbose` | `-v` | 启用详细日志输出 | `False` |
| `--log-level` | - | 日志级别 (DEBUG/INFO/WARNING/ERROR) | `INFO` |

#### 配置

该脚本支持通过命令行参数或字典配置：

```python
from generate_and_convert_dataset import ConfigLoader, DatasetProcessor

# 配置字典
config_dict = {
    "input": {
        "root_dir": "/path/to/ops"
    },
    "output": {
        "output_dir": "/path/to/output",
        "parquet_filename": "train.parquet"
    },
    "processing": {
        "skip_dirs": ["tests"],
        "exclude_files": ["config", "op_api"],
        "include_extensions": [".cpp", ".h"]
    },
    "field_mapping": {
        "instruction": "prompt",
        "output": "response"
    },
    "verbose": False,
    "log_level": "INFO"
}

# 加载配置并运行
config = ConfigLoader.from_dict(config_dict)
processor = DatasetProcessor(config)
results = processor.process()
```

#### 数据格式

生成的 Parquet 文件包含两列：

| 列名 | 描述 |
|------|------|
| `prompt` | 要求生成算子代码的指令 |
| `response` | 完整的算子源代码（.cpp 和 .h 文件拼接后的内容） |

#### 目录结构要求

输入目录应包含具有以下结构的算子目录：

```
ops_dataset/
├── op1/
│   ├── op_host/
│   │   ├── op1.cpp
│   │   └── op1.h
│   └── op_kernel/
│       ├── op1.cpp
│       └── op1.h
├── op2/
│   ├── op_host/
│   │   └── ...
│   └── op_kernel/
│       └── ...
└── ...
```

#### 处理流程

1. **发现**：扫描包含 `op_host` 和 `op_kernel` 子目录的目录
2. **收集**：从每个算子目录中获取所有 `.cpp` 和 `.h` 文件（排除 `config` 和 `op_api`）
3. **生成**：创建带有指令和拼接源代码的训练条目
4. **导出**：直接保存为 Parquet 格式