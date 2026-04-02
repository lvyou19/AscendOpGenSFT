# AscendOpGenSFT

A tool for generating AscendC operator training datasets in SFT (Supervised Fine-Tuning) format.

## Features

- Scan operator directories containing `op_host` and `op_kernel` subdirectories
- Extract source code files (.cpp, .h)
- Generate Alpaca-format training data
- Directly save to Parquet format for efficient storage and processing

## Usage

### generate_and_convert_dataset.py

This script processes operator source code directories and generates training datasets in Parquet format.

#### Basic Usage

```bash
# Use default configuration
python generate_and_convert_dataset.py

# Specify input and output directories
python generate_and_convert_dataset.py --root-dir /path/to/ops --output-dir /path/to/output

# Verbose logging
python generate_and_convert_dataset.py --verbose
```

#### Command-Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--root-dir` | `-r` | Operator project root directory | `./` |
| `--output-dir` | `-o` | Output directory for Parquet file | `./` |
| `--parquet-filename` | `-p` | Output Parquet filename | `train.parquet` |
| `--verbose` | `-v` | Enable detailed logging output | `False` |
| `--log-level` | - | Logging level (DEBUG/INFO/WARNING/ERROR) | `INFO` |

#### Configuration

The script supports configuration via command-line arguments or dictionary configuration:

```python
from generate_and_convert_dataset import ConfigLoader, DatasetProcessor

# Configuration dictionary
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

# Load configuration and run
config = ConfigLoader.from_dict(config_dict)
processor = DatasetProcessor(config)
results = processor.process()
```

#### Data Format

The generated Parquet file contains two columns:

| Column | Description |
|--------|-------------|
| `prompt` | The instruction asking to generate the operator code |
| `response` | The complete operator source code (concatenated .cpp and .h files) |

#### Directory Structure Requirements

The input directory should contain operator directories with the following structure:

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

#### Processing Pipeline

1. **Discovery**: Scan for directories containing both `op_host` and `op_kernel` subdirectories
2. **Collection**: Gather all `.cpp` and `.h` files from each operator directory (excluding `config` and `op_api`)
3. **Generation**: Create training entries with instruction and concatenated source code
4. **Export**: Save directly to Parquet format
