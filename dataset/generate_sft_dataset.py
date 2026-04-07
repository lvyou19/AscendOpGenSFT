#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AscendC SFT Dataset Generator

Generate Alpaca-format training dataset from operator directories and save directly to Parquet format.
Supports command-line arguments and configuration files.

Examples:
    # Run with default configuration
    python generate_and_convert_dataset.py
    
    # Specify input/output directories
    python generate_and_convert_dataset.py --root-dir /path/to/ops --output-dir /path/to/output
    
    # Verbose logging
    python generate_and_convert_dataset.py --verbose
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd
from tqdm import tqdm


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass(frozen=True)
class FieldMapping:
    """Field mapping configuration for Parquet output."""
    instruction: str = "prompt"
    output: str = "response"


@dataclass(frozen=True)
class InputConfig:
    """Input configuration."""
    root_dir: Path = Path("./")
    
    def __post_init__(self):
        if not self.root_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.root_dir}")


@dataclass(frozen=True)
class OutputConfig:
    """Output configuration."""
    output_dir: Path = Path("./")
    parquet_filename: str = "train.parquet"


@dataclass(frozen=True)
class ProcessingConfig:
    """Processing configuration."""
    skip_dirs: List[str] = field(default_factory=lambda: ["tests"])
    exclude_files: List[str] = field(default_factory=lambda: ["config", "op_api"])
    include_extensions: List[str] = field(default_factory=lambda: [".cpp", ".h"])
    separator: str = "\n" + "=" * 92 + "\n"


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    field_mapping: FieldMapping = field(default_factory=FieldMapping)
    verbose: bool = False
    log_level: str = "INFO"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class OpInfo:
    """Operator information."""
    name: str
    path: Path
    host_path: Path
    kernel_path: Path


@dataclass
class FileContent:
    """File content."""
    filename: str
    filepath: Path
    content: str


@dataclass
class AlpacaEntry:
    """Alpaca-format training data entry."""
    instruction: str
    output: str


# =============================================================================
# Protocol Definitions (Interfaces)
# =============================================================================

class SourceCollector(Protocol):
    """Source file collector protocol."""
    def collect(self, op_info: OpInfo) -> List[FileContent]:
        """Collect source file contents."""
        ...


class EntryCreator(Protocol):
    """Training entry creator protocol."""
    def create(self, op_info: OpInfo, files: List[FileContent]) -> AlpacaEntry:
        """Create training entry."""
        ...


class DataWriter(Protocol):
    """Data writer protocol."""
    def write(self, data: List[AlpacaEntry], output_path: Path) -> None:
        """Write data to file."""
        ...


# =============================================================================
# Core Component Implementations
# =============================================================================

class DefaultSourceCollector:
    """Default source file collector."""
    
    def __init__(self, config: ProcessingConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    def collect(self, op_info: OpInfo) -> List[FileContent]:
        """Collect source files from op_host and op_kernel directories."""
        files_content: List[FileContent] = []
        
        for base_path in [op_info.host_path, op_info.kernel_path]:
            if not base_path.exists():
                self.logger.warning(f"Directory does not exist: {base_path}")
                continue
                
            for item in os.listdir(base_path):
                item_path = base_path / item
                
                # Skip excluded files and directories
                if item in self.config.exclude_files:
                    continue
                if item_path.is_dir():
                    continue
                
                # Check file extensions
                if not any(item.endswith(ext) for ext in self.config.include_extensions):
                    continue
                
                try:
                    content = item_path.read_text(encoding='utf-8', errors='ignore')
                    files_content.append(FileContent(
                        filename=item,
                        filepath=item_path,
                        content=content
                    ))
                except Exception as e:
                    self.logger.error(f"Error reading file {item_path}: {e}")
        
        return files_content


class AlpacaEntryCreator:
    """Alpaca-format training entry creator."""
    
    def __init__(self, config: ProcessingConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    def create(self, op_info: OpInfo, files: List[FileContent]) -> AlpacaEntry:
        """Create Alpaca-format training data entry."""
        output_parts = []
        for file_info in files:
            output_parts.append(
                f"```cpp\n// File: {file_info.filename}\n{file_info.content}\n```"
            )
        
        output = self.config.separator.join(output_parts)
        
        return AlpacaEntry(
            instruction=f"请生成Ascend C高性能kernel算子 {op_info.name} 的完整实现代码",
            output=output
        )


class ParquetDataWriter:
    """Parquet data writer."""
    
    def __init__(self, field_mapping: FieldMapping, logger: Optional[logging.Logger] = None):
        self.field_mapping = field_mapping
        self.logger = logger or logging.getLogger(__name__)
    
    def write(self, data: List[AlpacaEntry], output_path: Path) -> pd.DataFrame:
        """Convert data to Parquet format and write to file."""
        rows = []
        for entry in data:
            row = {}
            for json_field, parquet_field in self.field_mapping.__dict__.items():
                row[parquet_field] = getattr(entry, json_field, "")
            rows.append(row)
        
        df = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        self.logger.info(f"Parquet file saved: {output_path}")
        return df


# =============================================================================
# Operator Discoverer
# =============================================================================

class OpDiscoverer:
    """Operator discoverer - responsible for finding valid operator directories."""
    
    def __init__(self, config: ProcessingConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    def discover(self, root_dir: Path) -> List[OpInfo]:
        """Find all valid operator directories containing op_host and op_kernel."""
        valid_ops: List[OpInfo] = []
        
        for dirpath_str, dirnames, filenames in os.walk(root_dir):
            dirpath = Path(dirpath_str)
            
            # Skip specified directories
            if any(skip_dir in dirpath.parts for skip_dir in self.config.skip_dirs):
                continue
            
            op_host_path = dirpath / 'op_host'
            op_kernel_path = dirpath / 'op_kernel'
            
            if op_host_path.is_dir() and op_kernel_path.is_dir():
                op_info = OpInfo(
                    name=dirpath.name,
                    path=dirpath,
                    host_path=op_host_path,
                    kernel_path=op_kernel_path
                )
                valid_ops.append(op_info)
                self.logger.debug(f"Found operator: {op_info.name} @ {op_info.path}")
        
        return valid_ops


# =============================================================================
# Main Processor
# =============================================================================

class DatasetProcessor:
    """Dataset processor - coordinates all components to complete the processing pipeline."""
    
    def __init__(
        self,
        config: AppConfig,
        discoverer: Optional[OpDiscoverer] = None,
        collector: Optional[SourceCollector] = None,
        entry_creator: Optional[EntryCreator] = None,
        data_writer: Optional[ParquetDataWriter] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components (using dependency injection or defaults)
        self.discoverer = discoverer or OpDiscoverer(config.processing, self.logger)
        self.collector = collector or DefaultSourceCollector(config.processing, self.logger)
        self.entry_creator = entry_creator or AlpacaEntryCreator(config.processing, self.logger)
        self.data_writer = data_writer or ParquetDataWriter(
            config.field_mapping, self.logger
        )
    
    def process(self) -> Dict[str, Any]:
        """Execute the complete processing pipeline.
        
        Returns:
            Dictionary containing processing results.
        """
        results = {
            "total_ops": 0,
            "processed": 0,
            "skipped": [],
            "dataset": [],
            "output_files": {}
        }
        
        self.logger.info("=" * 70)
        self.logger.info("AscendC SFT Dataset Generator")
        self.logger.info("=" * 70)
        
        # Stage 1: Discover operators
        self.logger.info("\n[Stage 1/3] Scanning for valid operator directories...")
        valid_ops = self.discoverer.discover(self.config.input.root_dir)
        results["total_ops"] = len(valid_ops)
        self.logger.info(f"✓ Found {len(valid_ops)} valid operator directories")
        
        # Stage 2: Process operators and generate training data
        self.logger.info("\n[Stage 2/3] Processing operators and generating training data...")
        dataset: List[AlpacaEntry] = []
        
        progress_bar = tqdm(valid_ops, desc="Processing operators", unit="op") if not self.config.verbose else valid_ops
        
        for idx, op_info in enumerate(progress_bar):
            if isinstance(progress_bar, tqdm):
                progress_bar.set_postfix({"current": op_info.name})
            else:
                self.logger.info(f"  Processing [{idx+1}/{len(valid_ops)}]: {op_info.name}")
            
            # Collect source files
            files = self.collector.collect(op_info)
            
            if not files:
                self.logger.warning(f"  ⚠️  Warning: No source files found for {op_info.name}, skipping")
                results["skipped"].append(op_info.name)
                continue
            
            self.logger.debug(f"  ✓ Collected {len(files)} source files")
            
            # Create training entry
            entry = self.entry_creator.create(op_info, files)
            dataset.append(entry)
        
        results["dataset"] = dataset
        results["processed"] = len(dataset)
        
        # Stage 3: Save to Parquet
        self.logger.info("\n[Stage 3/3] Saving to Parquet format...")
        parquet_path = self.config.output.output_dir / self.config.output.parquet_filename
        
        if dataset:
            df = self.data_writer.write(dataset, parquet_path)
            results["output_files"]["parquet"] = str(parquet_path)
            results["dataframe"] = df
            
            # Display field mapping
            self.logger.info("Parquet field mapping:")
            for json_field, parquet_field in self.config.field_mapping.__dict__.items():
                self.logger.info(f"  {json_field} -> {parquet_field}")
        else:
            self.logger.warning("⚠️ Warning: No data to save")
        
        return results
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print processing summary."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("Generation Summary Report")
        self.logger.info("=" * 70)
        self.logger.info(f"Input directory: {self.config.input.root_dir}")
        self.logger.info(f"Output directory: {self.config.output.output_dir}")
        
        self.logger.info("\nProcessing Statistics:")
        self.logger.info(f"  Total operators: {results['total_ops']}")
        self.logger.info(f"  Successfully processed: {results['processed']}")
        self.logger.info(f"  Skipped operators: {len(results['skipped'])}")
        if results['skipped']:
            self.logger.info(f"  Skipped operator list: {', '.join(results['skipped'])}")
        
        self.logger.info("\nOutput Files:")
        if "parquet" in results["output_files"]:
            parquet_path = Path(results["output_files"]["parquet"])
            if parquet_path.exists():
                parquet_size = parquet_path.stat().st_size / 1024 / 1024
                self.logger.info(f"  Parquet file: {parquet_path} ({parquet_size:.2f} MB)")
         
        self.logger.info("=" * 70)
        self.logger.info("Dataset generation completed!")


# =============================================================================
# Configuration Loader
# =============================================================================

class ConfigLoader:
    """Configuration loader - supports multiple configuration sources."""
    
    @staticmethod
    def from_args(args: argparse.Namespace) -> AppConfig:
        """Load configuration from command-line arguments."""
        input_config = InputConfig(
            root_dir=Path(args.root_dir) if args.root_dir else Path("/home/l00868164/ops_dataset")
        )
        
        output_config = OutputConfig(
            output_dir=Path(args.output_dir) if args.output_dir else Path("/home/l00868164/ops_dataset/output"),
            parquet_filename=args.parquet_filename or "train.parquet"
        )
        
        processing_config = ProcessingConfig()
        field_mapping = FieldMapping()
        
        return AppConfig(
            input=input_config,
            output=output_config,
            processing=processing_config,
            field_mapping=field_mapping,
            verbose=args.verbose,
            log_level="DEBUG" if args.verbose else "INFO"
        )
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> AppConfig:
        """Load configuration from dictionary (can be used for config files)."""
        input_config = InputConfig(
            root_dir=Path(config_dict.get("input", {}).get("root_dir", "/home/l00868164/ops_dataset"))
        )
        
        output_dict = config_dict.get("output", {})
        output_config = OutputConfig(
            output_dir=Path(output_dict.get("output_dir", "/home/l00868164/ops_dataset/output")),
            parquet_filename=output_dict.get("parquet_filename", "train.parquet")
        )
        
        processing_dict = config_dict.get("processing", {})
        processing_config = ProcessingConfig(
            skip_dirs=processing_dict.get("skip_dirs", ["tests"]),
            exclude_files=processing_dict.get("exclude_files", ["config", "op_api"]),
            include_extensions=processing_dict.get("include_extensions", [".cpp", ".h"]),
            separator=processing_dict.get("separator", "\n" + "=" * 92 + "\n")
        )
        
        mapping_dict = config_dict.get("field_mapping", {})
        field_mapping = FieldMapping(
            instruction=mapping_dict.get("instruction", "prompt"),
            output=mapping_dict.get("output", "response")
        )
        
        return AppConfig(
            input=input_config,
            output=output_config,
            processing=processing_config,
            field_mapping=field_mapping,
            verbose=config_dict.get("verbose", False),
            log_level=config_dict.get("log_level", "INFO")
        )


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging system."""
    logger = logging.getLogger("ascendc_dataset")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Command-Line Argument Parser
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog='generate_and_convert_dataset.py',
        description='AscendC SFT Dataset Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default configuration
    python generate_and_convert_dataset.py
    
    # Specify input/output directories
    python generate_and_convert_dataset.py --root-dir /path/to/ops --output-dir /path/to/output
    
    # Verbose logging
    python generate_and_convert_dataset.py --verbose
        """
    )
    
    # Input/output configuration
    parser.add_argument(
        '--root-dir', '-r',
        type=str,
        help='Operator project root directory'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory'
    )
    
    parser.add_argument(
        '--parquet-filename', '-p',
        type=str,
        default='train.parquet',
        help='Parquet filename (default: train.parquet)'
    )
    
    # Logging configuration
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function."""
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigLoader.from_args(args)
    
    # Setup logging
    logger = setup_logging(config.log_level)
    
    try:
        # Create and run processor
        processor = DatasetProcessor(config, logger=logger)
        results = processor.process()
        processor.print_summary(results)
        
        # Return success exit code
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("\nOperation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=config.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
