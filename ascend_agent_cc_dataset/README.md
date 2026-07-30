# AscendC Agent 数据集

## ascend_agent_cc_dataset
该目录下是到0629为止，成功蒸馏出的7502条AscendC Agent数据集，使用claude code作为agent框架

## scripts
该目录下是将数据集整合成为支持MindSpeed-MM训练的格式的脚本，同时也包含了筛选合格数据集、统计数据集信息等功能。
- 使用方式：
```
python run_pipeline.py --data-root <DIR> --out <DIR>
```
