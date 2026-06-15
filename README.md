# CLIPDataCleaner — 图文数据质量评估与清洗工具

基于 OpenAI CLIP 模型 (`openai/clip-vit-base-patch32`) 的多模态数据质量评估与清洗工具。

使用 **loremflickr.com 按关键词下载图片**，确保图片内容与描述真正匹配，
从而得到有区分度的 CLIP 相似度评分（贴合 >> 不相关），实现精准评估。

## 功能

| 方法 | 说明 | 输出 |
|------|------|------|
| `build_data()` | 按关键词下载图片并生成配对描述 | `raw_data.csv` |
| `calc_similarity()` | 使用 CLIP 计算图文余弦相似度 | `results.csv` |
| `filter_data()` | 基于阈值 (默认 0.25) 过滤低质量数据 | `clean_data.csv` |
| `generate_report()` | 生成统计报告，输出需人工复核的样本 | `low_quality.csv`, `report.csv` |
| `run_pipeline()` | 一键运行上述完整流水线 | 全部输出文件 |

## 数据集（15 个关键词）

cat, dog, sunset, mountain, pizza, sushi, castle, bridge,
beach, flower, forest, bird, coffee, car, horse

每个关键词对应 1 条贴合描述 + 1 条语义不相关描述，共 30 条记录。

## 安装

```bash
# Python 3.9+
pip install -r requirements.txt
```

## 使用

### 命令行

```bash
python main.py                     # 默认运行
python main.py --num 10 --threshold 0.30
```

### Python API

```python
from clip_data_cleaner import CLIPDataCleaner

cleaner = CLIPDataCleaner(output_dir="./output")
cleaner.run_pipeline(num_images=15, threshold=0.25)
```

## 输出文件

```
output/
├── images/           # 下载的图片
├── raw_data.csv      # 图片 + 描述（含 match / irrelevant 标签）
├── results.csv       # raw_data + CLIP 相似度分数
├── clean_data.csv    # 过滤后的高质量图文对
├── low_quality.csv   # 低质量样本（供人工复核）
└── report.csv        # 统计报告
```

## 依赖

- Python 3.9
- torch, transformers, Pillow, pandas, scikit-learn, requests
