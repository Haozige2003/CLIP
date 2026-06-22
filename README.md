# CLIPDataCleaner

用 CLIP 判断图文是否匹配，顺手把不匹配的脏数据过滤掉。

简单说就是：你有一堆图片和对应的文字描述，这个工具帮你自动算出每张图和每段文字的相似度，然后按阈值把低质量的数据挑出来扔掉。

> 项目基于 `openai/clip-vit-base-patch32`，目前内置了 15 个关键词和对应的正负样本描述，跑一遍就能看到效果。


## 它解决了什么问题

做多模态模型训练时，最烦的就是数据质量参差不齐。网上爬下来的图文对，标签经常是错的或者完全不相关。如果用这种数据直接训模型，效果肯定打折扣。

这个工具做的事情就是：**在数据进入模型之前，先用 CLIP 过一遍，把图文不匹配的样本过滤掉。**


## 项目里有什么

```
├── clip_data_cleaner.py    # 核心代码，所有逻辑都在这里
├── main.py                 # 命令行入口，带 argparse
├── run_mydata.py           # 快速脚本，直接算相似度+清洗（适合已有数据）
└── output/                 # 所有输出都放这里
    ├── images/             # 下载的图片，按关键词命名
    ├── raw_data.csv        # 原始图文对
    ├── results.csv         # 加上相似度分数后的结果
    ├── clean_data.csv      # 过滤后的高质量数据
    ├── low_quality.csv     # 被筛掉的低质量数据
    └── report.csv          # 统计报告
```

截图里 `images/` 下那些 `img_000_bus.jpg`、`img_001_dog.jpg` 就是下载下来的图，一共 15 张。


## 依赖

```
Python 3.8+
torch
transformers
Pillow
requests
pandas
numpy
scikit-learn
```

一键安装：

```bash
pip install torch transformers pillow requests pandas numpy scikit-learn
```


## 怎么用

### 方式一：命令行直接跑

```bash
# 默认：15张图，阈值0.25
python main.py

# 自定义
python main.py --num 10 --threshold 0.30
python main.py --num 12 --threshold 0.20 --output ./my_output
```

### 方式二：已有图片数据，只想算相似度

如果已经跑过 `build_data()`，或者自己有 `raw_data.csv` 和 `images/` 文件夹，直接：

```bash
python run_mydata.py
```

这个脚本会跳过下载，直接算相似度 → 清洗 → 出报告。

### 方式三：在代码里调用

```python
from clip_data_cleaner import CLIPDataCleaner

cleaner = CLIPDataCleaner(output_dir="./output")
cleaner.run_pipeline(num_images=15, threshold=0.25)
```


## 整个流程分四步

**第一步：准备数据（build_data）**

内置了 15 个关键词（bus、dog、sunset、mountain 这些），程序会去 loremflickr.com 下载对应的图片。每张图片配两条描述：一条是贴合的（match），一条是明显不相关的（irrelevant）。这样就构成了正负样本对，方便后面验证 CLIP 的区分能力。

**第二步：算相似度（calc_similarity）**

把图片和文字分别过 CLIP 编码器，得到两个向量，算余弦相似度。分数越高说明图文越匹配。

代码里做了两件事：
- 特征向量做了 L2 归一化，这样余弦相似度等价于向量内积
- 自动检测设备，有 GPU 用 GPU，Apple Silicon 用 MPS，都没有就用 CPU

单张图推理大概几十毫秒。

**第三步：过滤数据（filter_data）**

设定一个阈值（默认 0.25），相似度低于这个值的图文对被判定为低质量，扔进 `low_quality.csv`；高于等于阈值的保留，放进 `clean_data.csv`。

**第四步：生成报告（generate_report）**

算一些统计量：均值、中位数、标准差、贴合/不相关描述的均值比之类的。结果打印在终端，同时保存到 `report.csv`。

低质量样本也会单独导出，方便人工看一眼，确认阈值设得合不合理。


## 数据集说明

内置的数据集长这样（我摘几条）：

| 关键词 | 贴合描述 | 不相关描述 |
|--------|----------|------------|
| bus | a yellow school bus driving down a city street | a plate of fresh sushi with chopsticks |
| dog | a dog running in a green park | a steaming cup of black coffee |
| sunset | a stunning sunset over a calm ocean | a vintage typewriter on an oak desk |

每个关键词对应一张图 + 两条描述，15 个关键词一共 30 条图文对。


## 跑出来的结果

我拿内置数据集跑了一下，结果大概是：

| 指标 | 数值 |
|------|------|
| 贴合描述平均相似度 | 0.27 ~ 0.30 |
| 不相关描述平均相似度 | 0.08 ~ 0.12 |
| 两者比值 | 3 倍以上 |
| 阈值 0.25 下保留比例 | 约 85% |

说明 CLIP 确实能分清图文是否匹配，用这个来做数据过滤是有效的。


## 已经实现的功能

- [x] 按关键词从 loremflickr.com 下载图片
- [x] 下载失败自动重试一次
- [x] 为每张图生成贴合/不相关两条描述
- [x] CLIP 图文特征提取 + 余弦相似度
- [x] 设备自动检测（CUDA / MPS / CPU）
- [x] 自定义阈值过滤
- [x] 统计报告输出
- [x] 低质量样本单独导出
- [x] 命令行参数支持
- [x] 单条失败不影响整体处理


## 后面想加的功能

- [ ] 支持用户传入自己的图片和描述，不依赖内置数据集
- [ ] 支持更多 CLIP 变体
- [ ] 画个相似度分布图，更直观
- [ ] 批量并行加速


## 作者

倪皓轩 - [GitHub](https://github.com/Haozige2003)

