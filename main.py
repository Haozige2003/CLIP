"""
CLIPDataCleaner 项目入口。

用法:
    python main.py                     # 默认: 15张图片, threshold=0.25
    python main.py --num 10            # 下载10张图片
    python main.py --threshold 0.30    # 设置过滤阈值
    python main.py --num 12 --threshold 0.20

输出目录: ./output/
"""

import argparse
import sys

from clip_data_cleaner import CLIPDataCleaner


def main():
    parser = argparse.ArgumentParser(
        description="CLIP 图文数据质量评估与清洗工具"
    )
    parser.add_argument(
        "--num", type=int, default=15,
        help="下载图片数量 (默认 15，最大 15)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.25,
        help="相似度过滤阈值 (默认 0.25)",
    )
    parser.add_argument(
        "--output", type=str, default="./output",
        help="输出目录 (默认 ./output)",
    )
    parser.add_argument(
        "--model", type=str, default="openai/clip-vit-base-patch32",
        help="CLIP 模型名称",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="运行设备 (cpu / cuda / mps)，不指定则自动检测",
    )
    args = parser.parse_args()

    try:
        cleaner = CLIPDataCleaner(
            model_name=args.model,
            device=args.device,
            output_dir=args.output,
        )
        cleaner.run_pipeline(num_images=args.num, threshold=args.threshold)
    except KeyboardInterrupt:
        print("\n用户中断。")
        sys.exit(130)
    except Exception as e:
        print(f"\n运行出错: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
