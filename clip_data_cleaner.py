"""
CLIPDataCleaner: 基于 CLIP 模型的多模态图文数据质量评估与清洗工具。

功能:
    - build_data:      自动下载图片并生成配对描述，输出 raw_data.csv
    - calc_similarity: 使用 CLIP 计算图文相似度，输出 results.csv
    - filter_data:     基于阈值过滤低质量图文对，输出清洗后数据
    - generate_report: 生成统计报告，输出低质量样本 low_quality.csv

依赖: torch, transformers, Pillow, pandas, scikit-learn, requests
"""

import os
import time
import random
import logging
from io import BytesIO
from typing import List, Tuple, Optional

import requests
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import CLIPProcessor, CLIPModel

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CLIPDataCleaner")

# 数据集定义 — 关键词 + 贴合描述 + 不相关描述（一一对应）
#   使用 loremflickr.com 按关键词下载图片，确保图片内容与描述真正匹配，
#   从而得到有区分度的 CLIP 相似度评分（贴合 >> 不相关）。

IMAGE_DATASET = [
    {
        "keyword": "cat",
        "match_desc": "a cute cat sitting on a sofa",
        "irrelevant_desc": "a modern glass skyscraper against a blue sky",
    },
    {
        "keyword": "dog",
        "match_desc": "a dog running in a green park",
        "irrelevant_desc": "a steaming cup of black coffee on a wooden table",
    },
    {
        "keyword": "sunset",
        "match_desc": "a stunning sunset over a calm ocean",
        "irrelevant_desc": "a vintage typewriter sitting on an oak desk",
    },
    {
        "keyword": "mountain",
        "match_desc": "a snowy mountain peak reflecting in a lake",
        "irrelevant_desc": "a plate of freshly baked chocolate chip cookies",
    },
    {
        "keyword": "pizza",
        "match_desc": "a slice of pepperoni pizza fresh from the oven",
        "irrelevant_desc": "a wild tiger walking through the jungle",
    },
    {
        "keyword": "sushi",
        "match_desc": "a fresh sushi platter arranged on a bamboo tray",
        "irrelevant_desc": "a lighthouse standing on a rocky cliff by the sea",
    },
    {
        "keyword": "castle",
        "match_desc": "a historic castle perched on a green hilltop",
        "irrelevant_desc": "a dolphin jumping out of the blue ocean",
    },
    {
        "keyword": "bridge",
        "match_desc": "a futuristic bridge spanning across a wide river",
        "irrelevant_desc": "a bowl of hot ramen with sliced eggs and spring onion",
    },
    {
        "keyword": "beach",
        "match_desc": "a tropical beach with crystal-clear turquoise water",
        "irrelevant_desc": "a microscope on a laboratory bench with glass slides",
    },
    {
        "keyword": "flower",
        "match_desc": "a field full of blooming lavender under sunlight",
        "irrelevant_desc": "a chess board mid-game with pieces scattered around",
    },
    {
        "keyword": "forest",
        "match_desc": "a lush green forest path in spring",
        "irrelevant_desc": "a shiny electric guitar leaning against an amplifier",
    },
    {
        "keyword": "bird",
        "match_desc": "a bird perched on a tree branch",
        "irrelevant_desc": "a steam locomotive pulling into an old railway station",
    },
    {
        "keyword": "coffee",
        "match_desc": "a steaming cup of black coffee on a wooden table",
        "irrelevant_desc": "a snowy mountain peak reflecting in a lake",
    },
    {
        "keyword": "car",
        "match_desc": "a red sports car parked on a city street",
        "irrelevant_desc": "a tropical beach with crystal-clear turquoise water",
    },
    {
        "keyword": "horse",
        "match_desc": "a horse galloping across an open field",
        "irrelevant_desc": "a modern glass skyscraper against a blue sky",
    },
]

# CLIPDataCleaner

class CLIPDataCleaner:
    """
    基于 CLIP 模型的图文数据质量评估与清洗工具。

    Parameters
    ----------
    model_name : str
        HuggingFace 上的 CLIP 模型名称，默认 `openai/clip-vit-base-patch32`。
    device : str or None
        运行设备，可选 `"cpu"`, `"cuda"`, `"mps"`。为 None 时自动检测。
    output_dir : str
        输出目录，默认当前目录。
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        output_dir: str = ".",
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # ---------- 设备选择 ----------
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        logger.info(f"使用设备: {self.device}")

        # ---------- 加载 CLIP 模型 ----------
        logger.info(f"正在加载 CLIP 模型: {model_name} ...")
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            logger.info("CLIP 模型加载完成。")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    # 辅助方法

    def _path(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)

    def _download_image(self, url: str, timeout: int = 30) -> Optional[Image.Image]:
        """下载单张图片，返回 PIL Image 或 None。"""
        try:
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            return img
        except Exception as e:
            logger.warning(f"图片下载失败 [{url}]: {e}")
            return None

    def _encode_image(self, image: Image.Image) -> np.ndarray:
        """将 PIL Image 编码为 CLIP 图像特征向量。"""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    def _encode_text(self, text: str) -> np.ndarray:
        """将文本编码为 CLIP 文本特征向量。"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()


    # 1. 数据构建 — build_data
    def build_data(self, num_images: int = 15) -> pd.DataFrame:
        """
        根据 IMAGE_DATASET 按关键词从 loremflickr.com 下载图片，
        为每张图片构造 2 个描述（贴合 / 不相关），生成 raw_data.csv。

        Parameters
        ----------
        num_images : int
            下载图片数量，默认 15（对应 IMAGE_DATASET 全部条目）。

        Returns
        -------
        pd.DataFrame
            包含 image_path, description, label 字段的原始数据。
        """
        logger.info(f"===== 开始数据构建 (目标 {num_images} 张) =====")

        images_dir = self._path("images")
        os.makedirs(images_dir, exist_ok=True)

        # 取前 num_images 条
        dataset = IMAGE_DATASET[: min(num_images, len(IMAGE_DATASET))]
        if num_images > len(IMAGE_DATASET):
            logger.warning(
                f"请求 {num_images} 张但数据集只有 {len(IMAGE_DATASET)} 条，"
                f"将下载 {len(IMAGE_DATASET)} 张。"
            )
        actual_count = len(dataset)

        records: List[dict] = []
        downloaded = 0

        for i, item in enumerate(dataset):
            keyword = item["keyword"]
            match_desc = item["match_desc"]
            irrelevant_desc = item["irrelevant_desc"]

            # 使用 loremflickr.com 按关键词下载匹配图片
            # 添加随机 seed 避免同一关键词返回相同图片（缓存）
            seed = random.randint(1, 9999)
            url = f"https://loremflickr.com/400/300/{keyword}?random={seed}"

            img = self._download_image(url)
            if img is None:
                # 重试一次
                logger.info(f"  第 1 次下载失败，重试...")
                seed = random.randint(1, 9999)
                url = f"https://loremflickr.com/400/300/{keyword}?random={seed}"
                img = self._download_image(url)

            if img is None:
                logger.warning(
                    f"  [{i+1:02d}/{actual_count}] 关键词='{keyword}' 下载失败，跳过"
                )
                continue

            # 保存图片
            img_filename = f"img_{downloaded:03d}_{keyword}.jpg"
            img_rel_path = os.path.join("images", img_filename)
            try:
                img.save(self._path(img_rel_path))
            except Exception as e:
                logger.warning(f"图片保存失败: {e}")
                continue

            # 添加两条记录
            records.append({
                "image_path": img_rel_path,
                "description": match_desc,
                "label": "match",
            })
            records.append({
                "image_path": img_rel_path,
                "description": irrelevant_desc,
                "label": "irrelevant",
            })

            downloaded += 1
            logger.info(
                f"  [{downloaded:02d}/{actual_count}] 下载成功 | "
                f"关键词='{keyword}' | 贴合='{match_desc[:40]}...'"
            )

        if downloaded == 0:
            raise RuntimeError(
                "所有图片下载均失败，请检查网络连接或 https://loremflickr.com 是否可访问。"
            )

        if downloaded < actual_count:
            logger.warning(f"仅成功下载 {downloaded}/{actual_count} 张图片。")

        # 保存 raw_data.csv
        df = pd.DataFrame(records)
        raw_path = self._path("raw_data.csv")
        df.to_csv(raw_path, index=False, encoding="utf-8-sig")
        logger.info(f"raw_data.csv 已保存至: {raw_path}  (共 {len(df)} 条记录)")
        return df

    # 2. 相似度计算 — calc_similarity
    def calc_similarity(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        遍历 raw_data.csv 中的图文对，使用 CLIP 计算余弦相似度，
        结果写入 results.csv。

        Parameters
        ----------
        df : pd.DataFrame or None
            原始数据 DataFrame。为 None 时自动读取 raw_data.csv。

        Returns
        -------
        pd.DataFrame
            包含 similarity_score 列的结果数据。
        """
        logger.info("===== 开始相似度计算 =====")

        if df is None:
            raw_path = self._path("raw_data.csv")
            if not os.path.exists(raw_path):
                raise FileNotFoundError(
                    f"未找到 {raw_path}，请先运行 build_data()。"
                )
            df = pd.read_csv(raw_path, encoding="utf-8-sig")

        total = len(df)
        similarity_scores: List[float] = []

        for idx, row in df.iterrows():
            img_rel_path = row["image_path"]
            description = row["description"]
            img_abs_path = self._path(img_rel_path)

            try:
                if not os.path.exists(img_abs_path):
                    logger.warning(f"[{idx+1}/{total}] 图片不存在: {img_abs_path}")
                    similarity_scores.append(np.nan)
                    continue

                image = Image.open(img_abs_path).convert("RGB")
                img_vec = self._encode_image(image)
                txt_vec = self._encode_text(description)

                sim = float(cosine_similarity(img_vec, txt_vec)[0][0])
                similarity_scores.append(round(sim, 6))

                logger.info(
                    f"  [{idx+1:02d}/{total}] sim={sim:.4f}  |  "
                    f"label={row['label']:<10}  |  {description[:55]}..."
                )

            except Exception as e:
                logger.error(f"[{idx+1}/{total}] 计算失败: {e}")
                similarity_scores.append(np.nan)

        df["similarity_score"] = similarity_scores

        result_path = self._path("results.csv")
        df.to_csv(result_path, index=False, encoding="utf-8-sig")
        logger.info(f"results.csv 已保存至: {result_path}")
        return df


    # 3. 智能数据清洗 — filter_data
    def filter_data(
        self,
        df: Optional[pd.DataFrame] = None,
        threshold: float = 0.25,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        基于相似度阈值过滤低质量图文对。

        Parameters
        ----------
        df : pd.DataFrame or None
            包含 similarity_score 的 DataFrame。为 None 时自动读取 results.csv。
        threshold : float
            相似度阈值，低于此值的视为低质量。默认 0.25。

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (clean_df, low_quality_df) — 清洗后的数据和低质量数据。
        """
        logger.info(f" 开始数据清洗 (threshold={threshold}) ")

        if df is None:
            result_path = self._path("results.csv")
            if not os.path.exists(result_path):
                raise FileNotFoundError(
                    f"未找到 {result_path}，请先运行 calc_similarity()。"
                )
            df = pd.read_csv(result_path, encoding="utf-8-sig")

        if "similarity_score" not in df.columns:
            raise ValueError(
                "数据中缺少 'similarity_score' 列，请先运行 calc_similarity()。"
            )

        total = len(df)
        clean_mask = df["similarity_score"] >= threshold
        clean_df = df[clean_mask].copy().reset_index(drop=True)
        low_quality_df = df[~clean_mask].copy().reset_index(drop=True)

        removed = len(low_quality_df)
        kept = len(clean_df)

        logger.info(
            f"清洗完成: 保留 {kept} 条, 移除 {removed} 条 "
            f"({removed / total * 100:.1f}%)"
        )

        clean_path = self._path("clean_data.csv")
        clean_df.to_csv(clean_path, index=False, encoding="utf-8-sig")
        logger.info(f"clean_data.csv 已保存至: {clean_path}")
        return clean_df, low_quality_df

    # 4. 报告生成 — generate_report

    def generate_report(
        self,
        df: Optional[pd.DataFrame] = None,
        low_quality_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        输出数据统计信息，并将低质量样本保存为 low_quality.csv 供人工复核。

        Parameters
        ----------
        df : pd.DataFrame or None
            全量数据（含 similarity_score）。为 None 时自动读取 results.csv。
        low_quality_df : pd.DataFrame or None
            低质量数据。为 None 时从 df 中按阈值 (0.25) 提取。

        Returns
        -------
        dict
            统计信息字典。
        """
        logger.info("===== 开始报告生成 =====")

        if df is None:
            result_path = self._path("results.csv")
            if not os.path.exists(result_path):
                raise FileNotFoundError(
                    f"未找到 {result_path}，请先运行 calc_similarity()。"
                )
            df = pd.read_csv(result_path, encoding="utf-8-sig")

        if "similarity_score" not in df.columns:
            raise ValueError("数据中缺少 'similarity_score' 列。")

        scores = df["similarity_score"].dropna()
        match_scores = df[df["label"] == "match"]["similarity_score"].dropna()
        irrelevant_scores = df[df["label"] == "irrelevant"]["similarity_score"].dropna()

        stats = {
            "总记录数": int(len(df)),
            "有效记录数（相似度非空）": int(len(scores)),
            "相似度均值": float(round(scores.mean(), 6)),
            "相似度中位数": float(round(scores.median(), 6)),
            "相似度标准差": float(round(scores.std(ddof=0), 6)),
            "相似度最小值": float(round(scores.min(), 6)),
            "相似度最大值": float(round(scores.max(), 6)),
            "---": "---",
            "贴合描述均值": float(round(match_scores.mean(), 6)) if len(match_scores) > 0 else None,
            "不相关描述均值": float(round(irrelevant_scores.mean(), 6)) if len(irrelevant_scores) > 0 else None,
            "贴合 / 不相关 比值": (
                float(round(match_scores.mean() / irrelevant_scores.mean(), 4))
                if len(match_scores) > 0 and len(irrelevant_scores) > 0 and irrelevant_scores.mean() > 0
                else None
            ),
        }

        # 低质量样本
        if low_quality_df is None:
            low_quality_df = df[df["similarity_score"] < 0.25].copy()

        if not low_quality_df.empty:
            lq_path = self._path("low_quality.csv")
            low_quality_df.to_csv(lq_path, index=False, encoding="utf-8-sig")
            logger.info(f"low_quality.csv 已保存至: {lq_path}")
        else:
            logger.info("无低质量样本，未生成 low_quality.csv。")

        stats["低质量样本数 (sim < 0.25)"] = len(low_quality_df)

        # 打印报告
        logger.info("\n" + "=" * 50)
        logger.info("          数 据 质 量 报 告")
        logger.info("=" * 50)
        for key, val in stats.items():
            logger.info(f"  {key}: {val}")
        logger.info("=" * 50)

        # 保存统计报告
        report_df = pd.DataFrame(list(stats.items()), columns=["指标", "数值"])
        report_path = self._path("report.csv")
        report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
        logger.info(f"report.csv 已保存至: {report_path}")
        return stats

    # 5. 一键运行完整流程 — run_pipeline

    def run_pipeline(self, num_images: int = 15, threshold: float = 0.25) -> dict:
        """
        依次执行 数据构建 → 相似度计算 → 数据清洗 → 报告生成。

        Parameters
        ----------
        num_images : int
            下载图片数量。
        threshold : float
            清洗阈值。

        Returns
        -------
        dict
            最终统计报告。
        """
        logger.info("=" * 60)
        logger.info("   CLIPDataCleaner 完整流水线启动")
        logger.info("=" * 60)
        t_start = time.time()

        raw_df = self.build_data(num_images=num_images)
        result_df = self.calc_similarity(df=raw_df)
        clean_df, low_df = self.filter_data(df=result_df, threshold=threshold)
        stats = self.generate_report(df=result_df, low_quality_df=low_df)

        elapsed = time.time() - t_start
        logger.info(f"\n流水线执行完毕，总耗时: {elapsed:.1f} 秒")
        return stats


if __name__ == "__main__":
    cleaner = CLIPDataCleaner(
        model_name="openai/clip-vit-base-patch32",
        output_dir="./output",
    )
    cleaner.run_pipeline(num_images=15, threshold=0.25)
