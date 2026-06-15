# 使用准备好的 raw_data.csv 和 images 文件夹，直接进行相似度计算和清洗
from clip_data_cleaner import CLIPDataCleaner

def main():
    # 初始化清洗器（不调用 build_data）
    cleaner = CLIPDataCleaner(output_dir="./output")

    print("开始相似度计算")
    df = cleaner.calc_similarity()          # 读取 output/raw_data.csv，计算相似度
    print("相似度计算完成，结果保存在 output/results.csv")

    print("\n开始数据清洗")
    clean_df, low_df = cleaner.filter_data(df, threshold=0.25)
    print(f"清洗完成：保留 {len(clean_df)} 条，移除 {len(low_df)} 条")
    print("清洗结果保存在 output/clean_data.csv 和 output/low_quality.csv")

    print("\n生成质量报告")
    stats = cleaner.generate_report(df, low_df)
    print("报告已生成：output/report.csv")
    print("\n关键统计信息：")
    for key, value in stats.items():
        if not key.startswith("---"):
            print(f"  {key}: {value}")

    print("\n全部完成")

if __name__ == "__main__":
    main()