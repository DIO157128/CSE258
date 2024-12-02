import json
import logging
import os
import string
from collections import Counter, defaultdict
import seaborn as sns
import numpy as np
import pandas as pd
import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
colors = ["#3769AE", "#77ABD9", "#10779F", "#3892AA", "#1E9A93"]


def get_variation():
    # 数据集名称
    name_of_datasets = ['All_Beauty', 'Digital_Music', 'Gift_Cards', 'Magazine_Subscriptions', 'Subscription_Boxes']

    # 初始化结果字典
    results = {}

    # 遍历每个数据集
    for n in tqdm.tqdm(name_of_datasets, desc="Processing datasets"):
        # 加载数据集
        url = f"raw_review_{n}"
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", url, trust_remote_code=True)
        dataset = dataset["full"]

        # 提取所有评分
        ratings = []
        for review in tqdm.tqdm(dataset, desc=f"Processing reviews in {n}"):
            ratings.append(review['rating'])

        # 计算均值和方差
        mean_rating = round(np.mean(ratings), 2)
        variance_rating = round(np.var(ratings), 2)

        # 将结果存储到字典
        results[n] = {
            "mean": mean_rating,
            "variance": variance_rating
        }

    pd.DataFrame.from_dict(results, orient='index').to_csv("rating_stats.csv")


def preprocess_text(text):
    # 将文本转换为小写，去除标点符号
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))  # 去除标点
    return text

def get_frequency():
    name_of_datasets = ['All_Beauty', 'Digital_Music', 'Gift_Cards', 'Magazine_Subscriptions', 'Subscription_Boxes']
    stop_words = set([
        "i", "you", "he", "she", "it", "we", "they", "a", "an", "the",
        "and", "but", "or", "on", "in", "at", "to", "of", "with", "for", "is", "are", "was", "were", "be", "been",
        "being", "this", "", "my", "not", "that", "them", "its", "me", "your", "yours", "yourself", "yourselves",
    ])
    for n in tqdm.tqdm(name_of_datasets, desc="Processing datasets"):
        # 动态加载数据集
        url = f"raw_review_{n}"
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", url, trust_remote_code=True)
        dataset = dataset["full"]

        # 初始化词频字典
        word_frequency_dict = {}

        # 遍历每个评论
        for review in tqdm.tqdm(dataset, desc=f"Processing reviews in {n}"):
            text = review['text']
            title = review['title']
            total_text = title + " " + text.replace("<br />-", "").replace("<br />", "")
            total_text = preprocess_text(total_text)
            words = total_text.split(" ")
            for word in words:
                if word not in stop_words:
                    if word not in word_frequency_dict:
                        word_frequency_dict[word] = 1
                    else:
                        word_frequency_dict[word] += 1

        # 统计词频最高的前 25 个单词
        top_25 = sorted(word_frequency_dict.items(), key=lambda x: x[1], reverse=True)[:25]
        words, frequencies = zip(*top_25)

        # 绘制词频分布直方图
        plt.figure(figsize=(12, 8))
        plt.bar(words, frequencies, color="#3769AE")

        # 设置标题和轴标签
        plt.title(f"Top 25 Word Frequencies in {n}", fontsize=20, pad=10)
        plt.xlabel("Words", fontsize=18)
        plt.ylabel("Frequencies", fontsize=18)

        # 调整 x 轴标签的旋转角度和字体大小
        plt.xticks(rotation=45, fontsize=14, ha="right")
        plt.yticks(fontsize=16)

        # 加粗坐标轴
        ax = plt.gca()
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["bottom"].set_color('black')
        ax.spines["left"].set_linewidth(2)
        ax.spines["left"].set_color('black')

        # 隐藏右侧和顶部边框
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # 去掉横向网格线
        ax.grid(visible=False)

        # 调整布局
        plt.tight_layout()

        # 保存图像
        plt.savefig(f"figs/frequency/fre_{n}.png", dpi=300)

        # 显示图像
        plt.show()

def get_frequency_by_rating():
    name_of_datasets = ['All_Beauty', 'Digital_Music', 'Gift_Cards', 'Magazine_Subscriptions', 'Subscription_Boxes']

    # 停用词列表
    stop_words = set([
        "i", "you", "he", "she", "it", "we", "they", "a", "an", "the",
        "and", "but", "or", "on", "in", "at", "to", "of", "with", "for", "is", "are", "was", "were", "be", "been",
        "being", "this", "", "my", "not", "that", "them", "its", "me", "your", "yours", "yourself", "yourselves",
    ])


    # 遍历数据集
    for n in tqdm.tqdm(name_of_datasets, desc="Processing datasets"):
        # 动态加载数据集
        url = f"raw_review_{n}"
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", url, trust_remote_code=True)
        dataset = dataset["full"]

        # 初始化一个存储不同评分的词频字典
        rating_word_frequency = defaultdict(lambda: defaultdict(int))

        # 遍历每个评论
        for review in tqdm.tqdm(dataset, desc=f"Processing reviews in {n}"):
            rating = review['rating']  # 获取评分
            text = review['text']
            title = review['title']
            total_text = title + " " + text.replace("<br />-", "").replace("<br />", "")
            total_text = preprocess_text(total_text)
            words = total_text.split(" ")
            for word in words:
                if word not in stop_words:
                    rating_word_frequency[rating][word] += 1

        # 遍历每个评分绘制直方图
        for rating, word_freq_dict in rating_word_frequency.items():
            # 统计词频最高的前 25 个单词
            top_25 = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)[:25]
            words, frequencies = zip(*top_25)

            # 绘制词频分布直方图
            plt.figure(figsize=(12, 8))
            plt.bar(words, frequencies, color="#3769AE")

            # 设置标题和轴标签
            plt.title(f"Top 25 Word Frequencies in {n} (Rating {rating})", fontsize=20, pad=10)
            plt.xlabel("Words", fontsize=18)
            plt.ylabel("Frequencies", fontsize=18)

            # 调整 x 轴标签的旋转角度和字体大小
            plt.xticks(rotation=45, fontsize=14, ha="right")
            plt.yticks(fontsize=16)

            # 加粗坐标轴
            ax = plt.gca()
            ax.spines["bottom"].set_linewidth(2)
            ax.spines["bottom"].set_color('black')
            ax.spines["left"].set_linewidth(2)
            ax.spines["left"].set_color('black')

            # 隐藏右侧和顶部边框
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            # 去掉横向网格线
            ax.grid(visible=False)

            # 调整布局
            plt.tight_layout()

            # 保存图像
            plt.savefig(f"figs/frequency_rating/fre_{n}_rating_{rating}.png", dpi=300)

            # 显示图像
            plt.show()

def get_length():
    name_of_datasets = ['All_Beauty', 'Digital_Music', 'Gift_Cards', 'Magazine_Subscriptions', 'Subscription_Boxes']
    length_data = []
    for n in tqdm.tqdm(name_of_datasets):
        url = "raw_review_{}".format(n)
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", url, trust_remote_code=True)
        dataset = dataset["full"]

        for review in tqdm.tqdm(dataset):
            text = review['text']
            title = review['title']
            total_text = title + " " + text.replace("<br />-", "").replace("<br />", "")
            total_text = preprocess_text(total_text)
            words = total_text.split(" ")
            length_data.append({"Dataset": n, "Sequence Length": len(words)})
    violin_data = pd.DataFrame(length_data)

    # 使用 Seaborn 绘制小提琴图
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.violinplot(
        x="Dataset",
        y="Sequence Length",
        data=violin_data,
        palette=colors,
        cut=0,
    )

    color_mapping = dict(zip(name_of_datasets, colors))
    for dataset_name, color in color_mapping.items():
        plt.plot([], [], color=color, label=dataset_name, linewidth=6)  # 创建一个空线段用于图例

    # # 设置图例
    # plt.legend(title="Datasets", fontsize=14, title_fontsize=16)

    # 设置标题
    plt.title("Sequence Length Distributions Across Datasets", fontsize=20, pad=10)  # 标题上移

    # 设置坐标轴标签字体
    # plt.xlabel("Dataset", fontsize=18)
    plt.xlabel("", fontsize=18)
    plt.ylabel("Sequence Length", fontsize=18)

    # 调整x轴标签旋转
    plt.xticks(rotation=15, fontsize=16)
    plt.yticks(fontsize=16)

    # 加粗坐标轴
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["bottom"].set_color('black')
    ax.spines["left"].set_linewidth(2)
    ax.spines["left"].set_color('black')

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # 去掉横向等高线
    ax.grid(visible=False)

    # 设置y轴的0在原点
    ax.set_ylim(0, 300)

    plt.tight_layout()

    # 保存图像
    plt.savefig("figs/sequence_length.png", dpi=300)
    plt.show()

def get_rating_distribution():
    # List of dataset names
    name_of_datasets = ['All_Beauty', 'Digital_Music', 'Gift_Cards', 'Magazine_Subscriptions', 'Subscription_Boxes']

    # Initialize a dictionary to hold data for the bar chart
    combined_data = {"Rating": [1, 2, 3, 4, 5]}  # Initialize with rating labels

    # Loop through datasets
    for n in tqdm.tqdm(name_of_datasets, desc="Processing datasets"):
        # Dynamically create dataset URL
        url = f"raw_review_{n}"

        # Load dataset
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", url, trust_remote_code=True)
        dataset = dataset["full"]

        # Extract ratings and count occurrences
        ratings = dataset['rating']
        rating_count = Counter(ratings)
        total_ratings = sum(rating_count.values())

        # Calculate proportions for each rating (1 to 5)
        proportions = [rating_count.get(score, 0) / total_ratings for score in range(1, 6)]

        # Add proportions to combined_data
        combined_data[n] = proportions

    # Convert combined_data to a DataFrame
    rating_df = pd.DataFrame(combined_data)

    def plot_stacked_bar_chart(rating_df):
        # 设置 x 和 y 的数据
        datasets = rating_df.columns[1:]  # 跳过第一列 'Rating'
        ratings = rating_df["Rating"]  # 评分等级
        data = rating_df.iloc[:, 1:].T  # 转置评分数据，每行对应一个数据集

        # 累计偏移量
        cumulative = np.zeros(len(datasets))

        # 使用指定颜色
        colors = ["#3769AE", "#77ABD9", "#3892AA", "#10779F", "#1E9A93"]

        # 初始化图形
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制每个评分部分
        for i, rating in enumerate(ratings):
            ax.bar(
                datasets,
                data.iloc[:, i],  # 第 i 列的数据
                bottom=cumulative,  # 累计高度
                label=f"Rating {rating}",
                color=colors[i]
            )
            # 更新累计高度
            cumulative += data.iloc[:, i]

        # 设置标题和轴标签
        ax.set_title("Percentage Distribution of Ratings Across Datasets", fontsize=20, pad=10)
        plt.xlabel("", fontsize=18)
        plt.ylabel("Percentage (%)", fontsize=18)

        # 调整 x 和 y 轴标签的字体大小和旋转角度
        plt.xticks(rotation=15, fontsize=16)
        plt.yticks(fontsize=16)

        # 加粗坐标轴
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["bottom"].set_color('black')
        ax.spines["left"].set_linewidth(2)
        ax.spines["left"].set_color('black')

        # 隐藏右侧和顶部边框
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # 去掉横向网格线
        ax.grid(visible=False)

        # 设置 y 轴范围为 0 到 1（百分比）
        ax.set_ylim(0, 1)

        # 添加图例
        ax.legend(title="Ratings", fontsize=12, title_fontsize=14)

        # 调整布局
        plt.tight_layout()

        # 显示图表
        plt.show()
        plt.savefig("figs/rating_distribution.png", dpi=300)

    plot_stacked_bar_chart(rating_df)

if __name__ == '__main__':
    get_variation()
    # get_frequency()
    # get_rating_distribution()
    # get_length()
    # get_frequency_by_rating()
