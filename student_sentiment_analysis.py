import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
import re
import json
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# 1. 加载和预处理数据
print("加载数据...")
df = pd.read_excel('student_data.xlsx')
print(f"数据集大小: {df.shape}")

# 2. 数据预处理函数
def preprocess_text(text):
    """清洗文本数据"""
    if not isinstance(text, str):
        return ""
    
    # 转换为小写
    text = text.lower()
    # 删除特殊字符和多余的空格
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("预处理文本数据...")
df['Context_Clean'] = df['Context'].apply(preprocess_text)

# 3. 使用TextBlob进行情感分析
print("使用TextBlob进行情感分析...")
df['textblob_polarity'] = df['Context'].apply(lambda text: TextBlob(text).sentiment.polarity)
df['textblob_subjectivity'] = df['Context'].apply(lambda text: TextBlob(text).sentiment.subjectivity)
df['textblob_sentiment'] = df['textblob_polarity'].apply(lambda score: 'positive' if score > 0.1 
                                                     else ('negative' if score < -0.1 else 'neutral'))

# 4. 使用机器学习方法进行情感分析
print("使用机器学习方法进行情感分析...")
# 将TextBlob情感分类作为目标标签
df['sentiment_label'] = df['textblob_sentiment']

# 特征提取 - 使用TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, min_df=5, stop_words='english')
X = vectorizer.fit_transform(df['Context_Clean'])
y = df['sentiment_label']

# 训练-测试集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练逻辑回归模型
print("训练机器学习模型...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 预测并评估
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"机器学习模型准确率: {accuracy:.2f}")
print("分类报告:")
print(report)

# 5. 使用Hugging Face的预训练情感分析模型
print("使用Transformers进行情感分析...")
try:
    # 使用预训练模型
    sentiment_analyzer = pipeline("sentiment-analysis", 
                               model="distilbert-base-uncased-finetuned-sst-2-english",
                               max_length=512, 
                               truncation=True)
    
    # 由于情感分析可能较慢，我们只处理前50个样本
    sample_size = min(50, len(df))
    sample_indices = np.random.choice(df.index, sample_size, replace=False)
    
    # 应用预训练模型进行情感分析
    transformer_results = []
    for idx in sample_indices:
        text = df.loc[idx, 'Context']
        try:
            # 截断文本以适应模型
            if len(text) > 512:
                text = text[:512]
            result = sentiment_analyzer(text)[0]
            transformer_results.append({
                'index': idx,
                'label': result['label'],
                'score': result['score']
            })
        except Exception as e:
            print(f"处理索引 {idx} 时出错: {e}")
    
    # 将结果添加到数据框
    if transformer_results:
        transformer_df = pd.DataFrame(transformer_results)
        # 合并结果
        for idx, row in transformer_df.iterrows():
            df.loc[row['index'], 'transformer_sentiment'] = row['label']
            df.loc[row['index'], 'transformer_score'] = row['score']
    
except Exception as e:
    print(f"使用Transformers进行情感分析时出错: {e}")
    print("跳过Transformers情感分析...")

# 6. 情感分析结果汇总与可视化
print("生成情感分析统计与可视化...")

# 创建存储图像的目录
if not os.path.exists('sentiment_results'):
    os.makedirs('sentiment_results')

# 6.1 TextBlob情感分布
plt.figure(figsize=(10, 6))
textblob_sentiment_counts = df['textblob_sentiment'].value_counts()
plt.bar(textblob_sentiment_counts.index, textblob_sentiment_counts.values)
plt.title('TextBlob情感分析结果分布')
plt.xlabel('情感类别')
plt.ylabel('数量')
plt.savefig('sentiment_results/textblob_sentiment_distribution.png')
plt.close()

# 6.2 情感得分直方图
plt.figure(figsize=(10, 6))
plt.hist(df['textblob_polarity'], bins=20, alpha=0.7)
plt.title('TextBlob情感极性分数分布')
plt.xlabel('情感极性')
plt.ylabel('频率')
plt.grid(True, alpha=0.3)
plt.savefig('sentiment_results/textblob_polarity_histogram.png')
plt.close()

# 6.3 情感极性与主观性散点图
plt.figure(figsize=(10, 6))
plt.scatter(df['textblob_polarity'], df['textblob_subjectivity'], alpha=0.5)
plt.title('情感极性 vs 主观性')
plt.xlabel('情感极性')
plt.ylabel('主观性')
plt.grid(True)
plt.savefig('sentiment_results/polarity_subjectivity_scatter.png')
plt.close()

# 7. 保存结果到CSV文件
print("保存结果到文件...")
results_df = df[['Context', 'Response', 'textblob_sentiment', 'textblob_polarity', 'textblob_subjectivity']]
if 'transformer_sentiment' in df.columns:
    results_df = results_df.join(df[['transformer_sentiment', 'transformer_score']])

results_df.to_csv('sentiment_results/student_sentiment_results.csv', index=False)

# 8. 创建示例结果报告
print("生成示例结果报告...")

# 8.1 找出最积极和最消极的评论
most_positive_textblob = df.loc[df['textblob_polarity'].idxmax()]['Context']
most_negative_textblob = df.loc[df['textblob_polarity'].idxmin()]['Context']

# 8.2 计算各类情感的统计数据
sentiment_stats = {
    'total_samples': len(df),
    'textblob_stats': {
        'positive': int(df['textblob_sentiment'].value_counts().get('positive', 0)),
        'neutral': int(df['textblob_sentiment'].value_counts().get('neutral', 0)),
        'negative': int(df['textblob_sentiment'].value_counts().get('negative', 0))
    },
    'ml_model': {
        'accuracy': float(accuracy),
        'classification_report': report
    },
    'examples': {
        'most_positive_textblob': most_positive_textblob,
        'most_negative_textblob': most_negative_textblob
    }
}

# 8.3 将统计数据保存为JSON文件
with open('sentiment_results/sentiment_analysis_report.json', 'w') as f:
    json.dump(sentiment_stats, f, indent=4)

print("\n分析完成! 结果保存在sentiment_results目录")
print(f"总样本数: {len(df)}")
print(f"TextBlob情感分布: 积极({sentiment_stats['textblob_stats']['positive']}), 中性({sentiment_stats['textblob_stats']['neutral']}), 消极({sentiment_stats['textblob_stats']['negative']})")
print(f"机器学习模型准确率: {accuracy:.2f}") 