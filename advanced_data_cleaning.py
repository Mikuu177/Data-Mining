import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from collections import Counter

# 加载数据
print("加载数据集...")
df = pd.read_excel('student_data.xlsx')
original_df = df.copy()  # 保存原始数据副本
print(f"原始数据集大小: {df.shape}")

# 简化的高级文本清洗函数
def advanced_text_cleaning(text):
    """高级文本清洗，不依赖NLTK"""
    if not isinstance(text, str):
        return ""
    
    # 转换为小写
    text = text.lower()
    
    # 删除HTML标签
    text = re.sub(r'<.*?>', '', text)
    
    # 删除URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 替换标点符号为空格
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 删除数字
    text = re.sub(r'\d+', '', text)
    
    # 删除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 简单的英文停用词列表
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                  'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                  'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
                  'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                  'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
                  'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
                  'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                  'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                  'with', 'about', 'against', 'between', 'into', 'through', 'during', 
                  'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 
                  'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                  'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                  'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                  'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
                  's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}
    
    # 移除停用词
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    # 重新组合为文本
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text

# 情感特征提取函数
def extract_sentiment_features(text):
    """从文本中提取多种情感特征"""
    features = {}
    
    # TextBlob情感
    blob = TextBlob(text)
    features['polarity'] = blob.sentiment.polarity
    features['subjectivity'] = blob.sentiment.subjectivity
    
    # 文本长度特征
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = features['char_count'] / (features['word_count'] + 1)  # 避免除零
    
    # 特殊字符特征
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['uppercase_word_count'] = sum(1 for word in text.split() if word.isupper())
    
    # 词汇丰富度
    if features['word_count'] > 0:
        features['lexical_diversity'] = len(set(text.split())) / features['word_count']
    else:
        features['lexical_diversity'] = 0
        
    return features

# 应用高级清洗
print("应用高级文本清洗...")
df['Context_Clean'] = df['Context'].apply(lambda x: advanced_text_cleaning(x))
df['Response_Clean'] = df['Response'].apply(lambda x: advanced_text_cleaning(x))

# 检查清洗结果
print(f"清洗后的样本数量: {len(df)}")
print("\n清洗前后对比:")
for i in range(2):
    print(f"\n原始文本 {i+1}: {df.iloc[i]['Context'][:100]}...")
    print(f"清洗后文本 {i+1}: {df.iloc[i]['Context_Clean'][:100]}...")

# 提取高级特征
print("\n提取情感和文本特征...")
sentiment_features = df['Context'].apply(extract_sentiment_features)
sentiment_df = pd.DataFrame(sentiment_features.tolist())

# 合并特征到原始数据框
df = pd.concat([df, sentiment_df], axis=1)

# 检测和处理异常值
print("\n检测情感极性异常值...")
polarity_mean = df['polarity'].mean()
polarity_std = df['polarity'].std()
polarity_threshold = 2  # 标准差倍数
upper_bound = polarity_mean + polarity_threshold * polarity_std
lower_bound = polarity_mean - polarity_threshold * polarity_std

# 标记异常值
df['polarity_outlier'] = ((df['polarity'] > upper_bound) | (df['polarity'] < lower_bound))
outlier_count = df['polarity_outlier'].sum()
print(f"检测到 {outlier_count} 个情感极性异常值")

# 异常值处理 - 重新设置情感极性值为极限值
df.loc[df['polarity'] > upper_bound, 'polarity'] = upper_bound
df.loc[df['polarity'] < lower_bound, 'polarity'] = lower_bound

# 特征可视化
print("\n生成数据清洗和特征可视化...")
plt.figure(figsize=(10, 6))
sns.histplot(df['polarity'], kde=True)
plt.axvline(upper_bound, color='r', linestyle='--', label=f"Upper Bound ({upper_bound:.2f})")
plt.axvline(lower_bound, color='r', linestyle='--', label=f"Lower Bound ({lower_bound:.2f})")
plt.title('Text Polarity Distribution')
plt.xlabel('Polarity Score')
plt.ylabel('Count')
plt.legend()
plt.savefig('polarity_distribution.png')
plt.close()

# 特征相关性
plt.figure(figsize=(12, 10))
corr_matrix = df[['polarity', 'subjectivity', 'char_count', 'word_count', 
                 'avg_word_length', 'exclamation_count', 'question_count', 
                 'uppercase_word_count', 'lexical_diversity']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('feature_correlation.png')
plt.close()

# 词频分析
print("\n分析高频词...")
all_words = ' '.join(df['Context_Clean']).split()
word_counts = Counter(all_words)
most_common = word_counts.most_common(20)
print(f"20个最常见词: {most_common}")

# 生成最终清洗后的数据集
print("\n生成清洗后的数据集...")
# 定义情感标签，使用更精细的阈值
df['sentiment_label'] = df['polarity'].apply(
    lambda x: 'positive' if x > 0.2 else ('negative' if x < -0.2 else 'neutral'))

# 统计不同情感标签的样本数
sentiment_counts = df['sentiment_label'].value_counts()
print("\n情感标签分布:")
print(sentiment_counts)

# 保存处理后的数据
cleaned_df = df[['Context', 'Response', 'Context_Clean', 'Response_Clean', 
                 'polarity', 'subjectivity', 'sentiment_label',
                 'char_count', 'word_count', 'lexical_diversity']]
cleaned_df.to_csv('cleaned_student_data.csv', index=False)
print("\n清洗后的数据已保存到 cleaned_student_data.csv")

# 绘制清洗前后的情感分布对比
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment_label', data=df)
plt.title('Sentiment Distribution After Advanced Cleaning')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('sentiment_distribution_cleaned.png')
plt.close()

print("\n数据清洗和特征提取完成!")