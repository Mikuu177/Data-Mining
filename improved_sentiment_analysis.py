import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 创建结果目录
if not os.path.exists('model_results'):
    os.makedirs('model_results')

# 加载清洗后的数据
print("加载清洗后的数据...")
df = pd.read_csv('cleaned_student_data.csv')
print(f"数据集大小: {df.shape}")

# 查看情感标签分布
sentiment_counts = df['sentiment_label'].value_counts()
print("\n情感标签分布:")
print(sentiment_counts)

# 特征工程 - 将文本和数值特征结合
X_text = df['Context_Clean']  # 清洗后的文本
X_features = df[['subjectivity', 'char_count', 'word_count', 
                'lexical_diversity', 'polarity']]  # 数值特征
y = df['sentiment_label']  # 目标标签

# 1. 仅使用文本特征的基线模型
print("\n构建基线模型 (仅文本特征)...")
# 划分训练集和测试集
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.3, random_state=42, stratify=y)

# TF-IDF特征提取
tfidf_vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

# 训练基线模型 - 逻辑回归
baseline_model = LogisticRegression(max_iter=1000, class_weight='balanced')
baseline_model.fit(X_train_tfidf, y_train)

# 评估基线模型
baseline_predictions = baseline_model.predict(X_test_tfidf)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)
baseline_report = classification_report(y_test, baseline_predictions)

print(f"基线模型准确率: {baseline_accuracy:.4f}")
print("基线模型分类报告:")
print(baseline_report)

# 2. 混合特征模型（文本 + 数值特征）
print("\n构建混合特征模型...")
# 转换数值特征
X_train_features, X_test_features = X_features.iloc[X_train_text.index], X_features.iloc[X_test_text.index]

# 标准化数值特征
scaler = StandardScaler()
X_train_features_scaled = scaler.fit_transform(X_train_features)
X_test_features_scaled = scaler.transform(X_test_features)

# 合并TF-IDF和数值特征
X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_features_scaled))
X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test_features_scaled))

# 训练混合特征模型
hybrid_model = LogisticRegression(max_iter=1000, class_weight='balanced')
hybrid_model.fit(X_train_combined, y_train)

# 评估混合特征模型
hybrid_predictions = hybrid_model.predict(X_test_combined)
hybrid_accuracy = accuracy_score(y_test, hybrid_predictions)
hybrid_report = classification_report(y_test, hybrid_predictions)

print(f"混合特征模型准确率: {hybrid_accuracy:.4f}")
print("混合特征模型分类报告:")
print(hybrid_report)

# 3. 多模型比较
print("\n比较多种机器学习模型...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'SVM': SVC(kernel='linear', class_weight='balanced'),
    'Naive Bayes': MultinomialNB()
}

# 对每个模型进行训练和评估
results = {}
for name, model in models.items():
    if name == 'Naive Bayes':  # 朴素贝叶斯只能用于离散特征，因此只使用文本特征
        model.fit(X_train_tfidf, y_train)
        predictions = model.predict(X_test_tfidf)
    else:  # 其他模型使用组合特征
        model.fit(X_train_combined, y_train)
        predictions = model.predict(X_test_combined)
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    results[name] = {
        'accuracy': accuracy,
        'report': report,
        'predictions': predictions
    }
    print(f"{name} 准确率: {accuracy:.4f}")

# 保存最佳模型
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = models[best_model_name]
print(f"\n最佳模型是 {best_model_name}，准确率: {results[best_model_name]['accuracy']:.4f}")
print(f"分类报告:\n{classification_report(y_test, results[best_model_name]['predictions'])}")

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('model_results/confusion_matrix.png')
plt.close()

# 可视化各模型性能比较
accuracies = [results[model]['accuracy'] for model in models]
model_names = list(models.keys())

plt.figure(figsize=(12, 6))
sns.barplot(x=model_names, y=accuracies)
plt.title('Model Comparison - Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_results/model_comparison.png')
plt.close()

# 特征重要性 (仅用于随机森林和梯度提升模型)
if 'Random Forest' in results:
    # 提取特征名称
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    feature_names = list(tfidf_feature_names) + list(X_features.columns)
    
    # 获取特征重要性
    rf_model = models['Random Forest']
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[-20:]  # 前20个重要特征
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances (Random Forest)')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('model_results/feature_importance.png')
        plt.close()

# 4. 交叉验证评估最佳模型
print("\n对最佳模型进行交叉验证...")
if best_model_name == 'Naive Bayes':
    # 创建文本处理管道
    cv_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7)),
        ('model', best_model)
    ])
    X_cv = X_text
else:
    # 需要更复杂的方法来组合特征
    # 简化处理：仅使用文本特征进行交叉验证
    cv_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7)),
        ('model', best_model)
    ])
    X_cv = X_text

# 执行5折交叉验证
cv_scores = cross_val_score(cv_pipeline, X_cv, y, cv=5, scoring='accuracy')
print(f"交叉验证平均准确率: {np.mean(cv_scores):.4f}")
print(f"交叉验证标准差: {np.std(cv_scores):.4f}")

# 保存最佳模型
joblib.dump(best_model, 'model_results/best_sentiment_model.pkl')
joblib.dump(tfidf_vectorizer, 'model_results/tfidf_vectorizer.pkl')
if best_model_name != 'Naive Bayes':
    joblib.dump(scaler, 'model_results/feature_scaler.pkl')

print("\n分析完成! 结果保存在model_results目录") 