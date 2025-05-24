#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本分类教育应用
Text Classification Educational Application

这个应用展示了文本分类的各种技术：
- 朴素贝叶斯分类
- 支持向量机分类
- 深度学习分类
- 特征工程
- 模型评估和比较

This application demonstrates various text classification techniques:
- Naive Bayes classification
- Support Vector Machine classification
- Deep learning classification
- Feature engineering
- Model evaluation and comparison
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import re
import time
from collections import defaultdict, Counter
import json

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.text_vectorizer import TextVectorizer
from src.utils import load_documents

# 尝试导入机器学习库
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("提示: 安装scikit-learn库以使用机器学习功能: pip install scikit-learn")

# 尝试导入深度学习库
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("提示: 安装transformers库以使用深度学习分类: pip install transformers")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class ClassificationResult:
    """分类结果"""
    text: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    method: str
    features_used: Optional[List[str]] = None

class NaiveBayesClassifier:
    """朴素贝叶斯分类器"""
    
    def __init__(self, vectorizer_type: str = 'tfidf'):
        if not HAS_SKLEARN:
            raise ImportError("需要安装scikit-learn库")
        
        self.vectorizer_type = vectorizer_type
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        else:
            self.vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 2))
        
        self.classifier = MultinomialNB()
        self.is_trained = False
        self.classes_ = None
    
    def train(self, texts: List[str], labels: List[str]):
        """训练模型"""
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.classes_ = self.classifier.classes_
        self.is_trained = True
        print(f"朴素贝叶斯模型训练完成，使用 {len(texts)} 个样本")
    
    def predict(self, text: str) -> ClassificationResult:
        """预测文本类别"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X = self.vectorizer.transform([text])
        predicted_class = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        
        # 获取特征词
        feature_names = self.vectorizer.get_feature_names_out()
        feature_scores = X.toarray()[0]
        top_features = []
        if len(feature_scores) > 0:
            top_indices = np.argsort(feature_scores)[-10:]
            top_features = [feature_names[i] for i in top_indices if feature_scores[i] > 0]
        
        prob_dict = {cls: prob for cls, prob in zip(self.classes_, probabilities)}
        confidence = max(probabilities)
        
        return ClassificationResult(
            text=text,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            method=f'naive_bayes_{self.vectorizer_type}',
            features_used=top_features
        )

class SVMClassifier:
    """支持向量机分类器"""
    
    def __init__(self, kernel: str = 'linear'):
        if not HAS_SKLEARN:
            raise ImportError("需要安装scikit-learn库")
        
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = SVC(kernel=kernel, probability=True, random_state=42)
        self.kernel = kernel
        self.is_trained = False
        self.classes_ = None
    
    def train(self, texts: List[str], labels: List[str]):
        """训练模型"""
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.classes_ = self.classifier.classes_
        self.is_trained = True
        print(f"SVM模型训练完成，使用 {len(texts)} 个样本，核函数: {self.kernel}")
    
    def predict(self, text: str) -> ClassificationResult:
        """预测文本类别"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X = self.vectorizer.transform([text])
        predicted_class = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        
        prob_dict = {cls: prob for cls, prob in zip(self.classes_, probabilities)}
        confidence = max(probabilities)
        
        return ClassificationResult(
            text=text,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            method=f'svm_{self.kernel}'
        )

class TransformerClassifier:
    """基于Transformer的分类器"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        if not HAS_TRANSFORMERS:
            raise ImportError("需要安装transformers库")
        
        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                return_all_scores=True
            )
            self.model_name = model_name
        except Exception as e:
            print(f"无法加载模型 {model_name}，使用默认模型")
            self.classifier = pipeline(
                "text-classification",
                return_all_scores=True
            )
            self.model_name = "default"
    
    def predict(self, text: str) -> ClassificationResult:
        """预测文本类别"""
        results = self.classifier(text)
        
        if isinstance(results[0], list):
            results = results[0]
        
        # 找到最高分数的标签
        best_result = max(results, key=lambda x: x['score'])
        
        # 创建概率字典
        prob_dict = {result['label']: result['score'] for result in results}
        
        return ClassificationResult(
            text=text,
            predicted_class=best_result['label'],
            confidence=best_result['score'],
            probabilities=prob_dict,
            method=f'transformer_{self.model_name}'
        )

class DatasetGenerator:
    """数据集生成器"""
    
    @staticmethod
    def create_sample_dataset() -> Tuple[List[str], List[str]]:
        """创建示例数据集"""
        # 科技类文本
        tech_texts = [
            "人工智能正在改变我们的生活方式，从智能手机到自动驾驶汽车。",
            "深度学习算法在图像识别领域取得了突破性进展。",
            "云计算技术为企业提供了更灵活的IT解决方案。",
            "区块链技术有望革命性地改变金融行业。",
            "机器学习模型可以帮助医生更准确地诊断疾病。",
            "物联网设备正在连接我们周围的一切。",
            "量子计算可能在未来几十年内改变计算范式。",
            "数据科学家使用统计方法从大数据中提取有价值的信息。"
        ]
        
        # 体育类文本
        sports_texts = [
            "世界杯足球赛是全球最受关注的体育赛事之一。",
            "篮球运动员需要出色的身体素质和团队合作精神。",
            "奥运会汇聚了世界各国最优秀的运动员。",
            "网球比赛需要强大的心理素质和精湛的技术。",
            "马拉松跑步考验运动员的耐力和意志力。",
            "游泳是一项全身性的有氧运动。",
            "羽毛球运动在亚洲国家非常受欢迎。",
            "健身房训练可以帮助人们保持良好的身体状态。"
        ]
        
        # 娱乐类文本
        entertainment_texts = [
            "电影院里播放着最新的好莱坞大片。",
            "音乐会现场观众们热情高涨地为歌手喝彩。",
            "电视剧的剧情跌宕起伏，吸引了大量观众。",
            "游戏开发商推出了全新的虚拟现实游戏。",
            "明星们在红毯上展示最新的时尚造型。",
            "动漫作品深受年轻人的喜爱。",
            "综艺节目为观众带来了欢声笑语。",
            "读书是一种很好的娱乐和学习方式。"
        ]
        
        # 合并数据
        texts = tech_texts + sports_texts + entertainment_texts
        labels = ['科技'] * len(tech_texts) + ['体育'] * len(sports_texts) + ['娱乐'] * len(entertainment_texts)
        
        return texts, labels

class TextClassificationApp:
    """文本分类教育应用"""
    
    def __init__(self):
        self.classifiers = {}
        self.training_data = None
        self.test_data = None
        self.results_history = []
        
        # 初始化分类器
        if HAS_SKLEARN:
            self.classifiers['naive_bayes_tfidf'] = NaiveBayesClassifier('tfidf')
            self.classifiers['naive_bayes_count'] = NaiveBayesClassifier('count')
            self.classifiers['svm_linear'] = SVMClassifier('linear')
            self.classifiers['svm_rbf'] = SVMClassifier('rbf')
        
        if HAS_TRANSFORMERS:
            try:
                self.classifiers['transformer'] = TransformerClassifier()
            except Exception as e:
                print(f"无法初始化Transformer分类器: {e}")
    
    def load_dataset(self, texts: List[str] = None, labels: List[str] = None):
        """加载数据集"""
        if texts is None or labels is None:
            texts, labels = DatasetGenerator.create_sample_dataset()
        
        # 分割训练集和测试集
        if HAS_SKLEARN:
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            self.training_data = (train_texts, train_labels)
            self.test_data = (test_texts, test_labels)
        else:
            # 简单分割
            split_idx = int(len(texts) * 0.8)
            self.training_data = (texts[:split_idx], labels[:split_idx])
            self.test_data = (texts[split_idx:], labels[split_idx:])
        
        print(f"数据集加载完成:")
        print(f"训练集: {len(self.training_data[0])} 个样本")
        print(f"测试集: {len(self.test_data[0])} 个样本")
        print(f"类别: {set(labels)}")
    
    def train_all_classifiers(self):
        """训练所有分类器"""
        if self.training_data is None:
            self.load_dataset()
        
        train_texts, train_labels = self.training_data
        
        for name, classifier in self.classifiers.items():
            if name != 'transformer':  # Transformer不需要训练
                try:
                    print(f"\n正在训练 {name}...")
                    classifier.train(train_texts, train_labels)
                except Exception as e:
                    print(f"训练 {name} 失败: {e}")
    
    def classify_text(self, text: str, methods: List[str] = None) -> Dict[str, ClassificationResult]:
        """分类单个文本"""
        if methods is None:
            methods = list(self.classifiers.keys())
        
        results = {}
        for method in methods:
            if method in self.classifiers:
                try:
                    if method == 'transformer':
                        result = self.classifiers[method].predict(text)
                    else:
                        if hasattr(self.classifiers[method], 'is_trained') and not self.classifiers[method].is_trained:
                            print(f"模型 {method} 尚未训练")
                            continue
                        result = self.classifiers[method].predict(text)
                    results[method] = result
                except Exception as e:
                    print(f"方法 {method} 分类失败: {e}")
        
        return results
    
    def compare_methods(self, text: str):
        """比较不同方法的分类结果"""
        results = self.classify_text(text)
        
        print(f"\n文本: {text}")
        print("=" * 80)
        
        for method, result in results.items():
            print(f"\n{method.upper()}:")
            print(f"  预测类别: {result.predicted_class}")
            print(f"  置信度: {result.confidence:.3f}")
            print(f"  概率分布:")
            for cls, prob in result.probabilities.items():
                print(f"    {cls}: {prob:.3f}")
            
            if result.features_used:
                print(f"  关键特征: {', '.join(result.features_used[:5])}")
    
    def evaluate_performance(self):
        """评估模型性能"""
        if self.test_data is None:
            print("没有测试数据")
            return
        
        test_texts, test_labels = self.test_data
        
        print("\n模型性能评估:")
        print("=" * 60)
        
        for name, classifier in self.classifiers.items():
            if name == 'transformer':
                continue  # 跳过Transformer评估
            
            if not hasattr(classifier, 'is_trained') or not classifier.is_trained:
                continue
            
            try:
                predictions = []
                confidences = []
                
                for text in test_texts:
                    result = classifier.predict(text)
                    predictions.append(result.predicted_class)
                    confidences.append(result.confidence)
                
                if HAS_SKLEARN:
                    accuracy = accuracy_score(test_labels, predictions)
                    print(f"\n{name.upper()}:")
                    print(f"  准确率: {accuracy:.3f}")
                    print(f"  平均置信度: {np.mean(confidences):.3f}")
                    
                    # 详细分类报告
                    print("  分类报告:")
                    report = classification_report(test_labels, predictions)
                    for line in report.split('\n')[2:-3]:  # 跳过头部和尾部
                        if line.strip():
                            print(f"    {line}")
                
            except Exception as e:
                print(f"评估 {name} 失败: {e}")
    
    def visualize_classification_results(self, texts: List[str]):
        """可视化分类结果"""
        if not self.classifiers:
            print("没有可用的分类器")
            return
        
        # 收集所有结果
        all_results = []
        for text in texts:
            results = self.classify_text(text)
            for method, result in results.items():
                if method != 'transformer':  # 排除transformer结果
                    all_results.append({
                        'text': text[:30] + '...' if len(text) > 30 else text,
                        'method': method,
                        'predicted_class': result.predicted_class,
                        'confidence': result.confidence
                    })
        
        if not all_results:
            print("没有分类结果可视化")
            return
        
        df = pd.DataFrame(all_results)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 方法准确度比较
        if 'method' in df.columns:
            method_confidence = df.groupby('method')['confidence'].mean()
            axes[0, 0].bar(range(len(method_confidence)), method_confidence.values)
            axes[0, 0].set_xticks(range(len(method_confidence)))
            axes[0, 0].set_xticklabels(method_confidence.index, rotation=45)
            axes[0, 0].set_title('各方法平均置信度')
            axes[0, 0].set_ylabel('平均置信度')
        
        # 2. 类别分布
        if 'predicted_class' in df.columns:
            class_counts = df['predicted_class'].value_counts()
            axes[0, 1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title('预测类别分布')
        
        # 3. 置信度分布
        if 'confidence' in df.columns:
            axes[1, 0].hist(df['confidence'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('置信度')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].set_title('置信度分布')
        
        # 4. 方法vs类别热力图
        if 'method' in df.columns and 'predicted_class' in df.columns:
            pivot_table = df.pivot_table(
                values='confidence', 
                index='method', 
                columns='predicted_class', 
                aggfunc='mean'
            )
            sns.heatmap(pivot_table, annot=True, fmt='.3f', ax=axes[1, 1])
            axes[1, 1].set_title('方法-类别置信度热力图')
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance_analysis(self, method: str = 'naive_bayes_tfidf'):
        """特征重要性分析"""
        if method not in self.classifiers or method == 'transformer':
            print(f"方法 {method} 不支持特征重要性分析")
            return
        
        classifier = self.classifiers[method]
        if not hasattr(classifier, 'is_trained') or not classifier.is_trained:
            print(f"模型 {method} 尚未训练")
            return
        
        try:
            # 获取特征名称
            feature_names = classifier.vectorizer.get_feature_names_out()
            
            if hasattr(classifier.classifier, 'feature_log_prob_'):
                # 朴素贝叶斯特征重要性
                classes = classifier.classifier.classes_
                
                plt.figure(figsize=(15, 8))
                for i, cls in enumerate(classes):
                    feature_importance = classifier.classifier.feature_log_prob_[i]
                    top_indices = np.argsort(feature_importance)[-10:]
                    top_features = [feature_names[idx] for idx in top_indices]
                    top_scores = feature_importance[top_indices]
                    
                    plt.subplot(1, len(classes), i+1)
                    plt.barh(range(len(top_features)), top_scores)
                    plt.yticks(range(len(top_features)), top_features)
                    plt.title(f'类别: {cls}')
                    plt.xlabel('特征重要性')
                
                plt.tight_layout()
                plt.show()
            
            elif hasattr(classifier.classifier, 'coef_'):
                # SVM特征重要性
                feature_importance = np.abs(classifier.classifier.coef_[0])
                top_indices = np.argsort(feature_importance)[-20:]
                top_features = [feature_names[idx] for idx in top_indices]
                top_scores = feature_importance[top_indices]
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(top_features)), top_scores)
                plt.yticks(range(len(top_features)), top_features)
                plt.title(f'{method} 特征重要性')
                plt.xlabel('重要性分数')
                plt.show()
        
        except Exception as e:
            print(f"特征重要性分析失败: {e}")
    
    def run_interactive_demo(self):
        """运行交互式演示"""
        print("\n📊 文本分类教育应用")
        print("=" * 50)
        print("可用的分类方法:")
        for i, method in enumerate(self.classifiers.keys(), 1):
            print(f"  {i}. {method}")
        
        while True:
            print("\n选择操作:")
            print("1. 训练模型")
            print("2. 单文本分类")
            print("3. 方法比较")
            print("4. 性能评估")
            print("5. 批量分类可视化")
            print("6. 特征重要性分析")
            print("7. 加载自定义数据集")
            print("0. 退出")
            
            choice = input("\n请选择 (0-7): ").strip()
            
            if choice == '0':
                break
            
            elif choice == '1':
                self.train_all_classifiers()
            
            elif choice == '2':
                text = input("请输入要分类的文本: ").strip()
                if text:
                    method = input(f"选择方法 ({'/'.join(self.classifiers.keys())}): ").strip()
                    if method in self.classifiers:
                        try:
                            result = self.classify_text(text, [method])[method]
                            print(f"\n预测类别: {result.predicted_class}")
                            print(f"置信度: {result.confidence:.3f}")
                            print("概率分布:")
                            for cls, prob in result.probabilities.items():
                                print(f"  {cls}: {prob:.3f}")
                        except Exception as e:
                            print(f"分类失败: {e}")
                    else:
                        print("无效的方法")
            
            elif choice == '3':
                text = input("请输入要分析的文本: ").strip()
                if text:
                    self.compare_methods(text)
            
            elif choice == '4':
                self.evaluate_performance()
            
            elif choice == '5':
                print("请输入多个文本，每行一个 (输入空行结束):")
                texts = []
                while True:
                    line = input().strip()
                    if not line:
                        break
                    texts.append(line)
                
                if texts:
                    self.visualize_classification_results(texts)
            
            elif choice == '6':
                method = input(f"选择方法 ({'/'.join([m for m in self.classifiers.keys() if m != 'transformer'])}): ").strip()
                if method in self.classifiers and method != 'transformer':
                    self.feature_importance_analysis(method)
                else:
                    print("无效的方法或方法不支持特征分析")
            
            elif choice == '7':
                try:
                    file_path = input("请输入数据文件路径 (JSON格式): ").strip()
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if isinstance(data, list) and len(data) > 0:
                            if 'text' in data[0] and 'label' in data[0]:
                                texts = [item['text'] for item in data]
                                labels = [item['label'] for item in data]
                                self.load_dataset(texts, labels)
                            else:
                                print("数据格式错误，需要包含 'text' 和 'label' 字段")
                        else:
                            print("数据格式错误")
                    else:
                        print("文件不存在")
                except Exception as e:
                    print(f"加载数据失败: {e}")

def main():
    """主函数"""
    print("初始化文本分类应用...")
    
    app = TextClassificationApp()
    
    # 加载示例数据集
    app.load_dataset()
    
    # 训练模型
    print("\n训练分类模型...")
    app.train_all_classifiers()
    
    # 示例文本
    sample_texts = [
        "最新的人工智能技术将改变医疗诊断的准确性。",
        "足球世界杯决赛吸引了全球数亿观众观看。",
        "这部电影的特效制作非常精彩，故事情节引人入胜。"
    ]
    
    print("\n🎯 演示: 多方法文本分类")
    print("=" * 40)
    
    # 比较不同分类方法
    for text in sample_texts:
        app.compare_methods(text)
        print()
    
    # 性能评估
    print("\n📊 模型性能评估")
    app.evaluate_performance()
    
    # 可视化分析
    if sample_texts:
        print("\n📈 可视化分析")
        app.visualize_classification_results(sample_texts)
    
    # 运行交互式演示
    app.run_interactive_demo()

if __name__ == "__main__":
    main()
