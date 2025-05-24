#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感分析教育应用
Sentiment Analysis Educational Application

这个应用展示了情感分析的各种技术和方法：
- 基于词典的情感分析
- 机器学习情感分析
- 深度学习情感分析
- 情感趋势分析
- 情感可视化

This application demonstrates various sentiment analysis techniques:
- Lexicon-based sentiment analysis
- Machine learning sentiment analysis
- Deep learning sentiment analysis
- Sentiment trend analysis
- Sentiment visualization
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import re
import time
from collections import defaultdict, Counter

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.text_vectorizer import TextVectorizer
from src.utils import load_documents

# 尝试导入深度学习库
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("提示: 安装transformers库以使用深度学习情感分析: pip install transformers")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("提示: 安装scikit-learn库以使用机器学习功能: pip install scikit-learn")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class SentimentResult:
    """情感分析结果"""
    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    method: str
    detailed_scores: Optional[Dict[str, float]] = None

class LexiconSentimentAnalyzer:
    """基于词典的情感分析器"""
    
    def __init__(self):
        # 简单的中文情感词典
        self.positive_words = {
            '好', '棒', '赞', '优秀', '完美', '喜欢', '爱', '开心', '快乐', '满意',
            '精彩', '美好', '惊喜', '感谢', '推荐', '值得', '不错', '厉害', '成功', '胜利'
        }
        
        self.negative_words = {
            '坏', '差', '糟糕', '讨厌', '失望', '愤怒', '生气', '难过', '痛苦', '烦恼',
            '问题', '错误', '失败', '垃圾', '无聊', '后悔', '抱怨', '麻烦', '困难', '危险'
        }
        
        # 程度副词权重
        self.intensifiers = {
            '非常': 2.0, '特别': 2.0, '十分': 1.8, '极其': 2.5, '超级': 2.2,
            '很': 1.5, '挺': 1.3, '比较': 1.2, '稍微': 0.8, '有点': 0.9
        }
        
        # 否定词
        self.negations = {'不', '没', '无', '非', '未', '别', '莫', '勿'}
    
    def analyze(self, text: str) -> SentimentResult:
        """分析文本情感"""
        words = list(text)  # 简单分词
        
        positive_score = 0
        negative_score = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # 检查程度副词
            intensifier = 1.0
            if word in self.intensifiers:
                intensifier = self.intensifiers[word]
                i += 1
                if i >= len(words):
                    break
                word = words[i]
            
            # 检查否定词
            negation = False
            if word in self.negations:
                negation = True
                i += 1
                if i >= len(words):
                    break
                word = words[i]
            
            # 检查情感词
            if word in self.positive_words:
                score = 1.0 * intensifier
                if negation:
                    negative_score += score
                else:
                    positive_score += score
            elif word in self.negative_words:
                score = 1.0 * intensifier
                if negation:
                    positive_score += score
                else:
                    negative_score += score
            
            i += 1
        
        # 计算最终分数
        total_score = positive_score - negative_score
        
        if total_score > 0.5:
            sentiment = 'positive'
            confidence = min(total_score / 3.0, 1.0)
        elif total_score < -0.5:
            sentiment = 'negative'
            confidence = min(abs(total_score) / 3.0, 1.0)
        else:
            sentiment = 'neutral'
            confidence = 1.0 - abs(total_score) / 2.0
        
        return SentimentResult(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            method='lexicon',
            detailed_scores={
                'positive': positive_score,
                'negative': negative_score,
                'total': total_score
            }
        )

class MLSentimentAnalyzer:
    """基于机器学习的情感分析器"""
    
    def __init__(self):
        if not HAS_SKLEARN:
            raise ImportError("需要安装scikit-learn库")
        
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = LogisticRegression()
        self.is_trained = False
    
    def create_sample_data(self) -> Tuple[List[str], List[str]]:
        """创建示例训练数据"""
        positive_samples = [
            "这个产品真的很好用，我很满意！",
            "服务态度非常好，推荐大家来",
            "质量不错，值得购买",
            "效果很棒，超出预期",
            "非常开心，完美的体验",
            "很喜欢这个设计，太赞了",
            "物流很快，商品质量也很好",
            "客服很耐心，解决了我的问题",
            "性价比很高，推荐",
            "用起来很舒服，很满意"
        ]
        
        negative_samples = [
            "这个产品质量太差了，很失望",
            "服务态度很差，不推荐",
            "完全不值这个价格",
            "效果很差，浪费钱",
            "很生气，体验很糟糕",
            "设计很丑，不喜欢",
            "物流太慢，商品还有问题",
            "客服态度很差，不解决问题",
            "性价比很低，不建议购买",
            "用起来很不舒服，后悔买了"
        ]
        
        neutral_samples = [
            "这是一个普通的产品",
            "没什么特别的感觉",
            "还可以吧，一般般",
            "价格合理，功能基本够用",
            "没有期待也没有失望",
            "中规中矩的表现",
            "符合描述，没有惊喜",
            "平平常常，能用",
            "还行，不好不坏",
            "一般的体验"
        ]
        
        texts = positive_samples + negative_samples + neutral_samples
        labels = (['positive'] * len(positive_samples) + 
                 ['negative'] * len(negative_samples) + 
                 ['neutral'] * len(neutral_samples))
        
        return texts, labels
    
    def train(self, texts: List[str] = None, labels: List[str] = None):
        """训练模型"""
        if texts is None or labels is None:
            texts, labels = self.create_sample_data()
        
        # 向量化文本
        X = self.vectorizer.fit_transform(texts)
        
        # 训练分类器
        self.classifier.fit(X, labels)
        self.is_trained = True
        
        print(f"模型训练完成，使用 {len(texts)} 个样本")
    
    def analyze(self, text: str) -> SentimentResult:
        """分析文本情感"""
        if not self.is_trained:
            self.train()
        
        # 向量化文本
        X = self.vectorizer.transform([text])
        
        # 预测
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        
        # 获取类别
        classes = self.classifier.classes_
        confidence = max(probabilities)
        
        # 创建详细分数
        detailed_scores = {cls: prob for cls, prob in zip(classes, probabilities)}
        
        return SentimentResult(
            text=text,
            sentiment=prediction,
            confidence=confidence,
            method='machine_learning',
            detailed_scores=detailed_scores
        )

class TransformerSentimentAnalyzer:
    """基于Transformer的情感分析器"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        if not HAS_TRANSFORMERS:
            raise ImportError("需要安装transformers库")
        
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model=model_name,
                return_all_scores=True
            )
            self.model_name = model_name
        except Exception as e:
            print(f"无法加载模型 {model_name}，使用默认模型")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                return_all_scores=True
            )
            self.model_name = "default"
    
    def analyze(self, text: str) -> SentimentResult:
        """分析文本情感"""
        results = self.sentiment_pipeline(text)
        
        # 处理结果
        if isinstance(results[0], list):
            results = results[0]
        
        # 找到最高分数的标签
        best_result = max(results, key=lambda x: x['score'])
        
        # 标准化标签
        label_mapping = {
            'POSITIVE': 'positive',
            'NEGATIVE': 'negative',
            'NEUTRAL': 'neutral',
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral',
            'LABEL_2': 'positive'
        }
        
        sentiment = label_mapping.get(best_result['label'], best_result['label'].lower())
        confidence = best_result['score']
        
        # 创建详细分数
        detailed_scores = {
            label_mapping.get(r['label'], r['label'].lower()): r['score'] 
            for r in results
        }
        
        return SentimentResult(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            method=f'transformer_{self.model_name}',
            detailed_scores=detailed_scores
        )

class SentimentAnalysisApp:
    """情感分析教育应用"""
    
    def __init__(self):
        self.analyzers = {}
        self.results_history = []
        
        # 初始化分析器
        self.analyzers['lexicon'] = LexiconSentimentAnalyzer()
        
        if HAS_SKLEARN:
            self.analyzers['ml'] = MLSentimentAnalyzer()
        
        if HAS_TRANSFORMERS:
            try:
                self.analyzers['transformer'] = TransformerSentimentAnalyzer()
            except Exception as e:
                print(f"无法初始化Transformer分析器: {e}")
    
    def analyze_text(self, text: str, methods: List[str] = None) -> Dict[str, SentimentResult]:
        """使用多种方法分析文本情感"""
        if methods is None:
            methods = list(self.analyzers.keys())
        
        results = {}
        for method in methods:
            if method in self.analyzers:
                try:
                    result = self.analyzers[method].analyze(text)
                    results[method] = result
                except Exception as e:
                    print(f"方法 {method} 分析失败: {e}")
        
        # 保存到历史记录
        self.results_history.append({
            'text': text,
            'results': results,
            'timestamp': time.time()
        })
        
        return results
    
    def batch_analyze(self, texts: List[str], method: str = 'lexicon') -> List[SentimentResult]:
        """批量分析文本"""
        if method not in self.analyzers:
            raise ValueError(f"不支持的方法: {method}")
        
        results = []
        for text in texts:
            try:
                result = self.analyzers[method].analyze(text)
                results.append(result)
            except Exception as e:
                print(f"分析文本 '{text[:50]}...' 失败: {e}")
                results.append(SentimentResult(
                    text=text,
                    sentiment='neutral',
                    confidence=0.0,
                    method=method
                ))
        
        return results
    
    def compare_methods(self, text: str) -> None:
        """比较不同方法的分析结果"""
        results = self.analyze_text(text)
        
        print(f"\n文本: {text}")
        print("=" * 60)
        
        for method, result in results.items():
            print(f"\n{method.upper()} 方法:")
            print(f"  情感: {result.sentiment}")
            print(f"  置信度: {result.confidence:.3f}")
            if result.detailed_scores:
                print("  详细分数:")
                for label, score in result.detailed_scores.items():
                    print(f"    {label}: {score:.3f}")
    
    def visualize_sentiment_distribution(self, texts: List[str], method: str = 'lexicon'):
        """可视化情感分布"""
        results = self.batch_analyze(texts, method)
        
        # 统计情感分布
        sentiment_counts = Counter(result.sentiment for result in results)
        confidence_scores = [result.confidence for result in results]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 情感分布饼图
        axes[0, 0].pie(sentiment_counts.values(), labels=sentiment_counts.keys(), 
                       autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title(f'情感分布 ({method})')
        
        # 2. 置信度分布直方图
        axes[0, 1].hist(confidence_scores, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('置信度分布')
        axes[0, 1].set_xlabel('置信度')
        axes[0, 1].set_ylabel('频次')
        
        # 3. 情感-置信度散点图
        sentiments = [result.sentiment for result in results]
        sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
        sentiment_values = [sentiment_mapping.get(s, 0) for s in sentiments]
        
        axes[1, 0].scatter(sentiment_values, confidence_scores, alpha=0.6)
        axes[1, 0].set_title('情感与置信度关系')
        axes[1, 0].set_xlabel('情感 (-1:负面, 0:中性, 1:正面)')
        axes[1, 0].set_ylabel('置信度')
        
        # 4. 情感强度分析
        positive_confidences = [r.confidence for r in results if r.sentiment == 'positive']
        negative_confidences = [r.confidence for r in results if r.sentiment == 'negative']
        neutral_confidences = [r.confidence for r in results if r.sentiment == 'neutral']
        
        data_to_plot = []
        labels = []
        if positive_confidences:
            data_to_plot.append(positive_confidences)
            labels.append('正面')
        if negative_confidences:
            data_to_plot.append(negative_confidences)
            labels.append('负面')
        if neutral_confidences:
            data_to_plot.append(neutral_confidences)
            labels.append('中性')
        
        if data_to_plot:
            axes[1, 1].boxplot(data_to_plot, labels=labels)
            axes[1, 1].set_title('各情感类别置信度分布')
            axes[1, 1].set_ylabel('置信度')
        
        plt.tight_layout()
        plt.show()
    
    def sentiment_trend_analysis(self, texts: List[str], method: str = 'lexicon'):
        """情感趋势分析"""
        results = self.batch_analyze(texts, method)
        
        # 计算情感分数（正面=1，中性=0，负面=-1）
        sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
        sentiment_scores = []
        
        for result in results:
            base_score = sentiment_mapping.get(result.sentiment, 0)
            weighted_score = base_score * result.confidence
            sentiment_scores.append(weighted_score)
        
        # 计算滑动平均
        window_size = min(5, len(sentiment_scores))
        if window_size > 1:
            moving_avg = []
            for i in range(len(sentiment_scores)):
                start = max(0, i - window_size + 1)
                end = i + 1
                avg = np.mean(sentiment_scores[start:end])
                moving_avg.append(avg)
        else:
            moving_avg = sentiment_scores
        
        # 绘制趋势图
        plt.figure(figsize=(12, 6))
        
        x = range(len(sentiment_scores))
        plt.plot(x, sentiment_scores, 'o-', alpha=0.6, label='原始情感分数')
        plt.plot(x, moving_avg, 'r-', linewidth=2, label=f'滑动平均 (窗口={window_size})')
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.fill_between(x, sentiment_scores, 0, 
                        where=np.array(sentiment_scores) > 0, 
                        color='green', alpha=0.3, label='正面区域')
        plt.fill_between(x, sentiment_scores, 0, 
                        where=np.array(sentiment_scores) < 0, 
                        color='red', alpha=0.3, label='负面区域')
        
        plt.title(f'情感趋势分析 ({method})')
        plt.xlabel('文本序号')
        plt.ylabel('情感分数')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 打印统计信息
        print(f"\n情感趋势统计 ({method}):")
        print(f"平均情感分数: {np.mean(sentiment_scores):.3f}")
        print(f"情感分数标准差: {np.std(sentiment_scores):.3f}")
        print(f"最大情感分数: {np.max(sentiment_scores):.3f}")
        print(f"最小情感分数: {np.min(sentiment_scores):.3f}")
        
        # 趋势方向
        if len(sentiment_scores) > 1:
            trend = np.polyfit(range(len(sentiment_scores)), sentiment_scores, 1)[0]
            if trend > 0.01:
                trend_desc = "上升"
            elif trend < -0.01:
                trend_desc = "下降"
            else:
                trend_desc = "平稳"
            print(f"整体趋势: {trend_desc} (斜率: {trend:.4f})")
    
    def run_interactive_demo(self):
        """运行交互式演示"""
        print("\n🎭 情感分析教育应用")
        print("=" * 50)
        print("可用的分析方法:")
        for i, method in enumerate(self.analyzers.keys(), 1):
            print(f"  {i}. {method}")
        
        while True:
            print("\n选择操作:")
            print("1. 单文本分析")
            print("2. 方法比较")
            print("3. 批量分析")
            print("4. 趋势分析")
            print("5. 加载文档数据")
            print("0. 退出")
            
            choice = input("\n请选择 (0-5): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                text = input("请输入要分析的文本: ").strip()
                if text:
                    method = input(f"选择方法 ({'/'.join(self.analyzers.keys())}): ").strip()
                    if method in self.analyzers:
                        result = self.analyzers[method].analyze(text)
                        print(f"\n分析结果:")
                        print(f"情感: {result.sentiment}")
                        print(f"置信度: {result.confidence:.3f}")
                    else:
                        print("无效的方法")
            
            elif choice == '2':
                text = input("请输入要分析的文本: ").strip()
                if text:
                    self.compare_methods(text)
            
            elif choice == '3':
                print("请输入多个文本，每行一个 (输入空行结束):")
                texts = []
                while True:
                    line = input().strip()
                    if not line:
                        break
                    texts.append(line)
                
                if texts:
                    method = input(f"选择方法 ({'/'.join(self.analyzers.keys())}): ").strip()
                    if method in self.analyzers:
                        self.visualize_sentiment_distribution(texts, method)
                    else:
                        print("无效的方法")
            
            elif choice == '4':
                print("请输入多个文本，每行一个 (输入空行结束):")
                texts = []
                while True:
                    line = input().strip()
                    if not line:
                        break
                    texts.append(line)
                
                if texts:
                    method = input(f"选择方法 ({'/'.join(self.analyzers.keys())}): ").strip()
                    if method in self.analyzers:
                        self.sentiment_trend_analysis(texts, method)
                    else:
                        print("无效的方法")
            
            elif choice == '5':
                try:
                    documents = load_documents('data/sample_documents.json')
                    texts = [doc['content'] for doc in documents[:10]]  # 取前10个文档
                    print(f"加载了 {len(texts)} 个文档")
                    
                    method = input(f"选择分析方法 ({'/'.join(self.analyzers.keys())}): ").strip()
                    if method in self.analyzers:
                        print("正在分析...")
                        self.visualize_sentiment_distribution(texts, method)
                        self.sentiment_trend_analysis(texts, method)
                    else:
                        print("无效的方法")
                except Exception as e:
                    print(f"加载文档失败: {e}")

def main():
    """主函数"""
    print("初始化情感分析应用...")
    
    app = SentimentAnalysisApp()
    
    # 示例文本
    sample_texts = [
        "这个产品真的很棒，我非常满意！",
        "质量太差了，完全不值这个价格",
        "还可以吧，没什么特别的",
        "超级喜欢，推荐大家购买",
        "服务态度很差，很失望",
        "价格合理，功能够用",
        "非常开心，超出了我的期待",
        "有点小问题，但总体还行",
        "完美的体验，五星好评！",
        "不太满意，感觉被骗了"
    ]
    
    print("\n🎯 演示: 批量情感分析")
    print("=" * 40)
    
    # 如果有多个分析器，比较不同方法
    if len(app.analyzers) > 1:
        print("\n比较不同方法的分析结果:")
        test_text = "这个产品真的很棒，我非常满意！"
        app.compare_methods(test_text)
    
    # 可视化情感分布
    print("\n生成情感分析可视化...")
    app.visualize_sentiment_distribution(sample_texts)
    
    # 情感趋势分析
    print("\n进行情感趋势分析...")
    app.sentiment_trend_analysis(sample_texts)
    
    # 运行交互式演示
    app.run_interactive_demo()

if __name__ == "__main__":
    main()
