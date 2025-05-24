#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本摘要教育应用
Text Summarization Educational Application

这个应用展示了文本摘要的各种技术：
- 提取式摘要 (Extractive Summarization)
- 抽象式摘要 (Abstractive Summarization)
- 关键词提取
- 主题建模
- 摘要质量评估

This application demonstrates various text summarization techniques:
- Extractive summarization
- Abstractive summarization
- Keyword extraction
- Topic modeling
- Summary quality evaluation
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
import heapq

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.text_vectorizer import TextVectorizer
from src.utils import load_documents

# 尝试导入NLTK
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("提示: 安装NLTK库以获得更好的文本处理: pip install nltk")

# 尝试导入深度学习库
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("提示: 安装transformers库以使用深度学习摘要: pip install transformers")

# 尝试导入scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("提示: 安装scikit-learn库以使用机器学习功能: pip install scikit-learn")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class SummaryResult:
    """摘要结果"""
    original_text: str
    summary: str
    method: str
    compression_ratio: float
    keywords: Optional[List[str]] = None
    key_sentences: Optional[List[str]] = None
    topics: Optional[List[Tuple[str, float]]] = None
    quality_score: Optional[float] = None

class ExtractiveSummarizer:
    """提取式摘要器"""
    
    def __init__(self):
        self.vectorizer = TextVectorizer() if 'TextVectorizer' in globals() else None
        self.stop_words = {'的', '是', '在', '有', '和', '了', '一个', '这个', '那个', '我们', '他们', '它们'}
    
    def _sentence_tokenize(self, text: str) -> List[str]:
        """句子分词"""
        if HAS_NLTK:
            try:
                return sent_tokenize(text)
            except:
                pass
        
        # 简单的中文句子分割
        sentences = re.split(r'[。！？；\n]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_word_frequency(self, text: str) -> Dict[str, float]:
        """计算词频"""
        words = re.findall(r'\w+', text.lower())
        word_freq = {}
        
        for word in words:
            if word not in self.stop_words and len(word) > 1:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 归一化
        max_freq = max(word_freq.values()) if word_freq else 1
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        return word_freq
    
    def _score_sentences(self, sentences: List[str], word_freq: Dict[str, float]) -> List[Tuple[int, float]]:
        """给句子打分"""
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            words = re.findall(r'\w+', sentence.lower())
            score = 0
            word_count = 0
            
            for word in words:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            
            # 句子分数 = 词频分数总和 / 句子中的词数
            if word_count > 0:
                score = score / word_count
            
            # 句子长度惩罚 - 太短的句子分数降低
            if len(sentence) < 20:
                score *= 0.5
            
            sentence_scores.append((i, score))
        
        return sentence_scores
    
    def summarize(self, text: str, num_sentences: int = 3) -> SummaryResult:
        """生成提取式摘要"""
        sentences = self._sentence_tokenize(text)
        
        if len(sentences) <= num_sentences:
            summary = ' '.join(sentences)
            keywords = list(self._calculate_word_frequency(text).keys())[:10]
            
            return SummaryResult(
                original_text=text,
                summary=summary,
                method='extractive',
                compression_ratio=1.0,
                keywords=keywords,
                key_sentences=sentences
            )
        
        # 计算词频
        word_freq = self._calculate_word_frequency(text)
        
        # 给句子打分
        sentence_scores = self._score_sentences(sentences, word_freq)
        
        # 选择得分最高的句子
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = sentence_scores[:num_sentences]
        
        # 按原文顺序排列
        top_sentences.sort(key=lambda x: x[0])
        
        # 生成摘要
        summary_sentences = [sentences[i] for i, _ in top_sentences]
        summary = ' '.join(summary_sentences)
        
        # 提取关键词
        keywords = list(word_freq.keys())[:10]
        
        # 计算压缩比
        compression_ratio = len(summary) / len(text)
        
        return SummaryResult(
            original_text=text,
            summary=summary,
            method='extractive',
            compression_ratio=compression_ratio,
            keywords=keywords,
            key_sentences=summary_sentences
        )

class TfIdfSummarizer:
    """基于TF-IDF的摘要器"""
    
    def __init__(self):
        if not HAS_SKLEARN:
            raise ImportError("需要安装scikit-learn库")
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # 自定义中文停用词
            ngram_range=(1, 2)
        )
    
    def summarize(self, text: str, num_sentences: int = 3) -> SummaryResult:
        """使用TF-IDF生成摘要"""
        # 句子分词
        sentences = re.split(r'[。！？；\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= num_sentences:
            summary = ' '.join(sentences)
            return SummaryResult(
                original_text=text,
                summary=summary,
                method='tfidf',
                compression_ratio=1.0,
                key_sentences=sentences
            )
        
        # 计算TF-IDF矩阵
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
        except ValueError:
            # 如果句子太少或词汇太少，回退到简单摘要
            return ExtractiveSummarizer().summarize(text, num_sentences)
        
        # 计算句子重要性分数
        sentence_scores = []
        for i in range(tfidf_matrix.shape[0]):
            score = np.sum(tfidf_matrix[i].toarray())
            sentence_scores.append((i, score))
        
        # 选择得分最高的句子
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = sentence_scores[:num_sentences]
        
        # 按原文顺序排列
        top_sentences.sort(key=lambda x: x[0])
        
        # 生成摘要
        summary_sentences = [sentences[i] for i, _ in top_sentences]
        summary = ' '.join(summary_sentences)
        
        # 提取关键词
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        keyword_indices = np.argsort(tfidf_scores)[::-1][:10]
        keywords = [feature_names[i] for i in keyword_indices]
        
        compression_ratio = len(summary) / len(text)
        
        return SummaryResult(
            original_text=text,
            summary=summary,
            method='tfidf',
            compression_ratio=compression_ratio,
            keywords=keywords,
            key_sentences=summary_sentences
        )

class TransformerSummarizer:
    """基于Transformer的摘要器"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        if not HAS_TRANSFORMERS:
            raise ImportError("需要安装transformers库")
        
        try:
            self.summarizer = pipeline(
                "summarization", 
                model=model_name,
                tokenizer=model_name
            )
            self.model_name = model_name
        except Exception as e:
            print(f"无法加载模型 {model_name}，尝试使用默认模型")
            try:
                self.summarizer = pipeline("summarization")
                self.model_name = "default"
            except Exception as e2:
                raise ImportError(f"无法初始化摘要模型: {e2}")
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 30) -> SummaryResult:
        """使用Transformer生成摘要"""
        # 处理文本长度限制
        max_input_length = 1024  # BART的最大输入长度
        if len(text) > max_input_length:
            text = text[:max_input_length]
        
        try:
            # 生成摘要
            summary_result = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]
            
            summary = summary_result['summary_text']
            compression_ratio = len(summary) / len(text)
            
            return SummaryResult(
                original_text=text,
                summary=summary,
                method=f'transformer_{self.model_name}',
                compression_ratio=compression_ratio
            )
            
        except Exception as e:
            print(f"Transformer摘要失败: {e}")
            # 回退到提取式摘要
            fallback_summarizer = ExtractiveSummarizer()
            return fallback_summarizer.summarize(text, 3)

class TopicModeling:
    """主题建模"""
    
    def __init__(self, n_topics: int = 5):
        if not HAS_SKLEARN:
            raise ImportError("需要安装scikit-learn库")
        
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=2
        )
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method='batch'
        )
    
    def extract_topics(self, texts: List[str]) -> List[List[Tuple[str, float]]]:
        """提取主题"""
        if len(texts) < self.n_topics:
            return []
        
        try:
            # 向量化文本
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # 训练LDA模型
            self.lda.fit(tfidf_matrix)
            
            # 提取主题词
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(self.lda.components_):
                top_words_idx = topic.argsort()[::-1][:10]
                top_words = [(feature_names[i], topic[i]) for i in top_words_idx]
                topics.append(top_words)
            
            return topics
            
        except Exception as e:
            print(f"主题建模失败: {e}")
            return []

class SummarizationApp:
    """文本摘要教育应用"""
    
    def __init__(self):
        self.summarizers = {}
        self.topic_modeler = None
        
        # 初始化摘要器
        self.summarizers['extractive'] = ExtractiveSummarizer()
        
        if HAS_SKLEARN:
            self.summarizers['tfidf'] = TfIdfSummarizer()
            self.topic_modeler = TopicModeling()
        
        if HAS_TRANSFORMERS:
            try:
                self.summarizers['transformer'] = TransformerSummarizer()
            except Exception as e:
                print(f"无法初始化Transformer摘要器: {e}")
    
    def summarize_text(self, text: str, methods: List[str] = None, **kwargs) -> Dict[str, SummaryResult]:
        """使用多种方法生成摘要"""
        if methods is None:
            methods = list(self.summarizers.keys())
        
        results = {}
        for method in methods:
            if method in self.summarizers:
                try:
                    if method == 'transformer':
                        result = self.summarizers[method].summarize(text, **kwargs)
                    else:
                        num_sentences = kwargs.get('num_sentences', 3)
                        result = self.summarizers[method].summarize(text, num_sentences)
                    results[method] = result
                except Exception as e:
                    print(f"方法 {method} 摘要失败: {e}")
        
        return results
    
    def compare_summarization_methods(self, text: str):
        """比较不同摘要方法"""
        results = self.summarize_text(text)
        
        print(f"\n原文 ({len(text)} 字符):")
        print("=" * 60)
        print(text[:200] + "..." if len(text) > 200 else text)
        
        print(f"\n摘要比较:")
        print("=" * 60)
        
        for method, result in results.items():
            print(f"\n{method.upper()} 方法:")
            print(f"摘要: {result.summary}")
            print(f"压缩比: {result.compression_ratio:.2%}")
            
            if result.keywords:
                print(f"关键词: {', '.join(result.keywords[:5])}")
    
    def analyze_document_collection(self, texts: List[str], titles: List[str] = None):
        """分析文档集合"""
        if titles is None:
            titles = [f"文档{i+1}" for i in range(len(texts))]
        
        print(f"\n📚 文档集合分析 ({len(texts)} 个文档)")
        print("=" * 60)
        
        # 为每个文档生成摘要
        all_summaries = []
        compression_ratios = []
        
        for i, text in enumerate(texts):
            if len(text) > 100:  # 只处理较长的文档
                try:
                    result = self.summarizers['extractive'].summarize(text, 2)
                    all_summaries.append(result.summary)
                    compression_ratios.append(result.compression_ratio)
                    
                    print(f"\n{titles[i]}:")
                    print(f"摘要: {result.summary}")
                    if result.keywords:
                        print(f"关键词: {', '.join(result.keywords[:3])}")
                
                except Exception as e:
                    print(f"文档 {titles[i]} 摘要失败: {e}")
        
        # 主题建模
        if self.topic_modeler and len(texts) >= 3:
            print(f"\n📖 主题分析:")
            print("-" * 40)
            
            topics = self.topic_modeler.extract_topics(texts)
            for i, topic_words in enumerate(topics):
                if topic_words:
                    top_words = [word for word, _ in topic_words[:5]]
                    print(f"主题 {i+1}: {', '.join(top_words)}")
        
        # 统计分析
        if compression_ratios:
            print(f"\n📊 统计信息:")
            print("-" * 40)
            print(f"平均压缩比: {np.mean(compression_ratios):.2%}")
            print(f"压缩比标准差: {np.std(compression_ratios):.2%}")
            print(f"最小压缩比: {np.min(compression_ratios):.2%}")
            print(f"最大压缩比: {np.max(compression_ratios):.2%}")
    
    def visualize_summarization_analysis(self, texts: List[str], titles: List[str] = None):
        """可视化摘要分析"""
        if titles is None:
            titles = [f"文档{i+1}" for i in range(len(texts))]
        
        # 收集数据
        results_data = []
        
        for i, text in enumerate(texts):
            if len(text) > 50:
                results = self.summarize_text(text)
                for method, result in results.items():
                    results_data.append({
                        'document': titles[i],
                        'method': method,
                        'original_length': len(text),
                        'summary_length': len(result.summary),
                        'compression_ratio': result.compression_ratio,
                        'keyword_count': len(result.keywords) if result.keywords else 0
                    })
        
        if not results_data:
            print("没有足够的数据进行可视化")
            return
        
        df = pd.DataFrame(results_data)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 压缩比分布
        if 'compression_ratio' in df.columns:
            df.boxplot(column='compression_ratio', by='method', ax=axes[0, 0])
            axes[0, 0].set_title('各方法压缩比分布')
            axes[0, 0].set_xlabel('方法')
            axes[0, 0].set_ylabel('压缩比')
        
        # 2. 原文长度 vs 摘要长度
        if 'original_length' in df.columns and 'summary_length' in df.columns:
            for method in df['method'].unique():
                method_data = df[df['method'] == method]
                axes[0, 1].scatter(method_data['original_length'], 
                                 method_data['summary_length'], 
                                 label=method, alpha=0.7)
            axes[0, 1].set_xlabel('原文长度')
            axes[0, 1].set_ylabel('摘要长度')
            axes[0, 1].set_title('原文长度 vs 摘要长度')
            axes[0, 1].legend()
        
        # 3. 方法比较
        if len(df['method'].unique()) > 1:
            method_stats = df.groupby('method')['compression_ratio'].agg(['mean', 'std'])
            method_stats['mean'].plot(kind='bar', ax=axes[1, 0], 
                                    yerr=method_stats['std'], capsize=4)
            axes[1, 0].set_title('各方法平均压缩比')
            axes[1, 0].set_ylabel('压缩比')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 关键词数量分布
        if 'keyword_count' in df.columns:
            df['keyword_count'].hist(bins=10, ax=axes[1, 1], alpha=0.7)
            axes[1, 1].set_title('关键词数量分布')
            axes[1, 1].set_xlabel('关键词数量')
            axes[1, 1].set_ylabel('频次')
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_summary_quality(self, original: str, summary: str) -> Dict[str, float]:
        """评估摘要质量"""
        metrics = {}
        
        # 1. 压缩比
        metrics['compression_ratio'] = len(summary) / len(original)
        
        # 2. 信息密度 (关键词保留率)
        original_words = set(re.findall(r'\w+', original.lower()))
        summary_words = set(re.findall(r'\w+', summary.lower()))
        
        if original_words:
            metrics['word_coverage'] = len(summary_words & original_words) / len(original_words)
        else:
            metrics['word_coverage'] = 0.0
        
        # 3. 语义相似度 (如果可用)
        if hasattr(self, 'vectorizer') and HAS_SKLEARN:
            try:
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([original, summary])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                metrics['semantic_similarity'] = similarity
            except:
                metrics['semantic_similarity'] = 0.0
        
        # 4. 可读性分数 (简化版)
        sentence_count = len(re.split(r'[。！？]', summary))
        word_count = len(re.findall(r'\w+', summary))
        if sentence_count > 0:
            metrics['readability'] = word_count / sentence_count
        else:
            metrics['readability'] = 0.0
        
        return metrics
    
    def run_interactive_demo(self):
        """运行交互式演示"""
        print("\n📝 文本摘要教育应用")
        print("=" * 50)
        print("可用的摘要方法:")
        for i, method in enumerate(self.summarizers.keys(), 1):
            print(f"  {i}. {method}")
        
        while True:
            print("\n选择操作:")
            print("1. 单文本摘要")
            print("2. 方法比较")
            print("3. 文档集合分析")
            print("4. 摘要质量评估")
            print("5. 加载示例文档")
            print("0. 退出")
            
            choice = input("\n请选择 (0-5): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                text = input("请输入要摘要的文本: ").strip()
                if text:
                    method = input(f"选择方法 ({'/'.join(self.summarizers.keys())}): ").strip()
                    if method in self.summarizers:
                        result = self.summarizers[method].summarize(text)
                        print(f"\n摘要结果:")
                        print(f"摘要: {result.summary}")
                        print(f"压缩比: {result.compression_ratio:.2%}")
                        if result.keywords:
                            print(f"关键词: {', '.join(result.keywords[:5])}")
                    else:
                        print("无效的方法")
            
            elif choice == '2':
                text = input("请输入要分析的文本: ").strip()
                if text:
                    self.compare_summarization_methods(text)
            
            elif choice == '3':
                print("请输入多个文档，每行一个 (输入空行结束):")
                texts = []
                while True:
                    line = input().strip()
                    if not line:
                        break
                    texts.append(line)
                
                if texts:
                    self.analyze_document_collection(texts)
                    
                    visualize = input("是否进行可视化分析? (y/n): ").strip().lower()
                    if visualize == 'y':
                        self.visualize_summarization_analysis(texts)
            
            elif choice == '4':
                original = input("请输入原文: ").strip()
                summary = input("请输入摘要: ").strip()
                
                if original and summary:
                    metrics = self.evaluate_summary_quality(original, summary)
                    print("\n摘要质量评估:")
                    for metric, value in metrics.items():
                        print(f"{metric}: {value:.3f}")
            
            elif choice == '5':
                try:
                    documents = load_documents('data/sample_documents.json')
                    texts = [doc['content'] for doc in documents[:5]]
                    titles = [doc.get('title', f"文档{i+1}") for i, doc in enumerate(documents[:5])]
                    
                    print(f"加载了 {len(texts)} 个文档")
                    self.analyze_document_collection(texts, titles)
                    
                    visualize = input("是否进行可视化分析? (y/n): ").strip().lower()
                    if visualize == 'y':
                        self.visualize_summarization_analysis(texts, titles)
                        
                except Exception as e:
                    print(f"加载文档失败: {e}")

def main():
    """主函数"""
    print("初始化文本摘要应用...")
    
    app = SummarizationApp()
    
    # 示例文本
    sample_text = """
    人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，
    并生产出一种新的能以人类智能相似的方式作出反应的智能机器。该领域的研究包括机器人、
    语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，
    应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    
    机器学习是人工智能的一个重要分支，它通过算法让计算机从数据中学习和改进性能，
    而无需明确编程。深度学习是机器学习的一个子集，它使用神经网络来模拟人脑的学习过程。
    这些技术在图像识别、语音识别、自然语言处理等领域取得了显著的成就。
    
    随着大数据和云计算技术的发展，人工智能的应用越来越广泛，从智能手机到自动驾驶汽车，
    从智能家居到医疗诊断，人工智能正在改变我们的生活方式。然而，人工智能的发展也带来了
    一些挑战，如就业影响、隐私保护、算法偏见等问题需要我们认真对待。
    """
    
    print("\n🎯 演示: 多方法文本摘要")
    print("=" * 40)
    
    # 比较不同摘要方法
    app.compare_summarization_methods(sample_text)
    
    # 运行交互式演示
    app.run_interactive_demo()

if __name__ == "__main__":
    main()
