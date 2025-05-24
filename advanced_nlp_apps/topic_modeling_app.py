#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主题建模教育应用
Topic Modeling Educational Application

这个应用展示了主题建模的各种技术：
- LDA (Latent Dirichlet Allocation)
- NMF (Non-negative Matrix Factorization)
- LSA (Latent Semantic Analysis)
- 主题可视化
- 文档主题分析

This application demonstrates various topic modeling techniques:
- LDA (Latent Dirichlet Allocation)
- NMF (Non-negative Matrix Factorization)
- LSA (Latent Semantic Analysis)
- Topic visualization
- Document topic analysis
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
from wordcloud import WordCloud

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.text_vectorizer import TextVectorizer
from src.utils import load_documents

# 尝试导入机器学习库
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
    from sklearn.metrics import coherence_score
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("提示: 安装scikit-learn库以使用主题建模功能: pip install scikit-learn")

# 尝试导入gensim
try:
    import gensim
    from gensim import corpora, models
    from gensim.models import LdaModel, HdpModel
    from gensim.models.coherencemodel import CoherenceModel
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
    print("提示: 安装gensim库以获得更好的主题建模: pip install gensim")

# 尝试导入pyLDAvis
try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    HAS_PYLDAVIS = True
except ImportError:
    HAS_PYLDAVIS = False
    print("提示: 安装pyLDAvis库以进行主题可视化: pip install pyldavis")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class Topic:
    """主题数据结构"""
    id: int
    words: List[Tuple[str, float]]  # (词, 权重)
    description: str = ""
    coherence_score: float = 0.0

@dataclass
class DocumentTopic:
    """文档主题分布"""
    document_id: int
    text: str
    topics: List[Tuple[int, float]]  # (主题ID, 概率)
    dominant_topic: int
    dominant_probability: float

class LDATopicModeler:
    """LDA主题建模器"""
    
    def __init__(self, n_topics: int = 5, random_state: int = 42):
        if not HAS_SKLEARN:
            raise ImportError("需要安装scikit-learn库")
        
        self.n_topics = n_topics
        self.random_state = random_state
        self.vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2), 
                                         stop_words='english')
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=20
        )
        self.is_fitted = False
        self.feature_names = None
        self.doc_topic_dist = None
    
    def fit(self, documents: List[str]) -> List[Topic]:
        """训练LDA模型"""
        # 向量化文档
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # 训练LDA模型
        self.lda_model.fit(doc_term_matrix)
        self.doc_topic_dist = self.lda_model.transform(doc_term_matrix)
        self.is_fitted = True
        
        # 提取主题
        topics = self._extract_topics()
        
        print(f"LDA模型训练完成，识别出 {len(topics)} 个主题")
        return topics
    
    def _extract_topics(self) -> List[Topic]:
        """提取主题"""
        topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            # 获取主题中权重最高的词
            top_word_indices = topic.argsort()[-10:][::-1]
            topic_words = [(self.feature_names[i], topic[i]) for i in top_word_indices]
            
            # 生成主题描述
            top_words = [word for word, _ in topic_words[:3]]
            description = f"主题_{topic_idx+1}: {', '.join(top_words)}"
            
            topics.append(Topic(
                id=topic_idx,
                words=topic_words,
                description=description
            ))
        
        return topics
    
    def get_document_topics(self, documents: List[str]) -> List[DocumentTopic]:
        """获取文档主题分布"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        doc_topics = []
        
        for doc_idx, (doc, topic_dist) in enumerate(zip(documents, self.doc_topic_dist)):
            # 获取主题分布
            topics = [(i, prob) for i, prob in enumerate(topic_dist)]
            topics.sort(key=lambda x: x[1], reverse=True)
            
            # 主导主题
            dominant_topic = topics[0][0]
            dominant_prob = topics[0][1]
            
            doc_topics.append(DocumentTopic(
                document_id=doc_idx,
                text=doc,
                topics=topics,
                dominant_topic=dominant_topic,
                dominant_probability=dominant_prob
            ))
        
        return doc_topics

class NMFTopicModeler:
    """NMF主题建模器"""
    
    def __init__(self, n_topics: int = 5, random_state: int = 42):
        if not HAS_SKLEARN:
            raise ImportError("需要安装scikit-learn库")
        
        self.n_topics = n_topics
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.nmf_model = NMF(n_components=n_topics, random_state=random_state)
        self.is_fitted = False
        self.feature_names = None
        self.doc_topic_dist = None
    
    def fit(self, documents: List[str]) -> List[Topic]:
        """训练NMF模型"""
        # 向量化文档
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # 训练NMF模型
        self.doc_topic_dist = self.nmf_model.fit_transform(doc_term_matrix)
        self.is_fitted = True
        
        # 提取主题
        topics = self._extract_topics()
        
        print(f"NMF模型训练完成，识别出 {len(topics)} 个主题")
        return topics
    
    def _extract_topics(self) -> List[Topic]:
        """提取主题"""
        topics = []
        
        for topic_idx, topic in enumerate(self.nmf_model.components_):
            # 获取主题中权重最高的词
            top_word_indices = topic.argsort()[-10:][::-1]
            topic_words = [(self.feature_names[i], topic[i]) for i in top_word_indices]
            
            # 生成主题描述
            top_words = [word for word, _ in topic_words[:3]]
            description = f"主题_{topic_idx+1}: {', '.join(top_words)}"
            
            topics.append(Topic(
                id=topic_idx,
                words=topic_words,
                description=description
            ))
        
        return topics
    
    def get_document_topics(self, documents: List[str]) -> List[DocumentTopic]:
        """获取文档主题分布"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        doc_topics = []
        
        for doc_idx, (doc, topic_dist) in enumerate(zip(documents, self.doc_topic_dist)):
            # 标准化主题分布
            total = np.sum(topic_dist)
            if total > 0:
                topic_dist = topic_dist / total
            
            # 获取主题分布
            topics = [(i, prob) for i, prob in enumerate(topic_dist)]
            topics.sort(key=lambda x: x[1], reverse=True)
            
            # 主导主题
            dominant_topic = topics[0][0]
            dominant_prob = topics[0][1]
            
            doc_topics.append(DocumentTopic(
                document_id=doc_idx,
                text=doc,
                topics=topics,
                dominant_topic=dominant_topic,
                dominant_probability=dominant_prob
            ))
        
        return doc_topics

class LSATopicModeler:
    """LSA主题建模器"""
    
    def __init__(self, n_topics: int = 5, random_state: int = 42):
        if not HAS_SKLEARN:
            raise ImportError("需要安装scikit-learn库")
        
        self.n_topics = n_topics
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.lsa_model = TruncatedSVD(n_components=n_topics, random_state=random_state)
        self.is_fitted = False
        self.feature_names = None
        self.doc_topic_dist = None
    
    def fit(self, documents: List[str]) -> List[Topic]:
        """训练LSA模型"""
        # 向量化文档
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # 训练LSA模型
        self.doc_topic_dist = self.lsa_model.fit_transform(doc_term_matrix)
        self.is_fitted = True
        
        # 提取主题
        topics = self._extract_topics()
        
        print(f"LSA模型训练完成，识别出 {len(topics)} 个主题")
        return topics
    
    def _extract_topics(self) -> List[Topic]:
        """提取主题"""
        topics = []
        
        for topic_idx, topic in enumerate(self.lsa_model.components_):
            # 获取主题中权重最高的词（注意LSA可能有负值）
            top_word_indices = np.argsort(np.abs(topic))[-10:][::-1]
            topic_words = [(self.feature_names[i], abs(topic[i])) for i in top_word_indices]
            
            # 生成主题描述
            top_words = [word for word, _ in topic_words[:3]]
            description = f"主题_{topic_idx+1}: {', '.join(top_words)}"
            
            topics.append(Topic(
                id=topic_idx,
                words=topic_words,
                description=description
            ))
        
        return topics
    
    def get_document_topics(self, documents: List[str]) -> List[DocumentTopic]:
        """获取文档主题分布"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        doc_topics = []
        
        for doc_idx, (doc, topic_dist) in enumerate(zip(documents, self.doc_topic_dist)):
            # LSA可能产生负值，取绝对值并标准化
            topic_dist = np.abs(topic_dist)
            total = np.sum(topic_dist)
            if total > 0:
                topic_dist = topic_dist / total
            
            # 获取主题分布
            topics = [(i, prob) for i, prob in enumerate(topic_dist)]
            topics.sort(key=lambda x: x[1], reverse=True)
            
            # 主导主题
            dominant_topic = topics[0][0]
            dominant_prob = topics[0][1]
            
            doc_topics.append(DocumentTopic(
                document_id=doc_idx,
                text=doc,
                topics=topics,
                dominant_topic=dominant_topic,
                dominant_probability=dominant_prob
            ))
        
        return doc_topics

class TopicModelingApp:
    """主题建模教育应用"""
    
    def __init__(self):
        self.modelers = {}
        self.documents = []
        self.topics = {}
        self.document_topics = {}
        
        # 初始化主题建模器
        if HAS_SKLEARN:
            self.modelers['lda'] = LDATopicModeler()
            self.modelers['nmf'] = NMFTopicModeler()
            self.modelers['lsa'] = LSATopicModeler()
    
    def load_documents(self, documents: List[str] = None):
        """加载文档"""
        if documents is None:
            # 使用示例文档
            documents = self._create_sample_documents()
        
        self.documents = documents
        print(f"加载了 {len(documents)} 个文档")
    
    def _create_sample_documents(self) -> List[str]:
        """创建示例文档"""
        return [
            "人工智能和机器学习正在改变我们的世界。深度学习算法在图像识别和自然语言处理方面取得了重大突破。",
            "足球是世界上最受欢迎的运动之一。世界杯每四年举办一次，吸引了全球数十亿观众观看。",
            "电影工业不断发展，特效技术越来越先进。好莱坞大片在全球票房市场占据重要地位。",
            "新冠疫情对全球经济产生了深远影响。各国政府采取了不同的应对措施来控制病毒传播。",
            "云计算技术为企业提供了灵活的IT解决方案。亚马逊、微软和谷歌是云服务的主要提供商。",
            "环境保护是当今世界面临的重要挑战。气候变化和污染问题需要全球合作来解决。",
            "教育技术的发展使在线学习成为可能。疫情期间，远程教育得到了快速发展。",
            "电子商务改变了人们的购物方式。移动支付和物流配送系统的完善推动了在线购物的普及。",
            "医疗技术的进步提高了疾病诊断和治疗的准确性。基因治疗和精准医学为患者带来了新希望。",
            "数字货币和区块链技术正在重塑金融行业。比特币和其他加密货币的价值波动引起了广泛关注。"
        ]
    
    def train_models(self, n_topics: int = 5):
        """训练所有主题建模器"""
        if not self.documents:
            self.load_documents()
        
        print(f"\n开始训练主题模型，主题数量: {n_topics}")
        
        for name, modeler in self.modelers.items():
            try:
                print(f"\n正在训练 {name.upper()} 模型...")
                modeler.n_topics = n_topics
                if hasattr(modeler, 'lda_model'):
                    modeler.lda_model.n_components = n_topics
                elif hasattr(modeler, 'nmf_model'):
                    modeler.nmf_model.n_components = n_topics
                elif hasattr(modeler, 'lsa_model'):
                    modeler.lsa_model.n_components = n_topics
                
                topics = modeler.fit(self.documents)
                self.topics[name] = topics
                
                doc_topics = modeler.get_document_topics(self.documents)
                self.document_topics[name] = doc_topics
                
            except Exception as e:
                print(f"训练 {name} 模型失败: {e}")
    
    def display_topics(self, method: str = 'lda', n_words: int = 10):
        """显示主题"""
        if method not in self.topics:
            print(f"方法 {method} 的主题尚未生成")
            return
        
        topics = self.topics[method]
        
        print(f"\n{method.upper()} 主题分析结果:")
        print("=" * 60)
        
        for topic in topics:
            print(f"\n{topic.description}")
            print("关键词:")
            for word, weight in topic.words[:n_words]:
                print(f"  {word}: {weight:.4f}")
    
    def compare_methods(self):
        """比较不同方法的主题"""
        if not self.topics:
            print("没有训练好的模型")
            return
        
        print("\n主题建模方法比较:")
        print("=" * 80)
        
        for method, topics in self.topics.items():
            print(f"\n{method.upper()} 方法识别的主题:")
            for topic in topics:
                top_words = [word for word, _ in topic.words[:5]]
                print(f"  主题 {topic.id + 1}: {', '.join(top_words)}")
    
    def analyze_document_topics(self, method: str = 'lda', doc_index: int = None):
        """分析文档主题分布"""
        if method not in self.document_topics:
            print(f"方法 {method} 的文档主题尚未生成")
            return
        
        doc_topics = self.document_topics[method]
        
        if doc_index is not None:
            # 分析特定文档
            if 0 <= doc_index < len(doc_topics):
                doc_topic = doc_topics[doc_index]
                print(f"\n文档 {doc_index + 1}: {doc_topic.text[:100]}...")
                print(f"主导主题: 主题 {doc_topic.dominant_topic + 1} (概率: {doc_topic.dominant_probability:.3f})")
                print("主题分布:")
                for topic_id, prob in doc_topic.topics:
                    if prob > 0.05:  # 只显示概率大于5%的主题
                        print(f"  主题 {topic_id + 1}: {prob:.3f}")
            else:
                print("文档索引超出范围")
        else:
            # 分析所有文档
            print(f"\n{method.upper()} 文档主题分析:")
            print("=" * 60)
            
            for doc_topic in doc_topics:
                print(f"文档 {doc_topic.document_id + 1}: 主导主题 {doc_topic.dominant_topic + 1} "
                      f"(概率: {doc_topic.dominant_probability:.3f})")
    
    def visualize_topics(self, method: str = 'lda'):
        """可视化主题"""
        if method not in self.topics or method not in self.document_topics:
            print(f"方法 {method} 的结果尚未生成")
            return
        
        topics = self.topics[method]
        doc_topics = self.document_topics[method]
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 主题关键词权重
        n_topics_to_show = min(len(topics), 4)
        for i in range(n_topics_to_show):
            row = i // 2
            col = i % 2
            
            topic = topics[i]
            words = [word for word, _ in topic.words[:10]]
            weights = [weight for _, weight in topic.words[:10]]
            
            if row < 2 and col < 2:
                axes[row, col].barh(range(len(words)), weights)
                axes[row, col].set_yticks(range(len(words)))
                axes[row, col].set_yticklabels(words)
                axes[row, col].set_title(f'主题 {i + 1} 关键词')
                axes[row, col].set_xlabel('权重')
        
        # 如果主题数少于4个，隐藏多余的子图
        for i in range(n_topics_to_show, 4):
            row = i // 2
            col = i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # 2. 文档主题分布
        if len(doc_topics) > 0:
            # 统计每个主题的文档数量
            topic_doc_counts = Counter()
            for doc_topic in doc_topics:
                topic_doc_counts[doc_topic.dominant_topic] += 1
            
            # 绘制主题分布
            plt.figure(figsize=(10, 6))
            topics_ids = list(range(len(topics)))
            counts = [topic_doc_counts.get(i, 0) for i in topics_ids]
            
            plt.bar(topics_ids, counts)
            plt.xlabel('主题ID')
            plt.ylabel('文档数量')
            plt.title(f'{method.upper()} 主题文档分布')
            plt.xticks(topics_ids, [f'主题{i+1}' for i in topics_ids])
            plt.show()
    
    def create_topic_wordclouds(self, method: str = 'lda'):
        """创建主题词云"""
        if method not in self.topics:
            print(f"方法 {method} 的主题尚未生成")
            return
        
        topics = self.topics[method]
        
        n_topics = len(topics)
        cols = min(3, n_topics)
        rows = (n_topics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = [axes] if n_topics == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, topic in enumerate(topics):
            # 准备词频数据
            word_freq = {word: weight for word, weight in topic.words[:20]}
            
            if word_freq:
                # 创建词云
                wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    font_path='simhei.ttf',  # 支持中文
                    max_words=50
                ).generate_from_frequencies(word_freq)
                
                if n_topics == 1:
                    axes.imshow(wordcloud, interpolation='bilinear')
                    axes.set_title(f'主题 {i + 1} 词云')
                    axes.axis('off')
                else:
                    axes[i].imshow(wordcloud, interpolation='bilinear')
                    axes[i].set_title(f'主题 {i + 1} 词云')
                    axes[i].axis('off')
        
        # 隐藏多余的子图
        for j in range(len(topics), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def find_optimal_topics(self, max_topics: int = 10, method: str = 'lda'):
        """寻找最优主题数量"""
        if not self.documents:
            self.load_documents()
        
        print(f"\n寻找最优主题数量 (方法: {method})...")
        
        coherence_scores = []
        perplexity_scores = []
        topic_numbers = range(2, max_topics + 1)
        
        for n_topics in topic_numbers:
            try:
                if method == 'lda':
                    modeler = LDATopicModeler(n_topics=n_topics)
                elif method == 'nmf':
                    modeler = NMFTopicModeler(n_topics=n_topics)
                elif method == 'lsa':
                    modeler = LSATopicModeler(n_topics=n_topics)
                else:
                    print(f"不支持的方法: {method}")
                    return
                
                topics = modeler.fit(self.documents)
                
                # 计算困惑度（仅对LDA）
                if method == 'lda' and hasattr(modeler.lda_model, 'perplexity'):
                    doc_term_matrix = modeler.vectorizer.transform(self.documents)
                    perplexity = modeler.lda_model.perplexity(doc_term_matrix)
                    perplexity_scores.append(perplexity)
                else:
                    perplexity_scores.append(0)
                
                # 简单的一致性评分（基于主题内词语相关性）
                coherence = self._calculate_simple_coherence(topics)
                coherence_scores.append(coherence)
                
                print(f"主题数 {n_topics}: 一致性 {coherence:.3f}")
                
            except Exception as e:
                print(f"评估主题数 {n_topics} 失败: {e}")
                coherence_scores.append(0)
                perplexity_scores.append(0)
        
        # 可视化结果
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(topic_numbers, coherence_scores, 'bo-')
        plt.xlabel('主题数量')
        plt.ylabel('一致性分数')
        plt.title('主题一致性 vs 主题数量')
        plt.grid(True)
        
        if method == 'lda' and any(score > 0 for score in perplexity_scores):
            plt.subplot(1, 2, 2)
            valid_perplexity = [(n, p) for n, p in zip(topic_numbers, perplexity_scores) if p > 0]
            if valid_perplexity:
                numbers, scores = zip(*valid_perplexity)
                plt.plot(numbers, scores, 'ro-')
                plt.xlabel('主题数量')
                plt.ylabel('困惑度')
                plt.title('困惑度 vs 主题数量')
                plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 推荐最优主题数
        if coherence_scores:
            optimal_topics = topic_numbers[np.argmax(coherence_scores)]
            print(f"\n推荐的主题数量: {optimal_topics} (一致性分数: {max(coherence_scores):.3f})")
    
    def _calculate_simple_coherence(self, topics: List[Topic]) -> float:
        """计算简单的主题一致性"""
        # 这是一个简化的一致性计算
        # 实际应用中可以使用更复杂的方法
        total_coherence = 0
        
        for topic in topics:
            # 计算主题内词语权重的分布均匀性
            weights = [weight for _, weight in topic.words[:10]]
            if len(weights) > 1:
                # 使用权重的标准差的倒数作为一致性度量
                coherence = 1.0 / (np.std(weights) + 1e-6)
                total_coherence += coherence
        
        return total_coherence / len(topics) if topics else 0
    
    def run_interactive_demo(self):
        """运行交互式演示"""
        print("\n📊 主题建模教育应用")
        print("=" * 50)
        print("可用的主题建模方法:")
        for i, method in enumerate(self.modelers.keys(), 1):
            print(f"  {i}. {method.upper()}")
        
        while True:
            print("\n选择操作:")
            print("1. 加载文档")
            print("2. 训练主题模型")
            print("3. 显示主题")
            print("4. 比较方法")
            print("5. 分析文档主题")
            print("6. 可视化主题")
            print("7. 创建词云")
            print("8. 寻找最优主题数")
            print("9. 加载自定义文档")
            print("0. 退出")
            
            choice = input("\n请选择 (0-9): ").strip()
            
            if choice == '0':
                break
            
            elif choice == '1':
                self.load_documents()
            
            elif choice == '2':
                try:
                    n_topics = int(input("请输入主题数量 (默认5): ").strip() or "5")
                    self.train_models(n_topics)
                except ValueError:
                    print("无效的主题数量")
            
            elif choice == '3':
                method = input(f"选择方法 ({'/'.join(self.modelers.keys())}): ").strip()
                if method in self.topics:
                    n_words = int(input("显示多少个关键词 (默认10): ").strip() or "10")
                    self.display_topics(method, n_words)
                else:
                    print("请先训练模型或选择有效方法")
            
            elif choice == '4':
                self.compare_methods()
            
            elif choice == '5':
                method = input(f"选择方法 ({'/'.join(self.modelers.keys())}): ").strip()
                if method in self.document_topics:
                    doc_idx_str = input("输入文档索引 (留空分析所有文档): ").strip()
                    doc_idx = int(doc_idx_str) - 1 if doc_idx_str else None
                    self.analyze_document_topics(method, doc_idx)
                else:
                    print("请先训练模型或选择有效方法")
            
            elif choice == '6':
                method = input(f"选择方法 ({'/'.join(self.modelers.keys())}): ").strip()
                if method in self.topics:
                    self.visualize_topics(method)
                else:
                    print("请先训练模型或选择有效方法")
            
            elif choice == '7':
                method = input(f"选择方法 ({'/'.join(self.modelers.keys())}): ").strip()
                if method in self.topics:
                    try:
                        self.create_topic_wordclouds(method)
                    except Exception as e:
                        print(f"创建词云失败: {e}")
                        print("提示: 请确保安装了wordcloud库: pip install wordcloud")
                else:
                    print("请先训练模型或选择有效方法")
            
            elif choice == '8':
                method = input(f"选择方法 ({'/'.join(self.modelers.keys())}): ").strip()
                if method in self.modelers:
                    max_topics = int(input("最大主题数 (默认10): ").strip() or "10")
                    self.find_optimal_topics(max_topics, method)
                else:
                    print("无效的方法")
            
            elif choice == '9':
                try:
                    file_path = input("请输入文档文件路径 (TXT或JSON): ").strip()
                    if os.path.exists(file_path):
                        if file_path.endswith('.json'):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            if isinstance(data, list) and len(data) > 0:
                                if isinstance(data[0], dict) and 'content' in data[0]:
                                    documents = [item['content'] for item in data]
                                elif isinstance(data[0], str):
                                    documents = data
                                else:
                                    print("JSON格式错误")
                                    continue
                            else:
                                print("JSON数据为空或格式错误")
                                continue
                        
                        else:  # TXT文件
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            # 按段落分割
                            documents = [para.strip() for para in content.split('\n\n') if para.strip()]
                        
                        self.load_documents(documents)
                    else:
                        print("文件不存在")
                except Exception as e:
                    print(f"加载文档失败: {e}")

def main():
    """主函数"""
    print("初始化主题建模应用...")
    
    app = TopicModelingApp()
    
    # 加载示例文档
    app.load_documents()
    
    # 训练模型
    print("\n训练主题模型...")
    app.train_models(n_topics=3)
    
    print("\n🎯 演示: 主题建模分析")
    print("=" * 40)
    
    # 比较不同方法
    app.compare_methods()
    
    # 显示主题详情
    for method in app.modelers.keys():
        if method in app.topics:
            app.display_topics(method, n_words=5)
    
    # 可视化主题
    print("\n📊 主题可视化")
    for method in app.modelers.keys():
        if method in app.topics:
            print(f"\n{method.upper()} 方法可视化:")
            app.visualize_topics(method)
    
    # 运行交互式演示
    app.run_interactive_demo()

if __name__ == "__main__":
    main()
