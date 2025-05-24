#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义搜索示例应用
Semantic Search Example Application

这个应用展示了语义搜索的各种用法：
- 句子相似度分析
- 问答配对
- 主题聚类
- 语义关联发现
- 多语言语义搜索

This application demonstrates various semantic search use cases:
- Sentence similarity analysis
- Question-answer pairing
- Topic clustering
- Semantic association discovery
- Multilingual semantic search
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import argparse
import logging

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError as e:
    print(f"请安装必要的依赖: {e}")
    print("pip install sentence-transformers scikit-learn matplotlib seaborn")
    sys.exit(1)

from src.basic_vector_search import BasicVectorSearch

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SemanticMatch:
    """语义匹配结果"""
    query: str
    match: str
    similarity: float
    rank: int
    category: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class ClusterInfo:
    """聚类信息"""
    cluster_id: int
    label: str
    size: int
    centroid: np.ndarray
    examples: List[str]
    keywords: List[str]

class SemanticSearchEngine:
    """语义搜索引擎"""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        初始化语义搜索引擎
        
        Args:
            model_name: 预训练模型名称
        """
        self.model_name = model_name
        self.model = None
        self.documents = []
        self.embeddings = None
        self.metadata = []
        self.clusters = []
        
        self._load_model()
    
    def _load_model(self):
        """加载预训练模型"""
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        添加文档到搜索引擎
        
        Args:
            documents: 文档列表
            metadata: 元数据列表
        """
        logger.info(f"正在添加 {len(documents)} 个文档")
        
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in documents])
        
        # 计算新文档的嵌入
        new_embeddings = self.model.encode(documents)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        logger.info(f"文档添加完成，总计 {len(self.documents)} 个文档")
    
    def search(self, query: str, top_k: int = 10, threshold: float = 0.0) -> List[SemanticMatch]:
        """
        语义搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            threshold: 相似度阈值
        
        Returns:
            搜索结果列表
        """
        if self.embeddings is None:
            return []
        
        # 计算查询嵌入
        query_embedding = self.model.encode([query])
        
        # 计算相似度
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 排序并筛选
        indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(indices):
            if similarities[idx] >= threshold:
                result = SemanticMatch(
                    query=query,
                    match=self.documents[idx],
                    similarity=float(similarities[idx]),
                    rank=rank + 1,
                    metadata=self.metadata[idx] if idx < len(self.metadata) else None
                )
                results.append(result)
        
        return results
    
    def find_similar_documents(self, doc_index: int, top_k: int = 5) -> List[SemanticMatch]:
        """
        查找与指定文档相似的其他文档
        
        Args:
            doc_index: 文档索引
            top_k: 返回结果数量
        
        Returns:
            相似文档列表
        """
        if doc_index >= len(self.documents):
            return []
        
        doc_embedding = self.embeddings[doc_index:doc_index+1]
        similarities = cosine_similarity(doc_embedding, self.embeddings)[0]
        
        # 排除自身
        similarities[doc_index] = -1
        
        indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        query_doc = self.documents[doc_index]
        
        for rank, idx in enumerate(indices):
            if similarities[idx] > 0:
                result = SemanticMatch(
                    query=query_doc,
                    match=self.documents[idx],
                    similarity=float(similarities[idx]),
                    rank=rank + 1,
                    metadata=self.metadata[idx] if idx < len(self.metadata) else None
                )
                results.append(result)
        
        return results
    
    def cluster_documents(self, n_clusters: int = 5, min_cluster_size: int = 2) -> List[ClusterInfo]:
        """
        对文档进行主题聚类
        
        Args:
            n_clusters: 聚类数量
            min_cluster_size: 最小聚类大小
        
        Returns:
            聚类信息列表
        """
        if self.embeddings is None or len(self.documents) < n_clusters:
            return []
        
        logger.info(f"正在进行文档聚类，聚类数: {n_clusters}")
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.embeddings)
        
        # 分析聚类结果
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) < min_cluster_size:
                continue
            
            cluster_docs = [self.documents[i] for i in cluster_indices]
            cluster_centroid = kmeans.cluster_centers_[cluster_id]
            
            # 生成聚类标签（使用最靠近中心的文档）
            centroid_embedding = cluster_centroid.reshape(1, -1)
            cluster_embeddings = self.embeddings[cluster_indices]
            similarities = cosine_similarity(centroid_embedding, cluster_embeddings)[0]
            representative_idx = cluster_indices[np.argmax(similarities)]
            cluster_label = self.documents[representative_idx][:50] + "..."
            
            # 提取关键词（简化版）
            keywords = self._extract_keywords(cluster_docs)
            
            cluster_info = ClusterInfo(
                cluster_id=cluster_id,
                label=cluster_label,
                size=len(cluster_indices),
                centroid=cluster_centroid,
                examples=cluster_docs[:5],  # 前5个示例
                keywords=keywords
            )
            clusters.append(cluster_info)
        
        self.clusters = clusters
        logger.info(f"聚类完成，生成 {len(clusters)} 个有效聚类")
        return clusters
    
    def _extract_keywords(self, documents: List[str], max_keywords: int = 5) -> List[str]:
        """
        提取文档关键词（简化版）
        
        Args:
            documents: 文档列表
            max_keywords: 最大关键词数量
        
        Returns:
            关键词列表
        """
        # 简单的关键词提取：统计词频
        word_freq = defaultdict(int)
        
        for doc in documents:
            # 简单分词（可以替换为更好的分词工具）
            words = doc.lower().split()
            for word in words:
                # 过滤短词和标点
                if len(word) > 2 and word.isalpha():
                    word_freq[word] += 1
        
        # 按频次排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def analyze_semantic_relationships(self) -> Dict[str, Any]:
        """
        分析文档间的语义关系
        
        Returns:
            分析结果字典
        """
        if self.embeddings is None:
            return {}
        
        logger.info("正在分析语义关系")
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(self.embeddings)
        
        # 统计分析
        n_docs = len(self.documents)
        avg_similarity = np.mean(similarity_matrix[np.triu_indices(n_docs, k=1)])
        max_similarity = np.max(similarity_matrix[np.triu_indices(n_docs, k=1)])
        min_similarity = np.min(similarity_matrix[np.triu_indices(n_docs, k=1)])
        
        # 找出最相似的文档对
        upper_triangle = np.triu(similarity_matrix, k=1)
        max_idx = np.unravel_index(np.argmax(upper_triangle), upper_triangle.shape)
        most_similar_pair = (
            self.documents[max_idx[0]],
            self.documents[max_idx[1]],
            similarity_matrix[max_idx]
        )
        
        # 找出最不相似的文档对
        min_idx = np.unravel_index(np.argmin(upper_triangle), upper_triangle.shape)
        least_similar_pair = (
            self.documents[min_idx[0]],
            self.documents[min_idx[1]],
            similarity_matrix[min_idx]
        )
        
        analysis = {
            'total_documents': n_docs,
            'average_similarity': float(avg_similarity),
            'max_similarity': float(max_similarity),
            'min_similarity': float(min_similarity),
            'most_similar_pair': {
                'doc1': most_similar_pair[0],
                'doc2': most_similar_pair[1],
                'similarity': float(most_similar_pair[2])
            },
            'least_similar_pair': {
                'doc1': least_similar_pair[0],
                'doc2': least_similar_pair[1],
                'similarity': float(least_similar_pair[2])
            }
        }
        
        return analysis
    
    def visualize_embeddings(self, save_path: Optional[str] = None):
        """
        可视化文档嵌入
        
        Args:
            save_path: 保存路径
        """
        if self.embeddings is None:
            logger.warning("没有嵌入数据可视化")
            return
        
        logger.info("正在生成嵌入可视化")
        
        # 使用PCA降维到2D
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(self.embeddings)
        
        plt.figure(figsize=(12, 8))
        
        # 如果有聚类结果，用不同颜色显示
        if self.clusters:
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.clusters)))
            
            # 重新计算聚类标签用于可视化
            kmeans = KMeans(n_clusters=len(self.clusters), random_state=42)
            cluster_labels = kmeans.fit_predict(self.embeddings)
            
            for i, cluster in enumerate(self.clusters):
                mask = cluster_labels == cluster.cluster_id
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors[i]], label=f'Cluster {i}: {cluster.label[:30]}...', 
                           alpha=0.6, s=50)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)
        
        plt.title('文档嵌入可视化 (PCA 2D)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化已保存到: {save_path}")
        else:
            plt.show()
    
    def save_model(self, save_path: str):
        """保存模型和数据"""
        data = {
            'model_name': self.model_name,
            'documents': self.documents,
            'metadata': self.metadata,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else None,
            'clusters': [asdict(cluster) for cluster in self.clusters]
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型已保存到: {save_path}")
    
    def load_model(self, load_path: str):
        """加载模型和数据"""
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.model_name = data['model_name']
        self.documents = data['documents']
        self.metadata = data['metadata']
        self.embeddings = np.array(data['embeddings']) if data['embeddings'] else None
        
        # 重构聚类信息
        self.clusters = []
        for cluster_data in data['clusters']:
            cluster_data['centroid'] = np.array(cluster_data['centroid'])
            self.clusters.append(ClusterInfo(**cluster_data))
        
        self._load_model()  # 重新加载模型
        logger.info(f"模型已从 {load_path} 加载")

class SemanticSearchDemo:
    """语义搜索演示类"""
    
    def __init__(self):
        self.engine = SemanticSearchEngine()
        self.sample_data = self._load_sample_data()
    
    def _load_sample_data(self) -> Dict[str, List[str]]:
        """加载示例数据"""
        return {
            'chinese_sentences': [
                "今天天气很好，适合出去散步",
                "这部电影非常精彩，值得一看",
                "我喜欢在阳光明媚的日子里读书",
                "这家餐厅的菜品味道很棒",
                "学习编程需要大量的练习和耐心",
                "人工智能正在改变我们的生活方式",
                "音乐能够治愈人的心灵",
                "旅行可以开阔我们的视野",
                "健康的生活方式包括合理饮食和运动",
                "友谊是人生中最宝贵的财富之一"
            ],
            'questions_answers': [
                "什么是机器学习？",
                "机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
                "如何提高编程技能？",
                "通过大量练习、阅读优秀代码和参与项目来提高编程技能。",
                "Python有什么优势？",
                "Python语法简洁、生态丰富、适合快速开发和数据分析。",
                "什么是深度学习？",
                "深度学习是机器学习的一个子领域，使用神经网络来模拟人脑处理信息。",
                "如何学习数据科学？",
                "学习统计学、编程、数据处理和机器学习算法，并进行实际项目练习。"
            ],
            'multilingual': [
                "Hello, how are you today?",
                "你好，今天过得怎么样？",
                "Bonjour, comment allez-vous?",
                "I love programming and artificial intelligence.",
                "我热爱编程和人工智能。",
                "J'aime la programmation et l'intelligence artificielle.",
                "The weather is beautiful today.",
                "今天天气很美好。",
                "Il fait beau aujourd'hui."
            ]
        }
    
    def demo_basic_search(self):
        """演示基础语义搜索"""
        print("\n" + "="*60)
        print("基础语义搜索演示")
        print("="*60)
        
        # 添加中文句子
        self.engine.add_documents(self.sample_data['chinese_sentences'])
        
        queries = [
            "阳光明媚的天气",
            "看电影",
            "学习技术",
            "身体健康"
        ]
        
        for query in queries:
            print(f"\n查询: {query}")
            print("-" * 40)
            
            results = self.engine.search(query, top_k=3)
            for result in results:
                print(f"相似度: {result.similarity:.3f} | {result.match}")
    
    def demo_qa_matching(self):
        """演示问答匹配"""
        print("\n" + "="*60)
        print("问答匹配演示")
        print("="*60)
        
        # 重新初始化引擎
        self.engine = SemanticSearchEngine()
        qa_data = self.sample_data['questions_answers']
        
        # 将问答对分开
        questions = qa_data[::2]  # 偶数索引是问题
        answers = qa_data[1::2]   # 奇数索引是答案
        
        # 添加答案作为文档
        metadata = [{'type': 'answer', 'question': q} for q in questions]
        self.engine.add_documents(answers, metadata)
        
        # 测试问题
        test_questions = [
            "什么是AI？",
            "怎样学编程？",
            "Python的好处是什么？"
        ]
        
        for question in test_questions:
            print(f"\n问题: {question}")
            print("-" * 40)
            
            results = self.engine.search(question, top_k=2)
            for result in results:
                original_q = result.metadata.get('question', 'Unknown')
                print(f"匹配问题: {original_q}")
                print(f"答案: {result.match}")
                print(f"相似度: {result.similarity:.3f}")
                print()
    
    def demo_clustering(self):
        """演示文档聚类"""
        print("\n" + "="*60)
        print("文档聚类演示")
        print("="*60)
        
        # 重新初始化引擎
        self.engine = SemanticSearchEngine()
        
        # 混合所有文档
        all_docs = (self.sample_data['chinese_sentences'] + 
                   self.sample_data['questions_answers'])
        
        self.engine.add_documents(all_docs)
        
        # 执行聚类
        clusters = self.engine.cluster_documents(n_clusters=4)
        
        for cluster in clusters:
            print(f"\n聚类 {cluster.cluster_id} (大小: {cluster.size})")
            print(f"代表文档: {cluster.label}")
            print(f"关键词: {', '.join(cluster.keywords)}")
            print("示例文档:")
            for i, example in enumerate(cluster.examples[:3], 1):
                print(f"  {i}. {example}")
    
    def demo_multilingual_search(self):
        """演示多语言语义搜索"""
        print("\n" + "="*60)
        print("多语言语义搜索演示")
        print("="*60)
        
        # 重新初始化引擎
        self.engine = SemanticSearchEngine()
        self.engine.add_documents(self.sample_data['multilingual'])
        
        queries = [
            "greeting",  # 英文查询
            "问候",      # 中文查询
            "programming",  # 英文查询
            "编程",         # 中文查询
            "weather",      # 英文查询
            "天气"          # 中文查询
        ]
        
        for query in queries:
            print(f"\n查询: {query}")
            print("-" * 40)
            
            results = self.engine.search(query, top_k=3, threshold=0.3)
            for result in results:
                print(f"相似度: {result.similarity:.3f} | {result.match}")
    
    def demo_semantic_analysis(self):
        """演示语义关系分析"""
        print("\n" + "="*60)
        print("语义关系分析演示")
        print("="*60)
        
        # 使用中文句子
        self.engine = SemanticSearchEngine()
        self.engine.add_documents(self.sample_data['chinese_sentences'])
        
        # 分析语义关系
        analysis = self.engine.analyze_semantic_relationships()
        
        print(f"文档总数: {analysis['total_documents']}")
        print(f"平均相似度: {analysis['average_similarity']:.3f}")
        print(f"最高相似度: {analysis['max_similarity']:.3f}")
        print(f"最低相似度: {analysis['min_similarity']:.3f}")
        
        print(f"\n最相似的文档对 (相似度: {analysis['most_similar_pair']['similarity']:.3f}):")
        print(f"  文档1: {analysis['most_similar_pair']['doc1']}")
        print(f"  文档2: {analysis['most_similar_pair']['doc2']}")
        
        print(f"\n最不相似的文档对 (相似度: {analysis['least_similar_pair']['similarity']:.3f}):")
        print(f"  文档1: {analysis['least_similar_pair']['doc1']}")
        print(f"  文档2: {analysis['least_similar_pair']['doc2']}")
    
    def run_interactive_mode(self):
        """运行交互模式"""
        print("\n" + "="*60)
        print("交互式语义搜索")
        print("="*60)
        print("输入 'quit' 退出，'help' 查看帮助")
        
        # 加载所有示例数据
        all_docs = []
        for docs in self.sample_data.values():
            all_docs.extend(docs)
        
        self.engine = SemanticSearchEngine()
        self.engine.add_documents(all_docs)
        
        while True:
            try:
                query = input("\n请输入查询: ").strip()
                
                if query.lower() == 'quit':
                    print("再见！")
                    break
                elif query.lower() == 'help':
                    print("可用命令:")
                    print("  - 直接输入文本进行语义搜索")
                    print("  - 'cluster' - 显示文档聚类")
                    print("  - 'analyze' - 显示语义分析")
                    print("  - 'quit' - 退出程序")
                    continue
                elif query.lower() == 'cluster':
                    clusters = self.engine.cluster_documents(n_clusters=3)
                    for cluster in clusters:
                        print(f"\n聚类 {cluster.cluster_id}: {cluster.label}")
                        print(f"  大小: {cluster.size}, 关键词: {', '.join(cluster.keywords)}")
                    continue
                elif query.lower() == 'analyze':
                    analysis = self.engine.analyze_semantic_relationships()
                    print(f"\n文档总数: {analysis['total_documents']}")
                    print(f"平均相似度: {analysis['average_similarity']:.3f}")
                    continue
                
                if not query:
                    continue
                
                results = self.engine.search(query, top_k=5, threshold=0.1)
                
                if not results:
                    print("没有找到相关结果")
                else:
                    print(f"\n找到 {len(results)} 个相关结果:")
                    for i, result in enumerate(results, 1):
                        print(f"{i}. [相似度: {result.similarity:.3f}] {result.match}")
                
            except KeyboardInterrupt:
                print("\n\n程序被中断，再见！")
                break
            except Exception as e:
                print(f"发生错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='语义搜索示例应用')
    parser.add_argument('--mode', choices=['demo', 'interactive'], default='demo',
                       help='运行模式: demo(演示) 或 interactive(交互)')
    parser.add_argument('--save-model', type=str, help='保存模型到指定路径')
    parser.add_argument('--load-model', type=str, help='从指定路径加载模型')
    parser.add_argument('--visualize', action='store_true', help='生成嵌入可视化')
    
    args = parser.parse_args()
    
    try:
        demo = SemanticSearchDemo()
        
        if args.load_model:
            demo.engine.load_model(args.load_model)
            print(f"已加载模型: {args.load_model}")
        
        if args.mode == 'demo':
            print("语义搜索功能演示")
            print("=" * 60)
            
            demo.demo_basic_search()
            demo.demo_qa_matching()
            demo.demo_clustering()
            demo.demo_multilingual_search()
            demo.demo_semantic_analysis()
            
            if args.visualize:
                # 为可视化准备数据
                all_docs = []
                for docs in demo.sample_data.values():
                    all_docs.extend(docs)
                
                demo.engine = SemanticSearchEngine()
                demo.engine.add_documents(all_docs)
                demo.engine.cluster_documents(n_clusters=4)
                demo.engine.visualize_embeddings('semantic_embeddings.png')
        
        elif args.mode == 'interactive':
            demo.run_interactive_mode()
        
        if args.save_model:
            demo.engine.save_model(args.save_model)
            print(f"模型已保存到: {args.save_model}")
    
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
