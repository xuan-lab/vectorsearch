#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档搜索系统示例

这个模块演示如何构建一个完整的文档搜索系统，支持多种搜索方式和高级功能。

作者: Vector Search Learning Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict

from src.basic_vector_search import BasicVectorSearch
from src.text_vectorizer import TextVectorizer
from src.advanced_search import FAISSSearch
from src.utils import load_json, save_json, calculate_metrics

class DocumentSearchEngine:
    """
    文档搜索引擎
    
    支持多种搜索方式：
    - 关键词搜索 (TF-IDF)
    - 语义搜索 (Sentence Transformers + FAISS)
    - 混合搜索 (关键词 + 语义)
    """
    
    def __init__(self, index_path: Optional[str] = None):
        """
        初始化搜索引擎
        
        Args:
            index_path: 索引保存路径
        """
        self.documents = []
        self.vectorizer = TextVectorizer()
        self.basic_search = None
        self.faiss_search = None
        self.semantic_basic_search = None
        self.index_path = index_path
        self.search_history = []
        self.has_faiss = False
    
    def load_documents(self, documents_path: str):
        """
        加载文档数据
        
        Args:
            documents_path: 文档文件路径
        """
        print(f"正在加载文档: {documents_path}")
        self.documents = load_json(documents_path)
        print(f"成功加载 {len(self.documents)} 个文档")
        
        # 验证文档格式
        required_fields = ['id', 'title', 'content', 'category']
        for doc in self.documents:
            for field in required_fields:
                if field not in doc:
                    raise ValueError(f"文档缺少必要字段: {field}")
    
    def build_index(self):
        """构建搜索索引"""
        if not self.documents:
            raise ValueError("请先加载文档数据")
        
        print("正在构建搜索索引...")
        
        # 提取文档文本
        texts = [doc['content'] for doc in self.documents]
        
        # 构建TF-IDF索引（用于关键词搜索）
        print("构建TF-IDF索引...")
        start_time = time.time()
        tfidf_vectors = self.vectorizer.tfidf_vectorize(texts)
        self.basic_search = BasicVectorSearch()
        # Convert sparse matrix to dense array if needed
        if hasattr(tfidf_vectors, 'toarray'):
            self.basic_search.add_vectors(tfidf_vectors.toarray())
        else:
            self.basic_search.add_vectors(tfidf_vectors)
        tfidf_time = time.time() - start_time
        print(f"TF-IDF索引构建完成，耗时: {tfidf_time:.2f}秒")
        
        # 构建语义向量索引（用于语义搜索）
        print("构建语义向量索引...")
        start_time = time.time()
        try:
            semantic_vectors = self.vectorizer.sentence_transformer_vectorize(texts)
            print(f"使用Sentence Transformers，向量维度: {semantic_vectors.shape[1]}")
        except Exception as e:
            print(f"Sentence Transformers不可用: {e}")
            print("使用TF-IDF作为备选方案")
            from sklearn.preprocessing import normalize
            # Fix: Handle both sparse and dense matrices
            if hasattr(tfidf_vectors, 'toarray'):
                semantic_vectors = normalize(tfidf_vectors.toarray().astype('float32'))
            else:
                semantic_vectors = normalize(tfidf_vectors.astype('float32'))
        
        # Try to initialize FAISS search if available
        try:
            self.faiss_search = FAISSSearch(vector_dim=semantic_vectors.shape[1])
            self.faiss_search.add_vectors(semantic_vectors)
            self.has_faiss = True
            print(f"FAISS索引构建成功")
        except ImportError:
            print("FAISS不可用，将使用基础向量搜索进行语义搜索")
            self.semantic_basic_search = BasicVectorSearch()
            self.semantic_basic_search.add_vectors(semantic_vectors)
            self.has_faiss = False
        
        semantic_time = time.time() - start_time
        print(f"语义向量索引构建完成，耗时: {semantic_time:.2f}秒")
        
        # 保存索引（如果指定了路径）
        if self.index_path:
            self._save_index()
            
        print("所有索引构建完成!")
        
    def _save_index(self):
        """保存索引到文件"""
        if self.index_path:
            index_dir = Path(self.index_path)
            index_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存FAISS索引（如果可用）
            if hasattr(self, 'has_faiss') and self.has_faiss:
                faiss_path = index_dir / "faiss_index.bin"
                self.faiss_search.save_index(str(faiss_path))
            
            # 保存其他数据
            metadata = {
                'document_count': len(self.documents),
                'tfidf_vocab_size': len(self.vectorizer.tfidf_vectorizer.vocabulary_) if hasattr(self.vectorizer, 'tfidf_vectorizer') else 0,
                'semantic_dim': self.faiss_search.vector_dim if hasattr(self, 'has_faiss') and self.has_faiss else 0,
                'has_faiss': hasattr(self, 'has_faiss') and self.has_faiss
            }
            
            metadata_path = index_dir / "metadata.json"
            save_json(metadata, str(metadata_path))
            print(f"索引已保存到: {self.index_path}")
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        关键词搜索
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        if not self.basic_search:
            raise ValueError("搜索索引未构建，请先调用build_index()")
        
        start_time = time.time()
          # 向量化查询（使用已训练的TF-IDF向量化器）
        query_vector = self.vectorizer.tfidf_transform_query(query)
        
        # 执行搜索
        results = self.basic_search.search(query_vector, top_k=top_k, metric='cosine')
        search_time = time.time() - start_time
        
        # 组织结果
        formatted_results = []
        for i, result in enumerate(results):
            idx = result['index']
            if idx < len(self.documents):
                formatted_result = {
                    'rank': i + 1,
                    'document_id': self.documents[idx]['id'],
                    'title': self.documents[idx]['title'],
                    'content': self.documents[idx]['content'],
                    'category': self.documents[idx]['category'],
                    'similarity': float(result['similarity']),
                    'distance': float(result['distance']),
                    'search_type': 'keyword'
                }
                formatted_results.append(formatted_result)
        
        # 记录搜索历史
        self._record_search(query, 'keyword', len(formatted_results), search_time)
        
        return formatted_results
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        语义搜索
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        if not hasattr(self, 'has_faiss') or (not self.has_faiss and not hasattr(self, 'semantic_basic_search')):
            raise ValueError("搜索索引未构建，请先调用build_index()")
        
        start_time = time.time()
        
        # 向量化查询
        try:
            query_vector = self.vectorizer.sentence_transformer_vectorize([query])
        except:
            from sklearn.preprocessing import normalize
            query_vector = self.vectorizer.tfidf_vectorize([query])
            # Handle both sparse and dense matrices
            if hasattr(query_vector, 'toarray'):
                query_vector = normalize(query_vector.toarray().astype('float32'))
            else:
                query_vector = normalize(query_vector.astype('float32'))
        
        # 执行搜索
        if self.has_faiss:
            distances, indices = self.faiss_search.search(query_vector, top_k)
        else:
            # Use basic vector search as fallback
            results = self.semantic_basic_search.search(query_vector[0], top_k=top_k)
            # Convert to the expected format
            distances = [[result['distance'] for result in results]]
            indices = [[result['index'] for result in results]]
        
        search_time = time.time() - start_time
        
        # 组织结果
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.documents):
                # 将距离转换为相似度（距离越小相似度越高）
                similarity = 1 / (1 + distance)
                
                result = {
                    'rank': i + 1,
                    'document_id': self.documents[idx]['id'],
                    'title': self.documents[idx]['title'],
                    'content': self.documents[idx]['content'],
                    'category': self.documents[idx]['category'],
                    'similarity': float(similarity),
                    'distance': float(distance),
                    'search_type': 'semantic'
                }
                results.append(result)
        
        # 记录搜索历史
        self._record_search(query, 'semantic', len(results), search_time)
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        混合搜索
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            alpha: 关键词搜索权重 (1-alpha为语义搜索权重)
            
        Returns:
            搜索结果列表
        """
        start_time = time.time()
        
        # 获取关键词搜索结果
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # 获取语义搜索结果
        semantic_results = self.semantic_search(query, top_k * 2)
        
        # 合并结果
        combined_scores = defaultdict(float)
        document_data = {}
        
        # 处理关键词搜索结果
        for result in keyword_results:
            doc_id = result['document_id']
            combined_scores[doc_id] += alpha * result['similarity']
            document_data[doc_id] = result
        
        # 处理语义搜索结果
        for result in semantic_results:
            doc_id = result['document_id']
            combined_scores[doc_id] += (1 - alpha) * result['similarity']
            if doc_id not in document_data:
                document_data[doc_id] = result
        
        # 排序并组织最终结果
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        hybrid_results = []
        for i, (doc_id, score) in enumerate(sorted_docs):
            result = document_data[doc_id].copy()
            result.update({
                'rank': i + 1,
                'similarity': float(score),
                'search_type': 'hybrid'
            })
            hybrid_results.append(result)
        
        search_time = time.time() - start_time
        
        # 记录搜索历史
        self._record_search(query, 'hybrid', len(hybrid_results), search_time)
        
        return hybrid_results
    
    def _record_search(self, query: str, search_type: str, result_count: int, search_time: float):
        """记录搜索历史"""
        self.search_history.append({
            'timestamp': time.time(),
            'query': query,
            'search_type': search_type,
            'result_count': result_count,
            'search_time': search_time
        })
    
    def get_search_stats(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        if not self.search_history:
            return {}
        
        total_searches = len(self.search_history)
        avg_time = sum(s['search_time'] for s in self.search_history) / total_searches
        search_types = defaultdict(int)
        
        for search in self.search_history:
            search_types[search['search_type']] += 1
        
        return {
            'total_searches': total_searches,
            'average_search_time': avg_time,
            'search_types': dict(search_types),
            'total_documents': len(self.documents)
        }
    
    def interactive_search(self):
        """交互式搜索界面"""
        print("欢迎使用文档搜索系统!")
        print("支持的搜索类型: keyword, semantic, hybrid")
        print("输入 'quit' 退出，'stats' 查看统计信息")
        print("-" * 50)
        
        while True:
            try:
                query = input("\n请输入搜索查询: ").strip()
                
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'stats':
                    stats = self.get_search_stats()
                    print(f"\n搜索统计:")
                    print(f"总搜索次数: {stats.get('total_searches', 0)}")
                    print(f"平均搜索时间: {stats.get('average_search_time', 0):.3f}秒")
                    print(f"搜索类型分布: {stats.get('search_types', {})}")
                    continue
                elif not query:
                    continue
                
                # 选择搜索类型
                search_type = input("搜索类型 (keyword/semantic/hybrid) [semantic]: ").strip() or 'semantic'
                top_k = int(input("返回结果数量 [5]: ") or 5)
                
                # 执行搜索
                if search_type == 'keyword':
                    results = self.keyword_search(query, top_k)
                elif search_type == 'semantic':
                    results = self.semantic_search(query, top_k)
                elif search_type == 'hybrid':
                    alpha = float(input("关键词权重 (0-1) [0.5]: ") or 0.5)
                    results = self.hybrid_search(query, top_k, alpha)
                else:
                    print("不支持的搜索类型")
                    continue
                
                # 显示结果
                self._display_results(results, query)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"搜索出错: {e}")
        
        print("\n感谢使用文档搜索系统!")
    
    def _display_results(self, results: List[Dict[str, Any]], query: str):
        """显示搜索结果"""
        if not results:
            print("未找到相关文档")
            return
        
        print(f"\n搜索查询: '{query}'")
        print(f"找到 {len(results)} 个相关文档:")
        print("-" * 80)
        
        for result in results:
            print(f"排名: {result['rank']}")
            print(f"标题: {result['title']}")
            print(f"类别: {result['category']}")
            print(f"相似度: {result['similarity']:.4f}")
            print(f"内容摘要: {result['content'][:100]}...")
            print("-" * 80)
    
    def benchmark_search_methods(self, queries: List[str], top_k: int = 10) -> Dict[str, Any]:
        """基准测试不同搜索方法"""
        print(f"开始基准测试，使用 {len(queries)} 个查询...")
        
        results = {
            'keyword': {'times': [], 'result_counts': []},
            'semantic': {'times': [], 'result_counts': []},
            'hybrid': {'times': [], 'result_counts': []}
        }
        
        for i, query in enumerate(queries):
            print(f"测试查询 {i+1}/{len(queries)}: {query}")
            
            # 测试关键词搜索
            start_time = time.time()
            keyword_results = self.keyword_search(query, top_k)
            keyword_time = time.time() - start_time
            results['keyword']['times'].append(keyword_time)
            results['keyword']['result_counts'].append(len(keyword_results))
            
            # 测试语义搜索
            start_time = time.time()
            semantic_results = self.semantic_search(query, top_k)
            semantic_time = time.time() - start_time
            results['semantic']['times'].append(semantic_time)
            results['semantic']['result_counts'].append(len(semantic_results))
            
            # 测试混合搜索
            start_time = time.time()
            hybrid_results = self.hybrid_search(query, top_k)
            hybrid_time = time.time() - start_time
            results['hybrid']['times'].append(hybrid_time)
            results['hybrid']['result_counts'].append(len(hybrid_results))
        
        # 计算统计信息
        benchmark_stats = {}
        for method in results:
            times = results[method]['times']
            counts = results[method]['result_counts']
            
            benchmark_stats[method] = {
                'avg_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'std_time': np.std(times),
                'avg_results': np.mean(counts),
                'total_queries': len(queries)
            }
        
        return benchmark_stats

def main():
    parser = argparse.ArgumentParser(description='文档搜索系统')
    parser.add_argument('--documents', '-d', required=True, help='文档数据文件路径')
    parser.add_argument('--query', '-q', help='搜索查询')
    parser.add_argument('--method', '-m', choices=['keyword', 'semantic', 'hybrid'], 
                       default='semantic', help='搜索方法')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='返回结果数量')
    parser.add_argument('--alpha', '-a', type=float, default=0.5, 
                       help='混合搜索中关键词权重')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='交互式搜索模式')
    parser.add_argument('--benchmark', '-b', action='store_true', 
                       help='基准测试模式')
    
    args = parser.parse_args()
    
    # 初始化搜索引擎
    search_engine = DocumentSearchEngine()
    
    # 加载文档
    search_engine.load_documents(args.documents)
    
    # 构建索引
    search_engine.build_index()
    
    if args.interactive:
        # 交互式模式
        search_engine.interactive_search()
    elif args.benchmark:
        # 基准测试模式
        sample_queries = [
            "machine learning",
            "artificial intelligence",
            "deep learning",
            "neural networks",
            "data science"
        ]
        
        print("运行基准测试...")
        stats = search_engine.benchmark_search_methods(sample_queries)
        
        print("\n基准测试结果:")
        print("-" * 60)
        for method, stat in stats.items():
            print(f"{method.upper()}搜索:")
            print(f"  平均搜索时间: {stat['avg_time']:.4f}秒")
            print(f"  时间范围: {stat['min_time']:.4f} - {stat['max_time']:.4f}秒")
            print(f"  时间标准差: {stat['std_time']:.4f}秒")
            print(f"  平均结果数: {stat['avg_results']:.1f}")
            print()
    else:
        # 单次搜索模式
        if not args.query:
            print("请提供搜索查询 (--query)")
            return
        
        # 执行搜索
        if args.method == 'keyword':
            results = search_engine.keyword_search(args.query, args.top_k)
        elif args.method == 'semantic':
            results = search_engine.semantic_search(args.query, args.top_k)
        elif args.method == 'hybrid':
            results = search_engine.hybrid_search(args.query, args.top_k, args.alpha)
        
        # 显示结果
        search_engine._display_results(results, args.query)
        
        # 显示统计信息
        stats = search_engine.get_search_stats()
        if stats:
            print(f"\n最后一次搜索时间: {stats.get('search_types', {}).get(args.method, 0)} 次")

if __name__ == "__main__":
    main()
