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
from src.utils import load_json_data, save_json

class DocumentSearchEngine:
    """简化的文档搜索引擎"""
    
    def __init__(self):
        self.documents = []
        self.vectorizer = TextVectorizer()
        self.basic_search = None
        self.search_history = []
    
    def load_documents(self, documents_path: str):
        """加载文档数据"""
        print(f"正在加载文档: {documents_path}")
        self.documents = load_json_data(documents_path)
        print(f"成功加载 {len(self.documents)} 个文档")
    
    def build_index(self):
        """构建搜索索引"""
        if not self.documents:
            raise ValueError("请先加载文档数据")
        
        print("正在构建搜索索引...")
        texts = [doc['content'] for doc in self.documents]
        
        # 构建TF-IDF索引
        tfidf_vectors = self.vectorizer.tfidf_vectorize(texts)
        self.basic_search = BasicVectorSearch()
          # 处理稀疏矩阵
        if hasattr(tfidf_vectors, 'toarray'):
            self.basic_search.add_vectors(tfidf_vectors.toarray())
        else:
            self.basic_search.add_vectors(tfidf_vectors)
        
        print("索引构建完成!")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """执行搜索"""
        if not self.basic_search:
            raise ValueError("请先构建索引")
        
        # 向量化查询（使用已训练的TF-IDF向量化器）
        query_vector = self.vectorizer.tfidf_transform_query(query)
          # 执行搜索
        results = self.basic_search.search(query_vector, top_k=top_k, metric='cosine')
        
        # 格式化结果
        formatted_results = []
        for i, result in enumerate(results):
            idx, similarity, label = result  # 解包元组 (index, similarity, label)
            if idx < len(self.documents):
                formatted_result = {
                    'rank': i + 1,
                    'document_id': self.documents[idx]['id'],
                    'title': self.documents[idx]['title'],
                    'content': self.documents[idx]['content'],
                    'category': self.documents[idx]['category'],
                    'similarity': float(similarity)
                }
                formatted_results.append(formatted_result)
        
        return formatted_results
    
    def display_results(self, results: List[Dict[str, Any]], query: str):
        """显示搜索结果"""
        if not results:
            print("未找到相关文档")
            return
        
        print(f"\n搜索查询: '{query}'")
        print(f"找到 {len(results)} 个相关文档:")
        print("-" * 60)
        
        for result in results:
            print(f"排名: {result['rank']}")
            print(f"标题: {result['title']}")
            print(f"类别: {result['category']}")
            print(f"相似度: {result['similarity']:.4f}")
            print(f"内容: {result['content'][:100]}...")
            print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description='文档搜索系统')
    parser.add_argument('--documents', '-d', required=True, help='文档数据文件路径')
    parser.add_argument('--query', '-q', help='搜索查询')
    parser.add_argument('--method', '-m', choices=['keyword', 'semantic', 'hybrid'], 
                       default='keyword', help='搜索方法')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='返回结果数量')
    parser.add_argument('--alpha', '-a', type=float, default=0.5, 
                       help='混合搜索中关键词权重')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='交互式搜索模式')
    parser.add_argument('--benchmark', '-b', action='store_true', 
                       help='基准测试模式')
    
    args = parser.parse_args()
    
    try:
        # 初始化搜索引擎
        search_engine = DocumentSearchEngine()
        
        # 加载文档
        search_engine.load_documents(args.documents)
        
        # 构建索引
        search_engine.build_index()
        
        if args.query:
            # 执行搜索
            results = search_engine.search(args.query, args.top_k)
            search_engine.display_results(results, args.query)
        else:
            print("请提供搜索查询 (--query)")
    
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
