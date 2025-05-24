"""
基础向量搜索实现
Basic Vector Search Implementation

本模块包含向量搜索的基础实现，包括：
- 向量相似度计算
- 简单的向量搜索算法
- 性能比较
"""

import numpy as np
import time
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class BasicVectorSearch:
    """基础向量搜索类"""
    
    def __init__(self):
        self.vectors = None
        self.labels = None
        
    def add_vectors(self, vectors: np.ndarray, labels: Optional[List[str]] = None):
        """添加向量到搜索索引
        
        Args:
            vectors: 向量数组，形状为 (n_vectors, vector_dim)
            labels: 向量标签列表
        """
        self.vectors = np.array(vectors)
        self.labels = labels if labels is not None else [f"Vector_{i}" for i in range(len(vectors))]
        print(f"已添加 {len(vectors)} 个向量到搜索索引")
    
    def cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """计算余弦相似度
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            
        Returns:
            余弦相似度值 (-1 到 1)
        """
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def euclidean_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """计算欧几里得距离
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            
        Returns:
            欧几里得距离值
        """
        return np.sqrt(np.sum((vector1 - vector2) ** 2))
    
    def manhattan_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """计算曼哈顿距离
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            
        Returns:
            曼哈顿距离值
        """
        return np.sum(np.abs(vector1 - vector2))
    
    def search(self, query_vector: np.ndarray, top_k: int = 5, 
               metric: str = "cosine") -> List[Tuple[int, float, str]]:
        """搜索最相似的向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回前k个最相似的结果
            metric: 相似度度量方法 ("cosine", "euclidean", "manhattan")
            
        Returns:
            列表，包含 (索引, 相似度/距离, 标签) 的元组
        """
        if self.vectors is None:
            raise ValueError("请先添加向量到搜索索引")
        
        similarities = []
        
        for i, vector in enumerate(self.vectors):
            if metric == "cosine":
                score = self.cosine_similarity(query_vector, vector)
                similarities.append((i, score, self.labels[i]))
            elif metric == "euclidean":
                score = -self.euclidean_distance(query_vector, vector)  # 负值，距离越小相似度越高
                similarities.append((i, score, self.labels[i]))
            elif metric == "manhattan":
                score = -self.manhattan_distance(query_vector, vector)  # 负值，距离越小相似度越高
                similarities.append((i, score, self.labels[i]))
            else:
                raise ValueError("不支持的度量方法。请使用 'cosine', 'euclidean', 或 'manhattan'")
        
        # 按相似度排序（降序）
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def benchmark_search(self, query_vector: np.ndarray, 
                        metrics: List[str] = ["cosine", "euclidean", "manhattan"]) -> dict:
        """性能基准测试
        
        Args:
            query_vector: 查询向量
            metrics: 要测试的度量方法列表
            
        Returns:
            包含各种度量方法性能结果的字典
        """
        results = {}
        
        for metric in metrics:
            start_time = time.time()
            search_results = self.search(query_vector, metric=metric)
            end_time = time.time()
            
            results[metric] = {
                "search_time": end_time - start_time,
                "results": search_results
            }
        
        return results


def generate_sample_vectors(n_vectors: int = 1000, vector_dim: int = 128) -> Tuple[np.ndarray, List[str]]:
    """生成示例向量数据
    
    Args:
        n_vectors: 向量数量
        vector_dim: 向量维度
        
    Returns:
        向量数组和标签列表
    """
    # 生成随机向量
    vectors = np.random.rand(n_vectors, vector_dim)
    
    # 归一化向量（对余弦相似度很重要）
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # 生成标签
    labels = [f"Document_{i:04d}" for i in range(n_vectors)]
    
    return vectors, labels


def visualize_similarity_comparison(search_results: dict):
    """可视化不同相似度度量的比较结果
    
    Args:
        search_results: benchmark_search 的结果
    """
    metrics = list(search_results.keys())
    search_times = [search_results[metric]["search_time"] for metric in metrics]
    
    plt.figure(figsize=(12, 5))
    
    # 搜索时间比较
    plt.subplot(1, 2, 1)
    plt.bar(metrics, search_times)
    plt.title("搜索时间比较")
    plt.ylabel("时间 (秒)")
    plt.xticks(rotation=45)
    
    # 相似度分数比较
    plt.subplot(1, 2, 2)
    for metric in metrics:
        scores = [result[1] for result in search_results[metric]["results"]]
        plt.plot(range(len(scores)), scores, marker='o', label=metric)
    
    plt.title("相似度分数比较")
    plt.xlabel("排名")
    plt.ylabel("相似度分数")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数 - 演示基础向量搜索"""
    print("=== 基础向量搜索演示 ===\n")
    
    # 1. 生成示例数据
    print("1. 生成示例向量数据...")
    vectors, labels = generate_sample_vectors(n_vectors=1000, vector_dim=128)
    
    # 2. 创建搜索引擎
    print("2. 创建向量搜索引擎...")
    search_engine = BasicVectorSearch()
    search_engine.add_vectors(vectors, labels)
    
    # 3. 生成查询向量
    print("3. 生成查询向量...")
    query_vector = np.random.rand(128)
    query_vector = query_vector / np.linalg.norm(query_vector)  # 归一化
    
    # 4. 执行搜索
    print("4. 执行向量搜索...")
    results = search_engine.search(query_vector, top_k=5, metric="cosine")
    
    print("\n最相似的5个向量（余弦相似度）:")
    for rank, (idx, similarity, label) in enumerate(results, 1):
        print(f"  {rank}. {label}: {similarity:.4f}")
    
    # 5. 性能基准测试
    print("\n5. 性能基准测试...")
    benchmark_results = search_engine.benchmark_search(query_vector)
    
    print("\n性能比较:")
    for metric, result in benchmark_results.items():
        print(f"  {metric}: {result['search_time']:.6f} 秒")
    
    # 6. 可视化结果
    print("\n6. 可视化结果...")
    try:
        visualize_similarity_comparison(benchmark_results)
    except Exception as e:
        print(f"可视化失败: {e}")
        print("请确保已安装 matplotlib")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    main()
