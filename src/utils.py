"""
工具函数模块
Utility Functions Module

本模块包含各种辅助函数：
- 数据加载和保存
- 向量操作
- 可视化工具
- 性能测量
"""

import numpy as np
import pandas as pd
import json
import pickle
import time
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_json_data(filepath: str) -> List[Dict[str, Any]]:
    """加载 JSON 数据文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        JSON 数据列表
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json_data(data: List[Dict[str, Any]], filepath: str):
    """保存数据为 JSON 文件
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_vectors(filepath: str) -> Tuple[np.ndarray, List[str]]:
    """加载向量和标签
    
    Args:
        filepath: 文件路径
        
    Returns:
        向量数组和标签列表
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['vectors'], data['labels']


def save_vectors(vectors: np.ndarray, labels: List[str], filepath: str):
    """保存向量和标签
    
    Args:
        vectors: 向量数组
        labels: 标签列表
        filepath: 文件路径
    """
    data = {'vectors': vectors, 'labels': labels}
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def normalize_vectors(vectors: np.ndarray, method: str = "l2") -> np.ndarray:
    """向量归一化
    
    Args:
        vectors: 向量数组
        method: 归一化方法 ("l2", "l1", "max")
        
    Returns:
        归一化后的向量
    """
    if method == "l2":
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    elif method == "l1":
        return vectors / np.sum(np.abs(vectors), axis=1, keepdims=True)
    elif method == "max":
        return vectors / np.max(np.abs(vectors), axis=1, keepdims=True)
    else:
        raise ValueError(f"不支持的归一化方法: {method}")


def calculate_vector_stats(vectors: np.ndarray) -> Dict[str, float]:
    """计算向量统计信息
    
    Args:
        vectors: 向量数组
        
    Returns:
        统计信息字典
    """
    stats = {
        'n_vectors': len(vectors),
        'vector_dim': vectors.shape[1],
        'mean': np.mean(vectors),
        'std': np.std(vectors),
        'min': np.min(vectors),
        'max': np.max(vectors),
        'l2_norm_mean': np.mean(np.linalg.norm(vectors, axis=1)),
        'l2_norm_std': np.std(np.linalg.norm(vectors, axis=1))
    }
    return stats


def compute_similarity_matrix(vectors: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """计算向量相似度矩阵
    
    Args:
        vectors: 向量数组
        metric: 相似度度量 ("cosine", "euclidean", "manhattan")
        
    Returns:
        相似度矩阵
    """
    n = len(vectors)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if metric == "cosine":
                sim = np.dot(vectors[i], vectors[j]) / (
                    np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])
                )
            elif metric == "euclidean":
                sim = -np.linalg.norm(vectors[i] - vectors[j])  # 负距离作为相似度
            elif metric == "manhattan":
                sim = -np.sum(np.abs(vectors[i] - vectors[j]))  # 负距离作为相似度
            else:
                raise ValueError(f"不支持的度量方法: {metric}")
            
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    
    return similarity_matrix


def plot_vector_distribution(vectors: np.ndarray, title: str = "向量分布"):
    """绘制向量分布图
    
    Args:
        vectors: 向量数组
        title: 图表标题
    """
    # 计算每个向量的 L2 范数
    norms = np.linalg.norm(vectors, axis=1)
    
    # 计算每个维度的均值和标准差
    dim_means = np.mean(vectors, axis=0)
    dim_stds = np.std(vectors, axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)
    
    # 向量范数分布
    axes[0, 0].hist(norms, bins=50, alpha=0.7)
    axes[0, 0].set_title('向量 L2 范数分布')
    axes[0, 0].set_xlabel('L2 范数')
    axes[0, 0].set_ylabel('频次')
    
    # 维度均值
    axes[0, 1].plot(dim_means)
    axes[0, 1].set_title('各维度均值')
    axes[0, 1].set_xlabel('维度')
    axes[0, 1].set_ylabel('均值')
    
    # 维度标准差
    axes[1, 0].plot(dim_stds)
    axes[1, 0].set_title('各维度标准差')
    axes[1, 0].set_xlabel('维度')
    axes[1, 0].set_ylabel('标准差')
    
    # 向量值分布
    axes[1, 1].hist(vectors.flatten(), bins=50, alpha=0.7)
    axes[1, 1].set_title('向量值分布')
    axes[1, 1].set_xlabel('向量值')
    axes[1, 1].set_ylabel('频次')
    
    plt.tight_layout()
    plt.show()


def plot_similarity_heatmap(similarity_matrix: np.ndarray, labels: Optional[List[str]] = None,
                           max_display: int = 20):
    """绘制相似度热力图
    
    Args:
        similarity_matrix: 相似度矩阵
        labels: 标签列表
        max_display: 最大显示数量
    """
    # 如果矩阵太大，只显示前 max_display 个
    if len(similarity_matrix) > max_display:
        similarity_matrix = similarity_matrix[:max_display, :max_display]
        if labels:
            labels = labels[:max_display]
    
    plt.figure(figsize=(10, 8))
    
    if labels:
        sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm', center=0,
                   xticklabels=labels, yticklabels=labels)
    else:
        sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm', center=0)
    
    plt.title('向量相似度热力图')
    plt.tight_layout()
    plt.show()


class Timer:
    """简单的计时器类"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
    
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self) -> float:
        """获取已用时间"""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def benchmark_function(func, *args, n_runs: int = 10, **kwargs) -> Dict[str, float]:
    """基准测试函数性能
    
    Args:
        func: 要测试的函数
        args: 函数参数
        n_runs: 运行次数
        kwargs: 函数关键字参数
        
    Returns:
        性能统计信息
    """
    times = []
    
    for _ in range(n_runs):
        with Timer() as timer:
            result = func(*args, **kwargs)
        times.append(timer.elapsed())
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'total_time': np.sum(times)
    }


def create_sample_data_files(output_dir: str = "data"):
    """创建示例数据文件
    
    Args:
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. 创建示例文档数据
    sample_documents = [
        {
            "id": i,
            "title": f"Document {i:03d}",
            "content": f"This is sample document number {i}. " + 
                      "It contains various topics related to technology, science, and research. " * (i % 3 + 1),
            "category": ["technology", "science", "research"][i % 3],
            "tags": [f"tag_{i%5}", f"tag_{(i+1)%5}"]
        }
        for i in range(100)
    ]
    
    save_json_data(sample_documents, output_path / "sample_documents.json")
    
    # 2. 创建示例查询
    sample_queries = [
        "machine learning algorithms",
        "artificial intelligence applications",
        "data science techniques",
        "computer vision methods",
        "natural language processing",
        "deep learning models",
        "neural network architectures",
        "text classification approaches",
        "image recognition systems",
        "recommendation algorithms"
    ]
    
    with open(output_path / "sample_queries.txt", 'w', encoding='utf-8') as f:
        for query in sample_queries:
            f.write(query + '\n')
    
    # 3. 创建示例向量数据
    vectors = np.random.rand(100, 128).astype(np.float32)
    vectors = normalize_vectors(vectors)
    labels = [f"Vector_{i:03d}" for i in range(100)]
    
    save_vectors(vectors, labels, output_path / "sample_vectors.pkl")
    
    print(f"示例数据文件已创建在 {output_dir} 目录中:")
    print(f"  - sample_documents.json: {len(sample_documents)} 个文档")
    print(f"  - sample_queries.txt: {len(sample_queries)} 个查询")
    print(f"  - sample_vectors.pkl: {len(vectors)} 个向量")


def print_vector_stats(vectors: np.ndarray, name: str = "向量"):
    """打印向量统计信息
    
    Args:
        vectors: 向量数组
        name: 向量名称
    """
    stats = calculate_vector_stats(vectors)
    
    print(f"\n{name} 统计信息:")
    print(f"  向量数量: {stats['n_vectors']}")
    print(f"  向量维度: {stats['vector_dim']}")
    print(f"  值范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  均值: {stats['mean']:.4f}")
    print(f"  标准差: {stats['std']:.4f}")
    print(f"  L2范数均值: {stats['l2_norm_mean']:.4f}")
    print(f"  L2范数标准差: {stats['l2_norm_std']:.4f}")


def compare_vectors(vectors1: np.ndarray, vectors2: np.ndarray, 
                   names: Tuple[str, str] = ("向量集1", "向量集2")):
    """比较两组向量的统计信息
    
    Args:
        vectors1: 第一组向量
        vectors2: 第二组向量
        names: 向量集名称
    """
    print("=== 向量比较 ===")
    
    print_vector_stats(vectors1, names[0])
    print_vector_stats(vectors2, names[1])
    
    # 比较相似度分布
    if vectors1.shape[1] == vectors2.shape[1]:
        print(f"\n相似度比较:")
        
        # 计算内部相似度
        sample_size = min(100, len(vectors1), len(vectors2))
        sample1 = vectors1[:sample_size]
        sample2 = vectors2[:sample_size]
        
        # 内部相似度
        sim1 = compute_similarity_matrix(sample1[:10], "cosine")
        sim2 = compute_similarity_matrix(sample2[:10], "cosine")
        
        print(f"  {names[0]} 内部平均相似度: {np.mean(sim1[np.triu_indices_from(sim1, k=1)]):.4f}")
        print(f"  {names[1]} 内部平均相似度: {np.mean(sim2[np.triu_indices_from(sim2, k=1)]):.4f}")


def load_documents(filepath: str) -> List[Dict[str, Any]]:
    """加载文档数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        文档列表
    """
    if filepath.endswith('.json'):
        return load_json_data(filepath)
    elif filepath.endswith('.txt'):
        # 简单文本文件处理
        documents = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    documents.append({
                        'id': i,
                        'title': f'文档 {i+1}',
                        'content': line,
                        'category': 'general'
                    })
        return documents
    else:
        raise ValueError(f"不支持的文件格式: {filepath}")


def save_json(data: Any, filepath: str) -> None:
    """保存数据为JSON文件
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class VectorSearchBenchmark:
    """向量搜索性能基准测试"""
    
    def __init__(self):
        self.results = {}
    
    def run_benchmark(self, search_engines: List[Any], test_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行基准测试
        
        Args:
            search_engines: 搜索引擎列表
            test_data: 测试数据
            
        Returns:
            基准测试结果
        """
        results = {}
        
        for engine in search_engines:
            engine_name = engine.__class__.__name__
            start_time = time.time()
            
            # 简化的基准测试
            try:
                if hasattr(engine, 'search'):
                    test_query = np.random.rand(100)  # 假设100维向量
                    search_results = engine.search(test_query, top_k=5)
                    
                end_time = time.time()
                
                results[engine_name] = {
                    'search_time': end_time - start_time,
                    'results_count': len(search_results) if search_results else 0,
                    'status': 'success'
                }
            except Exception as e:
                results[engine_name] = {
                    'search_time': 0,
                    'results_count': 0,
                    'status': f'error: {str(e)}'
                }
        
        return results


def main():
    """主函数 - 演示工具函数"""
    print("=== 工具函数演示 ===\n")
    
    # 1. 创建示例数据文件
    print("1. 创建示例数据文件...")
    create_sample_data_files()
    
    # 2. 生成示例向量
    print("\n2. 生成示例向量...")
    vectors1 = np.random.rand(50, 64)
    vectors2 = np.random.randn(50, 64)  # 正态分布
    
    # 3. 向量统计和比较
    print("\n3. 向量统计和比较...")
    compare_vectors(vectors1, vectors2, ("随机向量", "正态分布向量"))
    
    # 4. 可视化向量分布
    print("\n4. 可视化向量分布...")
    try:
        plot_vector_distribution(vectors1, "随机向量分布")
    except Exception as e:
        print(f"可视化失败: {e}")
    
    # 5. 性能基准测试示例
    print("\n5. 性能基准测试...")
    def sample_function(n):
        return np.random.rand(n, 100)
    
    perf_stats = benchmark_function(sample_function, 1000, n_runs=5)
    print("函数性能统计:")
    for key, value in perf_stats.items():
        print(f"  {key}: {value:.6f}s")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    main()
