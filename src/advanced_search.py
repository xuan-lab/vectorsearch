"""
高级向量搜索模块
Advanced Vector Search Module

本模块包含高效的向量搜索实现：
- FAISS 索引
- Annoy 索引
- 批量搜索
- 索引持久化
"""

import numpy as np
import time
import pickle
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt

# 可选依赖
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: faiss 未安装，FAISS 功能将不可用")

try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False
    print("警告: annoy 未安装，Annoy 功能将不可用")


class FAISSSearch:
    """FAISS 向量搜索类"""
    
    def __init__(self, vector_dim: int, index_type: str = "flat"):
        """初始化 FAISS 搜索
        
        Args:
            vector_dim: 向量维度
            index_type: 索引类型 ("flat", "ivf", "hnsw")
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss 未安装，无法使用 FAISS 搜索")
        
        self.vector_dim = vector_dim
        self.index_type = index_type
        self.index = None
        self.labels = None
        self.vectors = None
        
        self._create_index()
    
    def _create_index(self):
        """创建 FAISS 索引"""
        if self.index_type == "flat":
            # 精确搜索，适合小数据集
            self.index = faiss.IndexFlatIP(self.vector_dim)  # 内积相似度
        elif self.index_type == "ivf":
            # IVF 索引，适合中等大小数据集
            nlist = 100  # 聚类中心数量
            quantizer = faiss.IndexFlatIP(self.vector_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, nlist)
        elif self.index_type == "hnsw":
            # HNSW 索引，适合大数据集
            M = 32  # 连接数
            self.index = faiss.IndexHNSWFlat(self.vector_dim, M)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
    
    def add_vectors(self, vectors: np.ndarray, labels: Optional[List[str]] = None):
        """添加向量到索引
        
        Args:
            vectors: 向量数组
            labels: 向量标签
        """
        vectors = np.array(vectors, dtype=np.float32)
        
        # 归一化向量（对内积相似度重要）
        faiss.normalize_L2(vectors)
        
        # 训练索引（某些索引类型需要）
        if self.index_type == "ivf" and not self.index.is_trained:
            print("训练 IVF 索引...")
            self.index.train(vectors)
        
        # 添加向量
        self.index.add(vectors)
        
        self.vectors = vectors
        self.labels = labels if labels is not None else [f"Vector_{i}" for i in range(len(vectors))]
        
        print(f"已添加 {len(vectors)} 个向量到 FAISS {self.index_type} 索引")
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """搜索最相似的向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回前k个结果
            
        Returns:
            (索引, 相似度, 标签) 的列表
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("索引为空，请先添加向量")
        
        query_vector = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # 搜索
        similarities, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx != -1:  # 有效索引
                results.append((idx, float(sim), self.labels[idx]))
        
        return results
    
    def batch_search(self, query_vectors: np.ndarray, top_k: int = 5) -> List[List[Tuple[int, float, str]]]:
        """批量搜索
        
        Args:
            query_vectors: 查询向量数组
            top_k: 每个查询返回前k个结果
            
        Returns:
            每个查询的搜索结果列表
        """
        query_vectors = np.array(query_vectors, dtype=np.float32)
        faiss.normalize_L2(query_vectors)
        
        similarities, indices = self.index.search(query_vectors, top_k)
        
        batch_results = []
        for i in range(len(query_vectors)):
            results = []
            for sim, idx in zip(similarities[i], indices[i]):
                if idx != -1:
                    results.append((idx, float(sim), self.labels[idx]))
            batch_results.append(results)
        
        return batch_results
    
    def save_index(self, filepath: str):
        """保存索引到文件
        
        Args:
            filepath: 文件路径
        """
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # 保存标签和其他元数据
        metadata = {
            'labels': self.labels,
            'vector_dim': self.vector_dim,
            'index_type': self.index_type
        }
        with open(f"{filepath}.metadata", 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"索引已保存到 {filepath}")
    
    def load_index(self, filepath: str):
        """从文件加载索引
        
        Args:
            filepath: 文件路径
        """
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # 加载元数据
        with open(f"{filepath}.metadata", 'rb') as f:
            metadata = pickle.load(f)
        
        self.labels = metadata['labels']
        self.vector_dim = metadata['vector_dim']
        self.index_type = metadata['index_type']
        
        print(f"索引已从 {filepath} 加载")


class AnnoySearch:
    """Annoy 向量搜索类"""
    
    def __init__(self, vector_dim: int, metric: str = "angular"):
        """初始化 Annoy 搜索
        
        Args:
            vector_dim: 向量维度
            metric: 距离度量 ("angular", "euclidean", "manhattan", "hamming", "dot")
        """
        if not ANNOY_AVAILABLE:
            raise ImportError("annoy 未安装，无法使用 Annoy 搜索")
        
        self.vector_dim = vector_dim
        self.metric = metric
        self.index = AnnoyIndex(vector_dim, metric)
        self.labels = []
        self.built = False
    
    def add_vectors(self, vectors: np.ndarray, labels: Optional[List[str]] = None):
        """添加向量到索引
        
        Args:
            vectors: 向量数组
            labels: 向量标签
        """
        vectors = np.array(vectors, dtype=np.float32)
        
        for i, vector in enumerate(vectors):
            self.index.add_item(i, vector)
        
        self.labels = labels if labels is not None else [f"Vector_{i}" for i in range(len(vectors))]
        
        print(f"已添加 {len(vectors)} 个向量到 Annoy 索引")
    
    def build_index(self, n_trees: int = 10):
        """构建索引
        
        Args:
            n_trees: 树的数量（更多树 = 更高精度，更慢构建）
        """
        print(f"构建 Annoy 索引 ({n_trees} 棵树)...")
        self.index.build(n_trees)
        self.built = True
        print("索引构建完成")
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """搜索最相似的向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回前k个结果
            
        Returns:
            (索引, 距离, 标签) 的列表
        """
        if not self.built:
            raise ValueError("索引未构建，请先调用 build_index()")
        
        indices, distances = self.index.get_nns_by_vector(
            query_vector, top_k, include_distances=True
        )
        
        results = []
        for idx, dist in zip(indices, distances):
            results.append((idx, dist, self.labels[idx]))
        
        return results
    
    def save_index(self, filepath: str):
        """保存索引到文件
        
        Args:
            filepath: 文件路径
        """
        self.index.save(f"{filepath}.ann")
        
        # 保存标签和元数据
        metadata = {
            'labels': self.labels,
            'vector_dim': self.vector_dim,
            'metric': self.metric,
            'built': self.built
        }
        with open(f"{filepath}.metadata", 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Annoy 索引已保存到 {filepath}")
    
    def load_index(self, filepath: str):
        """从文件加载索引
        
        Args:
            filepath: 文件路径
        """
        # 加载元数据
        with open(f"{filepath}.metadata", 'rb') as f:
            metadata = pickle.load(f)
        
        self.labels = metadata['labels']
        self.vector_dim = metadata['vector_dim']
        self.metric = metadata['metric']
        self.built = metadata['built']
        
        # 重新创建索引并加载
        self.index = AnnoyIndex(self.vector_dim, self.metric)
        self.index.load(f"{filepath}.ann")
        
        print(f"Annoy 索引已从 {filepath} 加载")


class VectorSearchBenchmark:
    """向量搜索性能基准测试"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_search_methods(self, vectors: np.ndarray, query_vectors: np.ndarray,
                                labels: Optional[List[str]] = None, top_k: int = 5) -> Dict[str, Any]:
        """对比不同搜索方法的性能
        
        Args:
            vectors: 数据向量
            query_vectors: 查询向量
            labels: 向量标签
            top_k: 返回结果数量
            
        Returns:
            性能比较结果
        """
        vector_dim = vectors.shape[1]
        n_vectors = len(vectors)
        n_queries = len(query_vectors)
        
        print(f"开始性能基准测试: {n_vectors} 个向量, {n_queries} 个查询, {vector_dim} 维")
        
        results = {}
        
        # 1. FAISS Flat
        if FAISS_AVAILABLE:
            print("\n测试 FAISS Flat...")
            try:
                start_time = time.time()
                faiss_flat = FAISSSearch(vector_dim, "flat")
                faiss_flat.add_vectors(vectors, labels)
                build_time = time.time() - start_time
                
                start_time = time.time()
                for query in query_vectors:
                    faiss_flat.search(query, top_k)
                search_time = time.time() - start_time
                
                results['FAISS_Flat'] = {
                    'build_time': build_time,
                    'search_time': search_time,
                    'avg_search_time': search_time / n_queries
                }
                print(f"  构建时间: {build_time:.4f}s, 搜索时间: {search_time:.4f}s")
            except Exception as e:
                print(f"  FAISS Flat 测试失败: {e}")
        
        # 2. FAISS HNSW
        if FAISS_AVAILABLE and n_vectors > 100:  # HNSW 适合较大数据集
            print("\n测试 FAISS HNSW...")
            try:
                start_time = time.time()
                faiss_hnsw = FAISSSearch(vector_dim, "hnsw")
                faiss_hnsw.add_vectors(vectors, labels)
                build_time = time.time() - start_time
                
                start_time = time.time()
                for query in query_vectors:
                    faiss_hnsw.search(query, top_k)
                search_time = time.time() - start_time
                
                results['FAISS_HNSW'] = {
                    'build_time': build_time,
                    'search_time': search_time,
                    'avg_search_time': search_time / n_queries
                }
                print(f"  构建时间: {build_time:.4f}s, 搜索时间: {search_time:.4f}s")
            except Exception as e:
                print(f"  FAISS HNSW 测试失败: {e}")
        
        # 3. Annoy
        if ANNOY_AVAILABLE:
            print("\n测试 Annoy...")
            try:
                start_time = time.time()
                annoy_search = AnnoySearch(vector_dim, "angular")
                annoy_search.add_vectors(vectors, labels)
                annoy_search.build_index(n_trees=10)
                build_time = time.time() - start_time
                
                start_time = time.time()
                for query in query_vectors:
                    annoy_search.search(query, top_k)
                search_time = time.time() - start_time
                
                results['Annoy'] = {
                    'build_time': build_time,
                    'search_time': search_time,
                    'avg_search_time': search_time / n_queries
                }
                print(f"  构建时间: {build_time:.4f}s, 搜索时间: {search_time:.4f}s")
            except Exception as e:
                print(f"  Annoy 测试失败: {e}")
        
        # 4. 线性搜索（基准）
        print("\n测试线性搜索...")
        try:
            from .basic_vector_search import BasicVectorSearch
            
            start_time = time.time()
            linear_search = BasicVectorSearch()
            linear_search.add_vectors(vectors, labels)
            build_time = time.time() - start_time
            
            start_time = time.time()
            for query in query_vectors:
                linear_search.search(query, top_k, metric="cosine")
            search_time = time.time() - start_time
            
            results['Linear'] = {
                'build_time': build_time,
                'search_time': search_time,
                'avg_search_time': search_time / n_queries
            }
            print(f"  构建时间: {build_time:.4f}s, 搜索时间: {search_time:.4f}s")
        except Exception as e:
            print(f"  线性搜索测试失败: {e}")
        
        self.results = results
        return results
    
    def visualize_benchmark_results(self):
        """可视化基准测试结果"""
        if not self.results:
            print("没有基准测试结果可视化")
            return
        
        methods = list(self.results.keys())
        build_times = [self.results[method]['build_time'] for method in methods]
        search_times = [self.results[method]['search_time'] for method in methods]
        avg_search_times = [self.results[method]['avg_search_time'] for method in methods]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 构建时间
        axes[0].bar(methods, build_times)
        axes[0].set_title('索引构建时间')
        axes[0].set_ylabel('时间 (秒)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 总搜索时间
        axes[1].bar(methods, search_times)
        axes[1].set_title('总搜索时间')
        axes[1].set_ylabel('时间 (秒)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 平均搜索时间
        axes[2].bar(methods, avg_search_times)
        axes[2].set_title('平均搜索时间')
        axes[2].set_ylabel('时间 (秒)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


def main():
    """主函数 - 演示高级向量搜索"""
    print("=== 高级向量搜索演示 ===\n")
    
    # 1. 生成示例数据
    print("1. 生成示例数据...")
    n_vectors = 1000
    vector_dim = 128
    n_queries = 50
    
    vectors = np.random.rand(n_vectors, vector_dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # 归一化
    
    query_vectors = np.random.rand(n_queries, vector_dim).astype(np.float32)
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    
    labels = [f"Document_{i:04d}" for i in range(n_vectors)]
    
    print(f"生成了 {n_vectors} 个 {vector_dim} 维向量")
    
    # 2. 测试 FAISS 搜索
    if FAISS_AVAILABLE:
        print("\n2. 测试 FAISS 搜索...")
        faiss_search = FAISSSearch(vector_dim, "flat")
        faiss_search.add_vectors(vectors, labels)
        
        # 单个查询
        results = faiss_search.search(query_vectors[0], top_k=5)
        print("FAISS 搜索结果:")
        for i, (idx, sim, label) in enumerate(results):
            print(f"  {i+1}. {label}: {sim:.4f}")
    
    # 3. 测试 Annoy 搜索
    if ANNOY_AVAILABLE:
        print("\n3. 测试 Annoy 搜索...")
        annoy_search = AnnoySearch(vector_dim, "angular")
        annoy_search.add_vectors(vectors, labels)
        annoy_search.build_index(n_trees=10)
        
        # 单个查询
        results = annoy_search.search(query_vectors[0], top_k=5)
        print("Annoy 搜索结果:")
        for i, (idx, dist, label) in enumerate(results):
            print(f"  {i+1}. {label}: {dist:.4f}")
    
    # 4. 性能基准测试
    print("\n4. 性能基准测试...")
    benchmark = VectorSearchBenchmark()
    benchmark_results = benchmark.benchmark_search_methods(
        vectors, query_vectors[:10], labels, top_k=5  # 使用较少查询以节省时间
    )
    
    print("\n性能比较结果:")
    for method, stats in benchmark_results.items():
        print(f"  {method}:")
        print(f"    构建时间: {stats['build_time']:.4f}s")
        print(f"    总搜索时间: {stats['search_time']:.4f}s")
        print(f"    平均搜索时间: {stats['avg_search_time']:.6f}s")
    
    # 5. 可视化结果
    print("\n5. 可视化性能结果...")
    try:
        benchmark.visualize_benchmark_results()
    except Exception as e:
        print(f"可视化失败: {e}")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    main()
