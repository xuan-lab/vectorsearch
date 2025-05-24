"""
文本向量化模块
Text Vectorizer Module

本模块包含多种文本向量化方法：
- TF-IDF 向量化
- Word2Vec 嵌入
- Sentence Transformers 嵌入
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 可选依赖，如果没有安装会给出提示
try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("警告: gensim 未安装，Word2Vec 功能将不可用")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("警告: sentence-transformers 未安装，Sentence Transformer 功能将不可用")

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("警告: nltk 未安装，文本预处理功能受限")


class TextVectorizer:
    """文本向量化类"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.sentence_transformer = None
        self.stemmer = PorterStemmer() if NLTK_AVAILABLE else None
        
    def preprocess_text(self, text: str) -> str:
        """文本预处理
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        # 转换为小写
        text = text.lower()
        
        # 如果有 NLTK，进行更高级的预处理
        if NLTK_AVAILABLE:
            try:
                # 分词
                tokens = word_tokenize(text)
                
                # 移除停用词
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
                  # 词干提取
                if self.stemmer:
                    tokens = [self.stemmer.stem(token) for token in tokens]
                
                text = ' '.join(tokens)
            except LookupError:
                print("NLTK 数据未下载，使用基础预处理")
                # 基础清理 - 保留中英文字符、数字和空格
                import re
                text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        else:            # 基础清理 - 保留中英文字符、数字和空格
            import re
            text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        
        return text
    
    def tfidf_vectorize(self, texts: List[str], max_features: int = 1000,
                       preprocess: bool = True) -> np.ndarray:
        """使用 TF-IDF 进行文本向量化
        
        Args:
            texts: 文本列表
            max_features: 最大特征数
            preprocess: 是否进行文本预处理
            
        Returns:
            TF-IDF 向量矩阵
        """
        if preprocess:
            texts = [self.preprocess_text(text) for text in texts]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=None,  # 不使用英文停用词，因为我们有中文内容
            lowercase=True,
            analyzer='char_wb',  # 使用字符n-gram，适合中文文本
            ngram_range=(1, 3),  # 1-3字符的n-gram
            min_df=1,  # 最小文档频率为1，保留更多词汇
            max_df=0.9  # 最大文档频率为90%
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        print(f"TF-IDF 向量化完成: {tfidf_matrix.shape[0]} 个文档, {tfidf_matrix.shape[1]} 个特征")
        
        return tfidf_matrix.toarray()
    
    def tfidf_transform_query(self, query_text: str, preprocess: bool = True) -> np.ndarray:
        """使用已训练的 TF-IDF 向量化器转换查询文本
        
        Args:
            query_text: 查询文本
            preprocess: 是否进行文本预处理
            
        Returns:
            查询的 TF-IDF 向量
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF 向量化器尚未训练，请先调用 tfidf_vectorize")
        
        if preprocess:
            query_text = self.preprocess_text(query_text)
        
        query_vector = self.tfidf_vectorizer.transform([query_text])
        
        return query_vector.toarray()[0]
    
    def word2vec_vectorize(self, texts: List[str], vector_size: int = 100, 
                          window: int = 5, min_count: int = 1, epochs: int = 10) -> np.ndarray:
        """使用 Word2Vec 进行文本向量化
        
        Args:
            texts: 文本列表
            vector_size: 向量维度
            window: 上下文窗口大小
            min_count: 最小词频
            epochs: 训练轮数
            
        Returns:
            Word2Vec 向量矩阵
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim 未安装，无法使用 Word2Vec")
        
        # 预处理并分词
        processed_texts = []
        for text in texts:
            processed_text = self.preprocess_text(text)
            tokens = simple_preprocess(processed_text)
            processed_texts.append(tokens)
        
        # 训练 Word2Vec 模型
        self.word2vec_model = Word2Vec(
            sentences=processed_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            workers=4
        )
        
        # 为每个文档生成向量（取词向量的平均值）
        document_vectors = []
        for tokens in processed_texts:
            if tokens:  # 确保文档不为空
                word_vectors = [self.word2vec_model.wv[word] for word in tokens 
                              if word in self.word2vec_model.wv]
                if word_vectors:
                    doc_vector = np.mean(word_vectors, axis=0)
                else:
                    doc_vector = np.zeros(vector_size)
            else:
                doc_vector = np.zeros(vector_size)
            
            document_vectors.append(doc_vector)
        
        print(f"Word2Vec 向量化完成: {len(document_vectors)} 个文档, {vector_size} 维向量")
        
        return np.array(document_vectors)
    
    def sentence_transformer_vectorize(self, texts: List[str], 
                                     model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
        """使用 Sentence Transformers 进行文本向量化
        
        Args:
            texts: 文本列表
            model_name: 预训练模型名称
            
        Returns:
            Sentence Transformer 向量矩阵
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers 未安装，无法使用 Sentence Transformer")
        
        # 加载预训练模型
        self.sentence_transformer = SentenceTransformer(model_name)
        
        # 生成句子嵌入
        embeddings = self.sentence_transformer.encode(texts, show_progress_bar=True)
        
        print(f"Sentence Transformer 向量化完成: {embeddings.shape[0]} 个文档, {embeddings.shape[1]} 维向量")
        
        return embeddings
    
    def compare_vectorization_methods(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """比较不同的向量化方法
        
        Args:
            texts: 文本列表
            
        Returns:
            包含不同方法结果的字典
        """
        results = {}
        
        print("正在比较不同的向量化方法...\n")
        
        # TF-IDF
        print("1. TF-IDF 向量化...")
        try:
            tfidf_vectors = self.tfidf_vectorize(texts)
            results['TF-IDF'] = tfidf_vectors
        except Exception as e:
            print(f"TF-IDF 向量化失败: {e}")
        
        # Word2Vec
        print("\n2. Word2Vec 向量化...")
        try:
            word2vec_vectors = self.word2vec_vectorize(texts)
            results['Word2Vec'] = word2vec_vectors
        except Exception as e:
            print(f"Word2Vec 向量化失败: {e}")
        
        # Sentence Transformers
        print("\n3. Sentence Transformer 向量化...")
        try:
            st_vectors = self.sentence_transformer_vectorize(texts)
            results['Sentence Transformer'] = st_vectors
        except Exception as e:
            print(f"Sentence Transformer 向量化失败: {e}")
        
        return results
    
    def visualize_vectors(self, vectors: np.ndarray, labels: List[str], 
                         method_name: str = "向量", max_points: int = 100):
        """可视化高维向量（使用 PCA 降维到 2D）
        
        Args:
            vectors: 向量矩阵
            labels: 标签列表
            method_name: 方法名称
            max_points: 最大显示点数
        """
        if vectors.shape[0] > max_points:
            # 随机采样
            indices = np.random.choice(vectors.shape[0], max_points, replace=False)
            vectors = vectors[indices]
            labels = [labels[i] for i in indices]
        
        # PCA 降维到 2D
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6, s=50)
        
        plt.title(f'{method_name} 向量可视化 (PCA)')
        plt.xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
        
        # 添加一些标签
        for i in range(min(10, len(labels))):  # 只显示前10个标签，避免拥挤
            plt.annotate(labels[i][:20], (vectors_2d[i, 0], vectors_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def load_sample_documents() -> List[str]:
    """加载示例文档
    
    Returns:
        示例文档列表
    """
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret and analyze visual information.",
        "Data science combines statistics, programming, and domain expertise.",
        "Artificial intelligence aims to create systems that can perform human-like tasks.",
        "Python is a popular programming language for data science and machine learning.",
        "Algorithms are step-by-step procedures for solving computational problems.",
        "Big data refers to extremely large datasets that require special tools to process.",
        "Cloud computing provides on-demand access to computing resources over the internet.",
        "Cybersecurity protects digital systems from malicious attacks and threats.",
        "Software engineering involves designing and developing software applications.",
        "Database management systems store and organize large amounts of data efficiently.",
        "Web development creates websites and web applications for the internet.",
        "Mobile app development focuses on creating applications for smartphones and tablets.",
        "User experience design ensures that digital products are intuitive and user-friendly.",
        "Blockchain technology creates secure and transparent digital ledgers.",
        "Internet of Things connects everyday devices to the internet for smart functionality.",
        "Robotics combines engineering and computer science to create autonomous machines.",
        "Quantum computing uses quantum mechanics principles for advanced computation."
    ]
    
    return sample_docs


def main():
    """主函数 - 演示文本向量化"""
    print("=== 文本向量化演示 ===\n")
    
    # 1. 加载示例文档
    print("1. 加载示例文档...")
    documents = load_sample_documents()
    print(f"已加载 {len(documents)} 个文档")
    
    # 2. 创建向量化器
    print("\n2. 创建文本向量化器...")
    vectorizer = TextVectorizer()
    
    # 3. 比较不同的向量化方法
    print("\n3. 比较不同的向量化方法...")
    vector_results = vectorizer.compare_vectorization_methods(documents)
    
    # 4. 显示结果统计
    print("\n4. 向量化结果统计:")
    for method, vectors in vector_results.items():
        print(f"  {method}: {vectors.shape[0]} 个文档, {vectors.shape[1]} 维向量")
    
    # 5. 可视化向量（如果有结果）
    print("\n5. 可视化向量...")
    labels = [f"Doc_{i:02d}" for i in range(len(documents))]
    
    for method, vectors in vector_results.items():
        try:
            vectorizer.visualize_vectors(vectors, labels, method)
        except Exception as e:
            print(f"可视化 {method} 失败: {e}")
    
    print("\n=== 演示完成 ===")
    
    return vector_results


if __name__ == "__main__":
    main()
