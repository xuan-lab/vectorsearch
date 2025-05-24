# 向量搜索学习项目

这是一个全面的向量搜索学习项目，从基础概念到高级实现，帮助您理解和掌握向量搜索技术。

## 项目结构

```
vectorsearch/
├── README.md                   # 项目说明
├── requirements.txt            # 依赖包
├── src/                       # 源代码
│   ├── __init__.py
│   ├── basic_vector_search.py  # 基础向量搜索实现
│   ├── text_vectorizer.py      # 文本向量化
│   ├── advanced_search.py      # 高级向量搜索
│   └── utils.py               # 工具函数
├── data/                      # 示例数据
│   ├── sample_documents.json   # 示例文档
│   └── sample_queries.txt      # 示例查询
├── notebooks/                 # Jupyter 笔记本
│   ├── 01_vector_basics.ipynb  # 向量基础
│   ├── 02_similarity_metrics.ipynb # 相似度计算
│   ├── 03_text_embeddings.ipynb   # 文本嵌入
│   ├── 04_faiss_demo.ipynb        # FAISS 演示
│   └── 05_applications.ipynb      # 实际应用
├── examples/                  # 示例应用
│   ├── document_search.py      # 文档搜索
│   ├── recommendation.py       # 推荐系统
│   └── semantic_search.py      # 语义搜索
└── tests/                     # 测试文件
    ├── __init__.py
    ├── test_basic_search.py
    └── test_vectorizer.py
```

## 学习目标

1. **理解向量搜索基础概念**
   - 向量表示
   - 相似度计算
   - 距离度量

2. **掌握文本向量化技术**
   - TF-IDF
   - Word2Vec
   - Sentence Transformers

3. **学习高效搜索算法**
   - 线性搜索
   - 近似最近邻 (ANN)
   - FAISS 库使用

4. **实际应用开发**
   - 文档搜索系统
   - 推荐系统
   - 语义搜索

## 快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **运行基础示例**
   ```bash
   python src/basic_vector_search.py
   ```

3. **启动 Jupyter 笔记本**
   ```bash
   jupyter notebook notebooks/
   ```

## 学习路径

建议按照以下顺序学习：

1. 阅读 `notebooks/01_vector_basics.ipynb` - 了解向量基础
2. 学习 `notebooks/02_similarity_metrics.ipynb` - 掌握相似度计算
3. 实践 `notebooks/03_text_embeddings.ipynb` - 文本向量化
4. 探索 `notebooks/04_faiss_demo.ipynb` - 高效搜索库
5. 应用 `notebooks/05_applications.ipynb` - 实际项目

## 核心概念

### 向量搜索
向量搜索是一种通过计算向量相似度来查找相关内容的技术。它将数据转换为高维向量，然后使用数学方法计算向量之间的相似性。

### 应用场景
- 搜索引擎
- 推荐系统
- 相似图片搜索
- 语义文本搜索
- 机器学习特征匹配

## 贡献

欢迎提交问题和改进建议！

## 许可证

MIT License
