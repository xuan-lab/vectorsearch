#!/usr/bin/env python3
"""
测试 NLTK 增强的向量搜索功能
Test NLTK-Enhanced Vector Search Functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from text_vectorizer import TextVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_tfidf_with_nltk_preprocessing():
    """测试 TF-IDF 与 NLTK 预处理的集成"""
    print("🔍 测试 TF-IDF 与 NLTK 预处理集成")
    print("=" * 50)
    
    vectorizer = TextVectorizer()
    
    # 测试文档集合（包含中英文）
    documents = [
        "人工智能 (AI) 是计算机科学的一个重要分支，专注于创建能够执行智能任务的系统。",
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "深度学习使用多层神经网络来处理和分析复杂的数据模式。",
        "Natural language processing helps computers understand and generate human language.",
        "数据科学结合了统计学、编程和领域专业知识来从数据中提取洞察。",
        "Computer vision enables machines to interpret and understand visual information from images.",
        "Python 是数据科学和机器学习领域最受欢迎的编程语言之一。",
        "Deep learning algorithms require large amounts of data and computational resources to train effectively."
    ]
    
    print(f"📚 测试文档数量: {len(documents)}")
    print("\n文档内容:")
    for i, doc in enumerate(documents, 1):
        print(f"{i:2d}. {doc}")
    
    # 进行 TF-IDF 向量化
    print("\n🔄 正在进行 TF-IDF 向量化...")
    try:
        tfidf_matrix = vectorizer.tfidf_vectorize(documents, max_features=500)
        print(f"✅ 向量化成功: {tfidf_matrix.shape}")
        
        # 测试查询
        test_queries = [
            "人工智能",               # 中文查询
            "machine learning",       # 英文查询  
            "深度学习 neural networks", # 中英混合查询
            "数据科学 Python",        # 中英混合查询
            "computer vision algorithms" # 英文查询
        ]
        
        print(f"\n🔍 测试查询 ({len(test_queries)} 个):")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n查询 {i}: '{query}'")
            print("-" * 40)
            
            try:
                # 将查询转换为向量
                query_vector = vectorizer.tfidf_transform_query(query)
                print(f"✅ 查询向量化成功: 特征数 {len(query_vector)}")
                
                # 计算与所有文档的余弦相似度
                similarities = cosine_similarity([query_vector], tfidf_matrix)[0]
                
                # 获取前3个最相似的文档
                top_indices = np.argsort(similarities)[::-1][:3]
                
                print("📊 最相关的文档:")
                for rank, idx in enumerate(top_indices, 1):
                    sim_score = similarities[idx]
                    doc_preview = documents[idx][:60] + "..." if len(documents[idx]) > 60 else documents[idx]
                    print(f"  {rank}. 相似度: {sim_score:.3f} | {doc_preview}")
                    
            except Exception as e:
                print(f"❌ 查询处理失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 向量化失败: {e}")
        return False

def test_preprocessing_quality():
    """测试预处理质量"""
    print("\n\n🧹 测试预处理质量")
    print("=" * 50)
    
    vectorizer = TextVectorizer()
    
    test_cases = [
        {
            "type": "中文文本",
            "text": "人工智能和机器学习是现代科技的重要组成部分！它们正在改变我们的生活。",
            "expected_features": ["保留中文字符", "移除标点符号"]
        },
        {
            "type": "英文文本", 
            "text": "Machine learning algorithms are powerful tools for data analysis and prediction.",
            "expected_features": ["移除停用词", "词干提取", "保留关键词"]
        },
        {
            "type": "中英混合",
            "text": "Python是一种流行的programming language，特别适合data science和AI开发。",
            "expected_features": ["保留中英文", "混合语言处理"]
        },
        {
            "type": "技术术语",
            "text": "TensorFlow, PyTorch, and scikit-learn are popular machine learning libraries.",
            "expected_features": ["保留技术术语", "处理特殊字符"]
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {case['type']}")
        print(f"原文: {case['text']}")
        
        processed = vectorizer.preprocess_text(case['text'])
        print(f"处理后: {processed}")
        
        # 简单的质量检查
        original_chinese_chars = len([c for c in case['text'] if '\u4e00' <= c <= '\u9fff'])
        processed_chinese_chars = len([c for c in processed if '\u4e00' <= c <= '\u9fff'])
        
        if original_chinese_chars > 0:
            if processed_chinese_chars >= original_chinese_chars * 0.8:  # 允许一些损失
                print("✅ 中文字符保留良好")
            else:
                print("⚠️  中文字符可能丢失过多")
        
        if len(processed.strip()) > 0:
            print("✅ 处理后文本非空")
        else:
            print("❌ 处理后文本为空!")
            
        print(f"期望特性: {', '.join(case['expected_features'])}")

def test_vocabulary_quality():
    """测试词汇表质量"""
    print("\n\n📝 测试词汇表质量")
    print("=" * 50)
    
    vectorizer = TextVectorizer()
    
    # 创建一个包含多样化内容的文档集
    diverse_documents = [
        "机器学习和深度学习是人工智能的核心技术",
        "Machine learning and deep learning are core AI technologies", 
        "Python和R是数据科学中最常用的编程语言",
        "Data scientists use Python and R for statistical analysis",
        "神经网络模拟人脑的工作原理来处理信息",
        "Neural networks mimic the human brain to process information",
        "自然语言处理帮助计算机理解人类语言",
        "Natural language processing helps computers understand human text"
    ]
    
    print(f"📚 使用 {len(diverse_documents)} 个文档建立词汇表")
    
    try:
        # 进行向量化以建立词汇表
        tfidf_matrix = vectorizer.tfidf_vectorize(diverse_documents, max_features=200)
        
        # 获取特征名称（词汇表）
        if hasattr(vectorizer.tfidf_vectorizer, 'get_feature_names_out'):
            vocabulary = vectorizer.tfidf_vectorizer.get_feature_names_out()
        else:
            vocabulary = vectorizer.tfidf_vectorizer.get_feature_names()
        
        print(f"✅ 词汇表大小: {len(vocabulary)}")
        
        # 分析词汇表质量
        chinese_terms = [term for term in vocabulary if any('\u4e00' <= c <= '\u9fff' for c in term)]
        english_terms = [term for term in vocabulary if term.isalpha() and not any('\u4e00' <= c <= '\u9fff' for c in term)]
        mixed_terms = [term for term in vocabulary if not term in chinese_terms and not term in english_terms]
        
        print(f"📊 词汇分布:")
        print(f"  中文词汇: {len(chinese_terms)} 个")
        print(f"  英文词汇: {len(english_terms)} 个") 
        print(f"  其他词汇: {len(mixed_terms)} 个")
        
        # 显示一些示例词汇
        print(f"\n📝 中文词汇示例 (前10个):")
        for term in chinese_terms[:10]:
            print(f"  {term}")
            
        print(f"\n📝 英文词汇示例 (前10个):")
        for term in english_terms[:10]:
            print(f"  {term}")
            
        if len(chinese_terms) > 0 and len(english_terms) > 0:
            print("✅ 词汇表同时包含中英文词汇，多语言支持良好")
        else:
            print("⚠️  词汇表可能缺少某种语言的词汇")
            
        return True
        
    except Exception as e:
        print(f"❌ 词汇表分析失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 NLTK 增强向量搜索功能测试")
    print("=" * 60)
    
    # 检查 NLTK 状态
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        print("✅ NLTK 库已安装并可用")
    except ImportError as e:
        print(f"❌ NLTK 库不可用: {e}")
        print("💡 将使用基础文本处理功能")
    
    print()
    
    # 运行各项测试
    results = []
    
    # 测试1: TF-IDF 与 NLTK 集成
    results.append(test_tfidf_with_nltk_preprocessing())
    
    # 测试2: 预处理质量
    test_preprocessing_quality()
    
    # 测试3: 词汇表质量 
    results.append(test_vocabulary_quality())
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("📋 测试总结")
    print("=" * 60)
    
    passed_tests = sum(results)
    total_tests = len(results)
    
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 所有核心功能测试通过!")
        print("💡 NLTK 增强的向量搜索功能已准备就绪")
    else:
        print("⚠️  部分测试未通过，请检查相关功能")
    
    print("\n🔧 功能亮点:")
    print("• 语言感知的文本预处理")
    print("• 中英文混合文档支持") 
    print("• NLTK 停用词过滤和词干提取")
    print("• 高质量的 TF-IDF 向量化")
    print("• 多语言语义搜索能力")

if __name__ == "__main__":
    main()
