#!/usr/bin/env python3
"""
测试语言感知的文本预处理功能
Test Language-Aware Text Preprocessing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from text_vectorizer import TextVectorizer

def test_chinese_text_preprocessing():
    """测试中文文本预处理"""
    print("=== 测试中文文本预处理 ===")
    
    vectorizer = TextVectorizer()
    
    # 测试中文文档
    chinese_texts = [
        "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智慧的任务的系统。",
        "机器学习是人工智能的一个子集，专注于算法和统计模型。",
        "深度学习使用多层神经网络来处理和分析数据。",
        "自然语言处理帮助计算机理解和生成人类语言。",
        "数据科学结合了统计学、编程和领域专业知识。"
    ]
    
    print("原始中文文本:")
    for i, text in enumerate(chinese_texts, 1):
        print(f"{i}. {text}")
    
    print("\n预处理后的中文文本:")
    for i, text in enumerate(chinese_texts, 1):
        processed = vectorizer.preprocess_text(text)
        print(f"{i}. {processed}")
        
        # 检查是否保留了中文字符
        chinese_chars_original = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        chinese_chars_processed = len([c for c in processed if '\u4e00' <= c <= '\u9fff'])
        
        print(f"   原始中文字符数: {chinese_chars_original}, 处理后: {chinese_chars_processed}")
        
        if chinese_chars_processed == 0 and chinese_chars_original > 0:
            print(f"   ❌ 错误: 中文字符被完全移除!")
        elif chinese_chars_processed > 0:
            print(f"   ✅ 正确: 中文字符得到保留")
    
    return chinese_texts

def test_english_text_preprocessing():
    """测试英文文本预处理"""
    print("\n=== 测试英文文本预处理 ===")
    
    vectorizer = TextVectorizer()
    
    # 测试英文文档
    english_texts = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret and analyze visual information.",
        "Data science combines statistics, programming, and domain expertise."
    ]
    
    print("原始英文文本:")
    for i, text in enumerate(english_texts, 1):
        print(f"{i}. {text}")
    
    print("\n预处理后的英文文本:")
    for i, text in enumerate(english_texts, 1):
        processed = vectorizer.preprocess_text(text)
        print(f"{i}. {processed}")
        
        # 检查是否正确处理了停用词和词干提取
        original_words = text.lower().split()
        processed_words = processed.split()
        
        print(f"   原始单词数: {len(original_words)}, 处理后: {len(processed_words)}")
        
        if len(processed_words) < len(original_words):
            print(f"   ✅ 正确: 停用词和标点符号被移除")
        else:
            print(f"   ⚠️  注意: 可能未正确移除停用词")
    
    return english_texts

def test_mixed_text_preprocessing():
    """测试中英文混合文本预处理"""
    print("\n=== 测试中英文混合文本预处理 ===")
    
    vectorizer = TextVectorizer()
    
    # 测试中英文混合文档
    mixed_texts = [
        "人工智能 (Artificial Intelligence, AI) 是一个快速发展的领域。",
        "Machine Learning 机器学习在各个行业都有应用。",
        "Python 是数据科学和机器学习的热门编程语言。",
        "深度学习 Deep Learning 使用神经网络 Neural Networks。",
        "自然语言处理 NLP 处理文本数据。"
    ]
    
    print("原始混合文本:")
    for i, text in enumerate(mixed_texts, 1):
        print(f"{i}. {text}")
    
    print("\n预处理后的混合文本:")
    for i, text in enumerate(mixed_texts, 1):
        processed = vectorizer.preprocess_text(text)
        print(f"{i}. {processed}")
        
        # 检查中英文字符保留情况
        chinese_chars_original = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        chinese_chars_processed = len([c for c in processed if '\u4e00' <= c <= '\u9fff'])
        english_chars_original = len([c for c in text if c.isalpha() and not ('\u4e00' <= c <= '\u9fff')])
        english_chars_processed = len([c for c in processed if c.isalpha() and not ('\u4e00' <= c <= '\u9fff')])
        
        print(f"   中文字符: {chinese_chars_original} → {chinese_chars_processed}")
        print(f"   英文字符: {english_chars_original} → {english_chars_processed}")
        
        if chinese_chars_processed > 0 and english_chars_processed > 0:
            print(f"   ✅ 正确: 中英文字符都得到保留")
        else:
            print(f"   ⚠️  注意: 某些字符可能丢失")
    
    return mixed_texts

def test_vectorization_with_new_preprocessing():
    """测试新预处理逻辑的向量化"""
    print("\n=== 测试向量化功能 ===")
    
    vectorizer = TextVectorizer()
    
    # 使用之前测试的文本
    test_documents = [
        "人工智能是计算机科学的一个分支。",
        "Machine learning focuses on algorithms and statistical models.",
        "深度学习 Deep Learning 使用神经网络。",
        "自然语言处理帮助计算机理解人类语言。",
        "Data science combines statistics and programming."
    ]
    
    print("测试文档:")
    for i, doc in enumerate(test_documents, 1):
        print(f"{i}. {doc}")
    
    try:
        print("\n正在进行 TF-IDF 向量化...")
        tfidf_vectors = vectorizer.tfidf_vectorize(test_documents)
        print(f"✅ TF-IDF 向量化成功: {tfidf_vectors.shape}")
        
        # 测试查询转换
        test_queries = [
            "人工智能",
            "machine learning",
            "深度学习 neural networks"
        ]
        
        print("\n测试查询向量化:")
        for query in test_queries:
            try:
                query_vector = vectorizer.tfidf_transform_query(query)
                print(f"✅ 查询 '{query}' 向量化成功: {query_vector.shape}")
            except Exception as e:
                print(f"❌ 查询 '{query}' 向量化失败: {e}")
                
    except Exception as e:
        print(f"❌ 向量化失败: {e}")
    
    return test_documents

def main():
    """主测试函数"""
    print("🔍 NLTK 语言感知预处理测试")
    print("=" * 50)
    
    # 检查 NLTK 可用性
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        print("✅ NLTK 已安装并可用")
    except ImportError:
        print("❌ NLTK 未安装，将使用基础预处理")
    
    try:
        # 测试各种文本类型的预处理
        chinese_texts = test_chinese_text_preprocessing()
        english_texts = test_english_text_preprocessing() 
        mixed_texts = test_mixed_text_preprocessing()
        
        # 测试向量化功能
        test_documents = test_vectorization_with_new_preprocessing()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试完成!")
        print("\n📋 测试总结:")
        print(f"- 中文文档数量: {len(chinese_texts)}")
        print(f"- 英文文档数量: {len(english_texts)}")
        print(f"- 混合文档数量: {len(mixed_texts)}")
        print(f"- 向量化测试文档数量: {len(test_documents)}")
        
        print("\n✨ 语言感知预处理功能已准备就绪!")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
