#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLTK功能完整测试脚本
"""

def test_nltk_basic():
    """基础NLTK测试"""
    print("开始NLTK基础测试...")
    
    try:
        # 导入测试
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        
        print(f"✅ NLTK版本: {nltk.__version__}")
        
        # 分词测试
        text = "Hello world! This is a test."
        tokens = word_tokenize(text)
        print(f"✅ 分词测试: {tokens}")
        
        # 停用词测试
        stop_words = stopwords.words('english')
        print(f"✅ 停用词数量: {len(stop_words)}")
        
        # 词干提取测试
        stemmer = PorterStemmer()
        stem_result = stemmer.stem("running")
        print(f"✅ 词干提取: running -> {stem_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ NLTK测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=== NLTK功能测试 ===")
    success = test_nltk_basic()
    if success:
        print("🎉 NLTK配置成功！")
    else:
        print("❌ NLTK配置失败！")
