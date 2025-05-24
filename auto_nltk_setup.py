#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动下载NLTK数据包
"""

import nltk
import sys

def auto_download_nltk():
    """自动下载NLTK数据包"""
    print("正在自动下载NLTK数据包...")
    
    # 必要的数据包列表
    required_packages = [
        'punkt',           # 分词器
        'stopwords',       # 停用词
        'wordnet',         # WordNet词典
        'averaged_perceptron_tagger',  # 词性标注器
    ]
    
    successful = 0
    total = len(required_packages)
    
    for package in required_packages:
        try:
            print(f"下载 {package}...")
            nltk.download(package, quiet=True)
            print(f"✅ {package} 成功")
            successful += 1
        except Exception as e:
            print(f"❌ {package} 失败: {e}")
    
    print(f"\n下载完成: {successful}/{total} 个包成功下载")
    
    # 测试功能
    print("\n测试NLTK功能...")
    try:
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        
        # 简单测试
        tokens = word_tokenize("Hello world")
        stop_words = stopwords.words('english')
        stemmer = PorterStemmer()
        
        print("✅ NLTK功能测试通过")
        return True
    except Exception as e:
        print(f"❌ NLTK功能测试失败: {e}")
        return False

if __name__ == "__main__":
    auto_download_nltk()
