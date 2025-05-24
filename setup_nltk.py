#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLTK数据下载和配置脚本
NLTK Data Download and Configuration Script

本脚本用于下载必要的NLTK数据包，确保文本预处理功能正常工作。
"""

import os
import sys
import nltk
from pathlib import Path

def download_nltk_data():
    """下载必要的NLTK数据包"""
    print("开始下载NLTK数据包...")
    
    # 设置NLTK数据下载目录（可选）
    nltk_data_dir = Path.home() / 'nltk_data'
    print(f"NLTK数据将存储在: {nltk_data_dir}")
    
    # 必要的数据包列表
    required_packages = [
        'punkt',           # 分词器
        'stopwords',       # 停用词
        'wordnet',         # WordNet词典
        'averaged_perceptron_tagger',  # 词性标注器
        'punkt_tab',       # Punkt分词器表格（新版本需要）
        'omw-1.4',         # 开放多语言WordNet
    ]
    
    successful_downloads = []
    failed_downloads = []
    
    for package in required_packages:
        try:
            print(f"正在下载 {package}...")
            nltk.download(package, quiet=False)
            successful_downloads.append(package)
            print(f"✅ {package} 下载成功")
        except Exception as e:
            print(f"❌ {package} 下载失败: {e}")
            failed_downloads.append(package)
    
    print(f"\n下载总结:")
    print(f"成功下载: {len(successful_downloads)} 个包")
    for package in successful_downloads:
        print(f"  ✅ {package}")
    
    if failed_downloads:
        print(f"下载失败: {len(failed_downloads)} 个包")
        for package in failed_downloads:
            print(f"  ❌ {package}")
    else:
        print("🎉 所有必要的NLTK数据包都已成功下载！")
    
    return len(failed_downloads) == 0

def test_nltk_functionality():
    """测试NLTK功能是否正常工作"""
    print("\n开始测试NLTK功能...")
    
    try:
        # 测试分词
        from nltk.tokenize import word_tokenize
        test_text = "This is a test sentence for NLTK functionality."
        tokens = word_tokenize(test_text)
        print(f"✅ 分词测试成功: {tokens[:5]}...")
        
        # 测试停用词
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        print(f"✅ 停用词测试成功: 加载了 {len(stop_words)} 个英文停用词")
        
        # 测试词干提取
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        stem_test = stemmer.stem("running")
        print(f"✅ 词干提取测试成功: 'running' -> '{stem_test}'")
        
        print("🎉 所有NLTK功能测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ NLTK功能测试失败: {e}")
        return False

def show_nltk_info():
    """显示NLTK配置信息"""
    print("\nNLTK配置信息:")
    print(f"NLTK版本: {nltk.__version__}")
    print(f"NLTK数据路径: {nltk.data.path}")
    
    # 检查已安装的数据包
    try:
        from nltk.data import find
        installed_packages = []
        test_packages = ['tokenizers/punkt', 'corpora/stopwords', 'corpora/wordnet']
        
        for package in test_packages:
            try:
                find(package)
                installed_packages.append(package.split('/')[-1])
            except LookupError:
                pass
        
        if installed_packages:
            print(f"已安装的数据包: {', '.join(installed_packages)}")
        else:
            print("未检测到已安装的数据包")
            
    except Exception as e:
        print(f"无法检查已安装的数据包: {e}")

def main():
    """主函数"""
    print("=== NLTK数据下载和配置工具 ===")
    print("此工具将下载必要的NLTK数据包以支持文本预处理功能。\n")
    
    # 显示当前NLTK信息
    show_nltk_info()
    
    # 询问用户是否继续
    response = input("\n是否开始下载NLTK数据包？(y/n): ").lower().strip()
    if response not in ['y', 'yes', '是']:
        print("取消下载。")
        return
    
    # 下载数据包
    success = download_nltk_data()
    
    if success:
        # 测试功能
        test_success = test_nltk_functionality()
        
        if test_success:
            print("\n🎉 NLTK配置完成！现在可以使用完整的文本预处理功能了。")
        else:
            print("\n⚠️ NLTK数据下载成功，但功能测试失败。请检查配置。")
    else:
        print("\n❌ 部分NLTK数据包下载失败。请检查网络连接并重试。")
        print("您也可以手动下载失败的数据包：")
        print("python -c \"import nltk; nltk.download('数据包名称')\"")

if __name__ == "__main__":
    main()
