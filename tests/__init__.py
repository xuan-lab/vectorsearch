#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试包初始化文件
Test Package Initialization

这个包包含了向量搜索项目的所有测试代码。

测试模块:
- test_vector_search.py: 核心功能测试
- test_examples.py: 示例应用测试
- test_notebooks.py: Jupyter笔记本测试

运行所有测试:
    python -m pytest tests/
    
或者运行特定测试:
    python tests/test_vector_search.py

This package contains all test code for the vector search project.

Test modules:
- test_vector_search.py: Core functionality tests
- test_examples.py: Example application tests  
- test_notebooks.py: Jupyter notebook tests

Run all tests:
    python -m pytest tests/
    
Or run specific tests:
    python tests/test_vector_search.py
"""

__version__ = "1.0.0"
__author__ = "Vector Search Project"
__description__ = "向量搜索项目测试套件 / Vector Search Project Test Suite"

# 测试配置
TEST_CONFIG = {
    'timeout': 30,  # 测试超时时间（秒）
    'verbose': True,  # 详细输出
    'coverage': True,  # 代码覆盖率
}

# 测试数据配置
TEST_DATA_CONFIG = {
    'sample_doc_count': 100,  # 样本文档数量
    'vector_dimension': 100,  # 向量维度
    'similarity_threshold': 0.5,  # 相似度阈值
}

def setup_test_environment():
    """设置测试环境"""
    import os
    import sys
    
    # 确保项目根目录在Python路径中
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 设置测试环境变量
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'WARNING'  # 减少测试时的日志输出
    
    print("测试环境设置完成")

def cleanup_test_environment():
    """清理测试环境"""
    import os
    
    # 清理环境变量
    if 'TESTING' in os.environ:
        del os.environ['TESTING']
    if 'LOG_LEVEL' in os.environ:
        del os.environ['LOG_LEVEL']
    
    print("测试环境清理完成")

# 自动设置测试环境
setup_test_environment()
