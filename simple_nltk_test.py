#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

print("开始NLTK测试...")

# 测试分词
tokens = word_tokenize("Hello world! This is a test.")
print(f"分词结果: {tokens}")

# 测试停用词
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w.lower() not in stop_words and w.isalpha()]
print(f"过滤停用词: {filtered}")

# 测试词干提取
stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered]
print(f"词干提取: {stemmed}")

print("✅ NLTK测试完成!")
