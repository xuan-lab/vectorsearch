#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多语言文本处理教育应用
Multilingual Text Processing Educational Application

这个应用展示了多语言文本处理的各种技术：
- 语言检测
- 多语言文本分析
- 跨语言相似性计算
- 机器翻译
- 多语言情感分析

This application demonstrates various multilingual text processing techniques:
- Language detection
- Multilingual text analysis
- Cross-lingual similarity computation
- Machine translation
- Multilingual sentiment analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import re
import time
from collections import defaultdict, Counter
import json

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.text_vectorizer import TextVectorizer
from src.utils import load_documents

# 尝试导入语言检测库
try:
    from langdetect import detect, detect_langs, DetectorFactory
    DetectorFactory.seed = 0  # 确保结果可重现
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("提示: 安装langdetect库以进行语言检测: pip install langdetect")

# 尝试导入翻译库
try:
    from googletrans import Translator
    HAS_GOOGLETRANS = True
except ImportError:
    HAS_GOOGLETRANS = False
    print("提示: 安装googletrans库以进行机器翻译: pip install googletrans==4.0.0-rc1")

# 尝试导入多语言BERT
try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("提示: 安装transformers库以使用多语言BERT: pip install transformers")

# 尝试导入句子变换器
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("提示: 安装sentence-transformers库以进行跨语言相似性计算: pip install sentence-transformers")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class LanguageDetectionResult:
    """语言检测结果"""
    text: str
    detected_language: str
    confidence: float
    all_probabilities: Dict[str, float]

@dataclass
class TranslationResult:
    """翻译结果"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float = 1.0

@dataclass
class MultilingualSimilarity:
    """多语言相似性结果"""
    text1: str
    text2: str
    language1: str
    language2: str
    similarity: float
    method: str

class LanguageDetector:
    """语言检测器"""
    
    def __init__(self):
        self.language_names = {
            'zh-cn': '中文',
            'zh': '中文',
            'en': '英语',
            'ja': '日语',
            'ko': '韩语',
            'fr': '法语',
            'de': '德语',
            'es': '西班牙语',
            'it': '意大利语',
            'pt': '葡萄牙语',
            'ru': '俄语',
            'ar': '阿拉伯语',
            'hi': '印地语',
            'th': '泰语',
            'vi': '越南语'
        }
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """检测文本语言"""
        # 简单的基于字符的检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return LanguageDetectionResult(
                text=text,
                detected_language='unknown',
                confidence=0.0,
                all_probabilities={'unknown': 1.0}
            )
        
        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        probabilities = {}
        
        if HAS_LANGDETECT and len(text.strip()) > 3:
            try:
                # 使用langdetect库
                detected_lang = detect(text)
                lang_probs = detect_langs(text)
                
                probabilities = {}
                for lang_prob in lang_probs:
                    lang_code = lang_prob.lang
                    probabilities[lang_code] = lang_prob.prob
                
                confidence = max(probabilities.values())
                
                return LanguageDetectionResult(
                    text=text,
                    detected_language=detected_lang,
                    confidence=confidence,
                    all_probabilities=probabilities
                )
            
            except Exception as e:
                print(f"语言检测失败，使用简单规则: {e}")
        
        # 基于规则的简单检测
        if chinese_ratio > 0.3:
            detected_lang = 'zh'
            confidence = chinese_ratio
            probabilities = {'zh': chinese_ratio, 'en': english_ratio}
        elif english_ratio > 0.5:
            detected_lang = 'en'
            confidence = english_ratio
            probabilities = {'en': english_ratio, 'zh': chinese_ratio}
        else:
            detected_lang = 'unknown'
            confidence = 0.5
            probabilities = {'unknown': 0.5, 'en': english_ratio, 'zh': chinese_ratio}
        
        return LanguageDetectionResult(
            text=text,
            detected_language=detected_lang,
            confidence=confidence,
            all_probabilities=probabilities
        )
    
    def get_language_name(self, lang_code: str) -> str:
        """获取语言名称"""
        return self.language_names.get(lang_code, lang_code)

class MachineTranslator:
    """机器翻译器"""
    
    def __init__(self):
        self.translator = None
        if HAS_GOOGLETRANS:
            try:
                self.translator = Translator()
            except Exception as e:
                print(f"初始化翻译器失败: {e}")
        
        # 语言代码映射
        self.language_codes = {
            '中文': 'zh',
            '英语': 'en',
            '日语': 'ja',
            '韩语': 'ko',
            '法语': 'fr',
            '德语': 'de',
            '西班牙语': 'es',
            '意大利语': 'it',
            '葡萄牙语': 'pt',
            '俄语': 'ru'
        }
    
    def translate_text(self, text: str, target_lang: str, source_lang: str = None) -> TranslationResult:
        """翻译文本"""
        if not self.translator:
            # 简单的模拟翻译
            return self._mock_translation(text, target_lang, source_lang)
        
        try:
            # 检测源语言
            if source_lang is None:
                detection = self.translator.detect(text)
                source_lang = detection.lang
                confidence = detection.confidence
            else:
                confidence = 1.0
            
            # 执行翻译
            translation = self.translator.translate(text, dest=target_lang, src=source_lang)
            
            return TranslationResult(
                original_text=text,
                translated_text=translation.text,
                source_language=source_lang,
                target_language=target_lang,
                confidence=confidence
            )
        
        except Exception as e:
            print(f"翻译失败: {e}")
            return self._mock_translation(text, target_lang, source_lang)
    
    def _mock_translation(self, text: str, target_lang: str, source_lang: str) -> TranslationResult:
        """模拟翻译（当真实翻译不可用时）"""
        # 简单的替换示例
        mock_translations = {
            ('zh', 'en'): {
                '你好': 'Hello',
                '谢谢': 'Thank you',
                '再见': 'Goodbye',
                '人工智能': 'Artificial Intelligence',
                '机器学习': 'Machine Learning',
                '深度学习': 'Deep Learning'
            },
            ('en', 'zh'): {
                'Hello': '你好',
                'Thank you': '谢谢',
                'Goodbye': '再见',
                'Artificial Intelligence': '人工智能',
                'Machine Learning': '机器学习',
                'Deep Learning': '深度学习'
            }
        }
        
        if source_lang is None:
            source_lang = 'auto'
        
        translation_dict = mock_translations.get((source_lang, target_lang), {})
        translated_text = text
        
        for original, translation in translation_dict.items():
            translated_text = translated_text.replace(original, translation)
        
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=source_lang,
            target_language=target_lang,
            confidence=0.8
        )

class CrosslingualSimilarity:
    """跨语言相似性计算"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = TextVectorizer()
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                # 使用多语言sentence transformer模型
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("加载多语言相似性模型成功")
            except Exception as e:
                print(f"加载多语言模型失败: {e}")
    
    def calculate_similarity(self, text1: str, text2: str, lang1: str = None, lang2: str = None) -> MultilingualSimilarity:
        """计算跨语言相似性"""
        if self.model:
            try:
                # 使用sentence transformer计算相似性
                embeddings = self.model.encode([text1, text2])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                method = 'sentence_transformer'
            except Exception as e:
                print(f"使用模型计算相似性失败: {e}")
                similarity = self._calculate_simple_similarity(text1, text2)
                method = 'simple'
        else:
            similarity = self._calculate_simple_similarity(text1, text2)
            method = 'simple'
        
        return MultilingualSimilarity(
            text1=text1,
            text2=text2,
            language1=lang1 or 'unknown',
            language2=lang2 or 'unknown',
            similarity=similarity,
            method=method
        )
    
    def _calculate_simple_similarity(self, text1: str, text2: str) -> float:
        """简单的相似性计算（基于字符重叠）"""
        # 将文本转换为字符集合
        chars1 = set(text1.lower())
        chars2 = set(text2.lower())
        
        # 计算Jaccard相似性
        intersection = len(chars1 & chars2)
        union = len(chars1 | chars2)
        
        return intersection / union if union > 0 else 0.0

class MultilingualSentiment:
    """多语言情感分析"""
    
    def __init__(self):
        self.sentiment_model = None
        
        if HAS_TRANSFORMERS:
            try:
                # 使用多语言情感分析模型
                self.sentiment_model = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                    return_all_scores=True
                )
                print("加载多语言情感分析模型成功")
            except Exception as e:
                print(f"加载多语言情感分析模型失败: {e}")
    
    def analyze_sentiment(self, text: str, language: str = None) -> Dict[str, Any]:
        """分析多语言情感"""
        if self.sentiment_model:
            try:
                results = self.sentiment_model(text)
                if isinstance(results[0], list):
                    results = results[0]
                
                sentiment_scores = {result['label']: result['score'] for result in results}
                predicted_sentiment = max(sentiment_scores, key=sentiment_scores.get)
                confidence = sentiment_scores[predicted_sentiment]
                
                return {
                    'text': text,
                    'language': language,
                    'sentiment': predicted_sentiment,
                    'confidence': confidence,
                    'scores': sentiment_scores,
                    'method': 'multilingual_transformer'
                }
            except Exception as e:
                print(f"多语言情感分析失败: {e}")
        
        # 简单的基于词典的情感分析
        positive_words = {
            'zh': ['好', '棒', '优秀', '喜欢', '高兴', '快乐', '满意', '赞'],
            'en': ['good', 'great', 'excellent', 'like', 'happy', 'love', 'satisfied', 'amazing']
        }
        
        negative_words = {
            'zh': ['坏', '差', '讨厌', '生气', '失望', '糟糕', '不满', '愤怒'],
            'en': ['bad', 'terrible', 'hate', 'angry', 'disappointed', 'awful', 'dissatisfied', 'furious']
        }
        
        lang = language or 'en'
        pos_words = positive_words.get(lang, positive_words['en'])
        neg_words = negative_words.get(lang, negative_words['en'])
        
        text_lower = text.lower()
        pos_count = sum(1 for word in pos_words if word in text_lower)
        neg_count = sum(1 for word in neg_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = 'POSITIVE'
            confidence = pos_count / (pos_count + neg_count + 1)
        elif neg_count > pos_count:
            sentiment = 'NEGATIVE'
            confidence = neg_count / (pos_count + neg_count + 1)
        else:
            sentiment = 'NEUTRAL'
            confidence = 0.5
        
        return {
            'text': text,
            'language': language,
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': {sentiment: confidence},
            'method': 'lexicon_based'
        }

class MultilingualApp:
    """多语言文本处理教育应用"""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.translator = MachineTranslator()
        self.similarity_calculator = CrosslingualSimilarity()
        self.sentiment_analyzer = MultilingualSentiment()
        self.processed_texts = []
        
    def analyze_text_languages(self, texts: List[str]) -> List[LanguageDetectionResult]:
        """分析文本语言"""
        results = []
        
        print("分析文本语言...")
        for i, text in enumerate(texts):
            result = self.language_detector.detect_language(text)
            results.append(result)
            
            lang_name = self.language_detector.get_language_name(result.detected_language)
            print(f"文本 {i+1}: {lang_name} (置信度: {result.confidence:.3f})")
            print(f"  内容: {text[:50]}...")
        
        return results
    
    def translate_texts(self, texts: List[str], target_language: str) -> List[TranslationResult]:
        """翻译文本"""
        results = []
        
        print(f"翻译文本到 {target_language}...")
        for i, text in enumerate(texts):
            result = self.translator.translate_text(text, target_language)
            results.append(result)
            
            print(f"文本 {i+1}:")
            print(f"  原文: {result.original_text}")
            print(f"  译文: {result.translated_text}")
            print(f"  置信度: {result.confidence:.3f}")
        
        return results
    
    def compare_crosslingual_similarity(self, text_pairs: List[Tuple[str, str]]) -> List[MultilingualSimilarity]:
        """比较跨语言相似性"""
        results = []
        
        print("计算跨语言相似性...")
        for i, (text1, text2) in enumerate(text_pairs):
            # 检测语言
            lang1_result = self.language_detector.detect_language(text1)
            lang2_result = self.language_detector.detect_language(text2)
            
            # 计算相似性
            similarity = self.similarity_calculator.calculate_similarity(
                text1, text2, 
                lang1_result.detected_language, 
                lang2_result.detected_language
            )
            results.append(similarity)
            
            lang1_name = self.language_detector.get_language_name(lang1_result.detected_language)
            lang2_name = self.language_detector.get_language_name(lang2_result.detected_language)
            
            print(f"文本对 {i+1}:")
            print(f"  文本1 ({lang1_name}): {text1}")
            print(f"  文本2 ({lang2_name}): {text2}")
            print(f"  相似性: {similarity.similarity:.3f} ({similarity.method})")
        
        return results
    
    def analyze_multilingual_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """分析多语言情感"""
        results = []
        
        print("分析多语言情感...")
        for i, text in enumerate(texts):
            # 检测语言
            lang_result = self.language_detector.detect_language(text)
            
            # 分析情感
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(
                text, lang_result.detected_language
            )
            results.append(sentiment_result)
            
            lang_name = self.language_detector.get_language_name(lang_result.detected_language)
            
            print(f"文本 {i+1} ({lang_name}):")
            print(f"  内容: {text}")
            print(f"  情感: {sentiment_result['sentiment']}")
            print(f"  置信度: {sentiment_result['confidence']:.3f}")
        
        return results
    
    def visualize_language_distribution(self, texts: List[str]):
        """可视化语言分布"""
        # 检测所有文本的语言
        language_counts = Counter()
        confidence_scores = []
        
        for text in texts:
            result = self.language_detector.detect_language(text)
            lang_name = self.language_detector.get_language_name(result.detected_language)
            language_counts[lang_name] += 1
            confidence_scores.append(result.confidence)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 语言分布饼图
        if language_counts:
            languages = list(language_counts.keys())
            counts = list(language_counts.values())
            
            axes[0].pie(counts, labels=languages, autopct='%1.1f%%')
            axes[0].set_title('语言分布')
        
        # 置信度分布直方图
        if confidence_scores:
            axes[1].hist(confidence_scores, bins=20, alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('置信度')
            axes[1].set_ylabel('频次')
            axes[1].set_title('语言检测置信度分布')
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print(f"\n语言统计:")
        for lang, count in language_counts.most_common():
            percentage = count / len(texts) * 100
            print(f"  {lang}: {count} 个文本 ({percentage:.1f}%)")
        
        if confidence_scores:
            print(f"\n置信度统计:")
            print(f"  平均置信度: {np.mean(confidence_scores):.3f}")
            print(f"  最低置信度: {np.min(confidence_scores):.3f}")
            print(f"  最高置信度: {np.max(confidence_scores):.3f}")
    
    def create_translation_comparison(self, texts: List[str], target_languages: List[str]):
        """创建翻译比较表"""
        print("创建翻译比较表...")
        
        # 检测原始语言
        original_results = []
        for text in texts:
            lang_result = self.language_detector.detect_language(text)
            original_results.append(lang_result)
        
        # 创建DataFrame
        data = []
        
        for i, text in enumerate(texts):
            row = {
                '原文': text[:50] + '...' if len(text) > 50 else text,
                '原始语言': self.language_detector.get_language_name(original_results[i].detected_language)
            }
            
            # 翻译到各目标语言
            for target_lang in target_languages:
                translation_result = self.translator.translate_text(text, target_lang)
                lang_name = self.language_detector.get_language_name(target_lang)
                row[f'翻译到{lang_name}'] = translation_result.translated_text[:50] + '...' if len(translation_result.translated_text) > 50 else translation_result.translated_text
            
            data.append(row)
        
        df = pd.DataFrame(data)
        print("\n翻译比较表:")
        print(df.to_string(index=False))
        
        return df
    
    def run_interactive_demo(self):
        """运行交互式演示"""
        print("\n🌍 多语言文本处理教育应用")
        print("=" * 50)
        
        sample_texts = [
            "Hello, how are you today?",
            "你好，今天过得怎么样？",
            "こんにちは、今日はどうですか？",
            "안녕하세요, 오늘 어떠세요?",
            "Bonjour, comment allez-vous aujourd'hui?"
        ]
        
        while True:
            print("\n选择操作:")
            print("1. 语言检测")
            print("2. 机器翻译")
            print("3. 跨语言相似性")
            print("4. 多语言情感分析")
            print("5. 语言分布可视化")
            print("6. 翻译比较表")
            print("7. 使用示例文本演示")
            print("8. 加载自定义文本")
            print("0. 退出")
            
            choice = input("\n请选择 (0-8): ").strip()
            
            if choice == '0':
                break
            
            elif choice == '1':
                text = input("请输入要检测语言的文本: ").strip()
                if text:
                    result = self.language_detector.detect_language(text)
                    lang_name = self.language_detector.get_language_name(result.detected_language)
                    print(f"\n检测结果:")
                    print(f"  语言: {lang_name}")
                    print(f"  置信度: {result.confidence:.3f}")
                    print(f"  所有概率:")
                    for lang, prob in result.all_probabilities.items():
                        lang_name = self.language_detector.get_language_name(lang)
                        print(f"    {lang_name}: {prob:.3f}")
            
            elif choice == '2':
                text = input("请输入要翻译的文本: ").strip()
                if text:
                    target_lang = input("目标语言代码 (en/zh/ja/ko/fr/de/es): ").strip()
                    if target_lang:
                        result = self.translator.translate_text(text, target_lang)
                        print(f"\n翻译结果:")
                        print(f"  原文: {result.original_text}")
                        print(f"  译文: {result.translated_text}")
                        print(f"  源语言: {result.source_language}")
                        print(f"  目标语言: {result.target_language}")
                        print(f"  置信度: {result.confidence:.3f}")
            
            elif choice == '3':
                text1 = input("请输入第一个文本: ").strip()
                text2 = input("请输入第二个文本: ").strip()
                if text1 and text2:
                    results = self.compare_crosslingual_similarity([(text1, text2)])
                    result = results[0]
                    print(f"\n相似性分析:")
                    print(f"  文本1语言: {result.language1}")
                    print(f"  文本2语言: {result.language2}")
                    print(f"  相似性分数: {result.similarity:.3f}")
                    print(f"  计算方法: {result.method}")
            
            elif choice == '4':
                text = input("请输入要分析情感的文本: ").strip()
                if text:
                    results = self.analyze_multilingual_sentiment([text])
                    result = results[0]
                    print(f"\n情感分析结果:")
                    print(f"  语言: {result['language']}")
                    print(f"  情感: {result['sentiment']}")
                    print(f"  置信度: {result['confidence']:.3f}")
                    print(f"  方法: {result['method']}")
            
            elif choice == '5':
                print("请输入多个文本，每行一个 (输入空行结束):")
                texts = []
                while True:
                    line = input().strip()
                    if not line:
                        break
                    texts.append(line)
                
                if texts:
                    self.visualize_language_distribution(texts)
                else:
                    print("没有输入文本")
            
            elif choice == '6':
                print("请输入多个文本，每行一个 (输入空行结束):")
                texts = []
                while True:
                    line = input().strip()
                    if not line:
                        break
                    texts.append(line)
                
                if texts:
                    target_langs = input("目标语言代码 (用逗号分隔, 如: en,zh,ja): ").strip().split(',')
                    target_langs = [lang.strip() for lang in target_langs if lang.strip()]
                    if target_langs:
                        self.create_translation_comparison(texts, target_langs)
                    else:
                        print("没有指定目标语言")
                else:
                    print("没有输入文本")
            
            elif choice == '7':
                print("使用示例文本进行演示...")
                
                # 语言检测
                print("\n1. 语言检测演示:")
                self.analyze_text_languages(sample_texts)
                
                # 翻译演示
                print("\n2. 翻译演示:")
                self.translate_texts(sample_texts[:3], 'zh')
                
                # 相似性演示
                print("\n3. 跨语言相似性演示:")
                text_pairs = [
                    ("Hello world", "你好世界"),
                    ("Good morning", "おはようございます"),
                    ("Thank you", "감사합니다")
                ]
                self.compare_crosslingual_similarity(text_pairs)
                
                # 情感分析演示
                print("\n4. 多语言情感分析演示:")
                sentiment_texts = [
                    "I love this product!",
                    "这个产品真的很棒！",
                    "Cette application est terrible.",
                    "이 서비스는 정말 좋습니다."
                ]
                self.analyze_multilingual_sentiment(sentiment_texts)
                
                # 可视化
                print("\n5. 语言分布可视化:")
                self.visualize_language_distribution(sample_texts + sentiment_texts)
            
            elif choice == '8':
                try:
                    file_path = input("请输入文本文件路径 (TXT或JSON): ").strip()
                    if os.path.exists(file_path):
                        if file_path.endswith('.json'):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            if isinstance(data, list):
                                if isinstance(data[0], dict) and 'text' in data[0]:
                                    texts = [item['text'] for item in data]
                                elif isinstance(data[0], str):
                                    texts = data
                                else:
                                    print("JSON格式错误")
                                    continue
                            else:
                                print("JSON数据格式错误")
                                continue
                        
                        else:  # TXT文件
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            texts = [line.strip() for line in content.split('\n') if line.strip()]
                        
                        print(f"加载了 {len(texts)} 个文本")
                        
                        # 分析语言分布
                        self.visualize_language_distribution(texts)
                        
                        # 询问是否进行其他分析
                        if input("是否进行情感分析? (y/n): ").strip().lower() == 'y':
                            self.analyze_multilingual_sentiment(texts[:5])  # 限制数量
                    
                    else:
                        print("文件不存在")
                
                except Exception as e:
                    print(f"加载文件失败: {e}")

def main():
    """主函数"""
    print("初始化多语言文本处理应用...")
    
    app = MultilingualApp()
    
    # 示例演示
    print("\n🎯 演示: 多语言文本处理")
    print("=" * 40)
    
    sample_texts = [
        "Artificial intelligence is changing the world.",
        "人工智能正在改变世界。",
        "人工知能は世界を変えています。",
        "인공지능이 세상을 바꾸고 있습니다.",
        "L'intelligence artificielle change le monde."
    ]
    
    # 语言检测演示
    print("\n📍 语言检测演示:")
    app.analyze_text_languages(sample_texts)
    
    # 可视化语言分布
    print("\n📊 语言分布可视化:")
    app.visualize_language_distribution(sample_texts)
    
    # 跨语言相似性演示
    print("\n🔗 跨语言相似性演示:")
    similar_pairs = [
        ("Hello world", "你好世界"),
        ("Machine learning", "機械学習"),
        ("Good morning", "Buenos días")
    ]
    app.compare_crosslingual_similarity(similar_pairs)
    
    # 运行交互式演示
    app.run_interactive_demo()

if __name__ == "__main__":
    main()
