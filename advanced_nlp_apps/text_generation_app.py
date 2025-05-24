#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本生成教育应用
Text Generation Educational Application

这个应用展示了文本生成的各种技术：
- N-gram语言模型
- 马尔可夫链文本生成
- 基于模板的文本生成
- Transformer文本生成
- 文本风格迁移
- 创意写作辅助

This application demonstrates various text generation techniques:
- N-gram language models
- Markov chain text generation
- Template-based text generation
- Transformer text generation
- Text style transfer
- Creative writing assistance
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
import re
import time
import random
from collections import defaultdict, Counter, deque
import json
import pickle

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.text_vectorizer import TextVectorizer
from src.utils import load_documents

# 尝试导入深度学习库
try:
    from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("提示: 安装transformers库以使用深度学习文本生成: pip install transformers")

# 尝试导入NLTK
try:
    import nltk
    from nltk.util import ngrams
    from nltk.tokenize import sent_tokenize, word_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("提示: 安装NLTK库以获得更好的文本处理: pip install nltk")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class GenerationResult:
    """文本生成结果"""
    generated_text: str
    method: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class NGramLanguageModel:
    """N-gram语言模型"""
    
    def __init__(self, n: int = 3):
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.vocabulary = set()
        self.is_trained = False
    
    def train(self, texts: List[str]):
        """训练N-gram模型"""
        print(f"训练 {self.n}-gram 语言模型...")
        
        all_tokens = []
        for text in texts:
            # 简单分词
            tokens = self._tokenize(text)
            all_tokens.extend(tokens)
            self.vocabulary.update(tokens)
        
        # 生成n-grams
        for i in range(len(all_tokens) - self.n + 1):
            prefix = tuple(all_tokens[i:i+self.n-1])
            next_word = all_tokens[i+self.n-1]
            self.ngrams[prefix][next_word] += 1
        
        self.is_trained = True
        print(f"训练完成，词汇量: {len(self.vocabulary)}, N-gram数量: {len(self.ngrams)}")
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 添加句子开始和结束标记
        tokens = ['<START>'] * (self.n - 1)
        
        # 中英文混合分词
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        words = re.findall(r'\w+', text)
        tokens.extend(words)
        tokens.append('<END>')
        
        return tokens
    
    def generate_text(self, prompt: str = "", max_length: int = 100) -> GenerationResult:
        """生成文本"""
        if not self.is_trained:
            return GenerationResult("", "ngram", 0.0, {"error": "模型未训练"})
        
        if prompt:
            tokens = self._tokenize(prompt)[-self.n+1:]  # 获取最后n-1个词作为开始
        else:
            # 随机选择一个开始
            tokens = list(random.choice(list(self.ngrams.keys())))
        
        generated_tokens = tokens[:]
        
        for _ in range(max_length):
            prefix = tuple(generated_tokens[-(self.n-1):])
            
            if prefix not in self.ngrams:
                break
            
            # 根据概率选择下一个词
            next_words = self.ngrams[prefix]
            if not next_words:
                break
            
            # 计算概率分布
            total_count = sum(next_words.values())
            probabilities = [count / total_count for count in next_words.values()]
            
            next_word = np.random.choice(list(next_words.keys()), p=probabilities)
            
            if next_word == '<END>':
                break
            
            generated_tokens.append(next_word)
        
        # 移除特殊标记并重构文本
        filtered_tokens = [token for token in generated_tokens if token not in ['<START>', '<END>']]
        generated_text = ' '.join(filtered_tokens)
        
        # 计算平均概率作为置信度
        confidence = self._calculate_confidence(generated_tokens)
        
        return GenerationResult(
            generated_text,
            "ngram",
            confidence,
            {"n": self.n, "tokens_generated": len(filtered_tokens)}
        )
    
    def _calculate_confidence(self, tokens: List[str]) -> float:
        """计算生成文本的置信度"""
        if len(tokens) < self.n:
            return 0.0
        
        probabilities = []
        for i in range(self.n-1, len(tokens)):
            prefix = tuple(tokens[i-self.n+1:i])
            if prefix in self.ngrams and tokens[i] in self.ngrams[prefix]:
                total_count = sum(self.ngrams[prefix].values())
                prob = self.ngrams[prefix][tokens[i]] / total_count
                probabilities.append(prob)
        
        return np.mean(probabilities) if probabilities else 0.0

class MarkovChainGenerator:
    """马尔可夫链文本生成器"""
    
    def __init__(self, order: int = 2):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.is_trained = False
    
    def train(self, texts: List[str]):
        """训练马尔可夫链"""
        print(f"训练 {self.order} 阶马尔可夫链...")
        
        for text in texts:
            sentences = self._split_sentences(text)
            for sentence in sentences:
                words = self._tokenize(sentence)
                if len(words) >= self.order + 1:
                    for i in range(len(words) - self.order):
                        state = tuple(words[i:i+self.order])
                        next_word = words[i+self.order]
                        self.transitions[state][next_word] += 1
        
        self.is_trained = True
        print(f"训练完成，状态数量: {len(self.transitions)}")
    
    def _split_sentences(self, text: str) -> List[str]:
        """分句"""
        if HAS_NLTK:
            return sent_tokenize(text)
        else:
            return re.split(r'[.!?。！？]', text)
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        if HAS_NLTK:
            return word_tokenize(text.lower())
        else:
            return re.findall(r'\w+', text.lower())
    
    def generate_text(self, prompt: str = "", max_length: int = 100) -> GenerationResult:
        """生成文本"""
        if not self.is_trained:
            return GenerationResult("", "markov", 0.0, {"error": "模型未训练"})
        
        if prompt:
            words = self._tokenize(prompt)
            if len(words) >= self.order:
                current_state = tuple(words[-self.order:])
            else:
                current_state = random.choice(list(self.transitions.keys()))
        else:
            current_state = random.choice(list(self.transitions.keys()))
        
        result = list(current_state)
        
        for _ in range(max_length):
            if current_state not in self.transitions:
                break
            
            next_words = self.transitions[current_state]
            if not next_words:
                break
            
            # 选择下一个词
            total_count = sum(next_words.values())
            probabilities = [count / total_count for count in next_words.values()]
            next_word = np.random.choice(list(next_words.keys()), p=probabilities)
            
            result.append(next_word)
            current_state = tuple(result[-self.order:])
        
        generated_text = ' '.join(result)
        confidence = len(result) / max_length  # 简单的置信度计算
        
        return GenerationResult(
            generated_text,
            "markov",
            confidence,
            {"order": self.order, "words_generated": len(result)}
        )

class TemplateGenerator:
    """基于模板的文本生成器"""
    
    def __init__(self):
        self.templates = {}
        self.word_categories = defaultdict(list)
        self._initialize_templates()
    
    def _initialize_templates(self):
        """初始化模板"""
        self.templates = {
            "product_review": [
                "这个{product}真的很{adjective}，我{verb}它。",
                "我对这个{product}感到{emotion}，因为它{reason}。",
                "{product}的{feature}让人印象深刻，特别是{detail}。"
            ],
            "news_headline": [
                "{location}发生{event}，{result}。",
                "{person}宣布{announcement}，引发{reaction}。",
                "{number}{unit}{item}在{place}{action}。"
            ],
            "story_beginning": [
                "在一个{weather}的{time}，{character}走进了{place}。",
                "{character}从来没有想过{event}会发生在{situation}。",
                "当{character}看到{object}时，{emotion}涌上心头。"
            ]
        }
        
        # 词汇分类
        self.word_categories = {
            "product": ["手机", "电脑", "书籍", "电影", "餐厅", "汽车"],
            "adjective": ["棒", "差", "贵", "便宜", "实用", "漂亮"],
            "verb": ["喜欢", "讨厌", "推荐", "购买", "使用", "评价"],
            "emotion": ["满意", "失望", "惊喜", "愤怒", "开心", "担心"],
            "reason": ["质量好", "价格合理", "服务优秀", "功能强大", "设计精美"],
            "feature": ["外观", "性能", "价格", "服务", "功能", "质量"],
            "detail": ["颜色搭配", "运行速度", "性价比", "用户体验"],
            "location": ["北京", "上海", "广州", "深圳", "杭州", "成都"],
            "event": ["会议", "展览", "比赛", "事故", "庆典", "发布会"],
            "result": ["获得成功", "引起关注", "造成影响", "得到好评"],
            "person": ["专家", "官员", "企业家", "科学家", "艺术家"],
            "announcement": ["新政策", "合作计划", "研究成果", "产品发布"],
            "reaction": ["热烈讨论", "广泛关注", "积极响应", "不同看法"],
            "weather": ["晴朗", "阴沉", "下雨", "下雪", "多云"],
            "time": ["早晨", "下午", "傍晚", "深夜", "周末"],
            "character": ["小明", "女孩", "老人", "学生", "商人", "作家"],
            "place": ["图书馆", "咖啡厅", "公园", "商店", "学校", "家中"],
            "object": ["信件", "照片", "钥匙", "书本", "手机", "礼物"],
            "situation": ["这种情况下", "那个时候", "关键时刻", "意外情况下"]
        }
    
    def add_template(self, category: str, template: str):
        """添加模板"""
        if category not in self.templates:
            self.templates[category] = []
        self.templates[category].append(template)
    
    def add_words(self, category: str, words: List[str]):
        """添加词汇"""
        self.word_categories[category].extend(words)
    
    def generate_text(self, template_category: str, custom_values: Dict[str, str] = None) -> GenerationResult:
        """基于模板生成文本"""
        if template_category not in self.templates:
            return GenerationResult("", "template", 0.0, {"error": f"未找到模板类别: {template_category}"})
        
        template = random.choice(self.templates[template_category])
        
        # 查找模板中的占位符
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        # 填充占位符
        filled_template = template
        used_words = {}
        
        for placeholder in placeholders:
            if custom_values and placeholder in custom_values:
                word = custom_values[placeholder]
            elif placeholder in self.word_categories:
                word = random.choice(self.word_categories[placeholder])
            else:
                word = f"[{placeholder}]"  # 未找到对应词汇
            
            used_words[placeholder] = word
            filled_template = filled_template.replace(f"{{{placeholder}}}", word)
        
        confidence = 1.0 - (filled_template.count('[') / len(placeholders)) if placeholders else 1.0
        
        return GenerationResult(
            filled_template,
            "template",
            confidence,
            {
                "template_category": template_category,
                "used_words": used_words,
                "placeholders": placeholders
            }
        )

class TransformerGenerator:
    """基于Transformer的文本生成器"""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.generator = None
        self.is_available = False
        
        if HAS_TRANSFORMERS:
            try:
                print(f"加载 {model_name} 模型...")
                self.generator = pipeline("text-generation", model=model_name)
                self.is_available = True
                print("模型加载成功")
            except Exception as e:
                print(f"模型加载失败: {e}")
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> GenerationResult:
        """生成文本"""
        if not self.is_available:
            return GenerationResult("", "transformer", 0.0, {"error": "Transformer模型不可用"})
        
        try:
            results = self.generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            generated_text = results[0]['generated_text']
            
            # 移除原始prompt
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return GenerationResult(
                generated_text,
                "transformer",
                0.8,  # 固定置信度
                {
                    "model_name": self.model_name,
                    "temperature": temperature,
                    "max_length": max_length
                }
            )
        
        except Exception as e:
            return GenerationResult("", "transformer", 0.0, {"error": str(e)})

class TextStyleTransfer:
    """文本风格迁移"""
    
    def __init__(self):
        self.style_rules = {
            "formal": {
                "replacements": {
                    "很好": "非常优秀",
                    "不错": "令人满意",
                    "糟糕": "不尽如人意",
                    "便宜": "价格合理",
                    "贵": "价格偏高"
                },
                "patterns": [
                    (r"我觉得", "本人认为"),
                    (r"应该", "理应"),
                    (r"可能", "或许"),
                    (r"大概", "大致")
                ]
            },
            "casual": {
                "replacements": {
                    "非常优秀": "超棒",
                    "令人满意": "还不错",
                    "不尽如人意": "有点糟",
                    "价格合理": "便宜",
                    "价格偏高": "太贵了"
                },
                "patterns": [
                    (r"本人认为", "我觉得"),
                    (r"理应", "应该"),
                    (r"或许", "可能"),
                    (r"大致", "大概")
                ]
            },
            "poetic": {
                "replacements": {
                    "美丽": "绚烂",
                    "快乐": "欢愉",
                    "悲伤": "惆怅",
                    "思考": "沉思",
                    "看见": "望见"
                },
                "patterns": [
                    (r"(\w+)很(\w+)", r"\1如此\2"),
                    (r"在(\w+)", r"于\1之中"),
                    (r"走路", "踱步"),
                    (r"说话", "言语")
                ]
            }
        }
    
    def transfer_style(self, text: str, target_style: str) -> GenerationResult:
        """风格迁移"""
        if target_style not in self.style_rules:
            return GenerationResult("", "style_transfer", 0.0, {"error": f"不支持的风格: {target_style}"})
        
        rules = self.style_rules[target_style]
        result_text = text
        changes_made = 0
        
        # 应用词汇替换
        for old_word, new_word in rules["replacements"].items():
            if old_word in result_text:
                result_text = result_text.replace(old_word, new_word)
                changes_made += 1
        
        # 应用模式替换
        for pattern, replacement in rules["patterns"]:
            new_text = re.sub(pattern, replacement, result_text)
            if new_text != result_text:
                result_text = new_text
                changes_made += 1
        
        confidence = min(changes_made / 5.0, 1.0)  # 基于修改数量计算置信度
        
        return GenerationResult(
            result_text,
            "style_transfer",
            confidence,
            {
                "target_style": target_style,
                "changes_made": changes_made,
                "original_length": len(text),
                "result_length": len(result_text)
            }
        )

class TextGenerationApp:
    """文本生成教育应用"""
    
    def __init__(self):
        self.generators = {}
        self.generation_history = []
        
        # 初始化生成器
        self.generators['ngram'] = NGramLanguageModel(n=3)
        self.generators['markov'] = MarkovChainGenerator(order=2)
        self.generators['template'] = TemplateGenerator()
        self.generators['style_transfer'] = TextStyleTransfer()
        
        if HAS_TRANSFORMERS:
            try:
                self.generators['transformer'] = TransformerGenerator()
            except Exception as e:
                print(f"无法初始化Transformer生成器: {e}")
    
    def train_statistical_models(self, texts: List[str] = None):
        """训练统计模型"""
        if texts is None:
            texts = self._create_sample_texts()
        
        print("训练统计文本生成模型...")
        
        # 训练N-gram模型
        if 'ngram' in self.generators:
            self.generators['ngram'].train(texts)
        
        # 训练马尔可夫链
        if 'markov' in self.generators:
            self.generators['markov'].train(texts)
        
        print("统计模型训练完成!")
    
    def _create_sample_texts(self) -> List[str]:
        """创建示例训练文本"""
        return [
            "人工智能技术正在快速发展，为各行各业带来了革命性的变化。机器学习和深度学习算法使计算机能够处理复杂的任务。",
            "自然语言处理是人工智能的重要分支，它让计算机能够理解和生成人类语言。文本生成是其中的一个关键技术。",
            "在教育领域，智能化工具可以帮助学生更好地学习。个性化推荐系统能够根据学生的学习习惯提供定制化内容。",
            "云计算和大数据技术为企业提供了强大的计算能力。这些技术使得处理海量数据成为可能。",
            "移动互联网的普及改变了人们的生活方式。智能手机成为了人们日常生活中不可或缺的工具。",
            "电子商务平台利用推荐算法为用户提供个性化的购物体验。这提高了用户满意度和商业价值。",
            "社交媒体分析可以帮助企业了解消费者的需求和偏好。情感分析技术在这方面发挥了重要作用。",
            "智能家居系统通过物联网技术连接各种设备。语音助手使得人机交互变得更加自然。",
            "在医疗领域，人工智能辅助诊断系统提高了诊断的准确性。图像识别技术在医学影像分析中应用广泛。",
            "自动驾驶技术结合了计算机视觉、传感器融合和路径规划等多种技术。这代表了未来交通的发展方向。"
        ]
    
    def generate_text(self, method: str, prompt: str = "", **kwargs) -> GenerationResult:
        """生成文本"""
        if method not in self.generators:
            return GenerationResult("", method, 0.0, {"error": f"未知的生成方法: {method}"})
        
        generator = self.generators[method]
        
        try:
            if method == "template":
                template_category = kwargs.get("template_category", "story_beginning")
                custom_values = kwargs.get("custom_values", {})
                result = generator.generate_text(template_category, custom_values)
            
            elif method == "style_transfer":
                target_style = kwargs.get("target_style", "formal")
                result = generator.transfer_style(prompt, target_style)
            
            elif method == "transformer":
                max_length = kwargs.get("max_length", 100)
                temperature = kwargs.get("temperature", 0.7)
                result = generator.generate_text(prompt, max_length, temperature)
            
            else:  # ngram, markov
                max_length = kwargs.get("max_length", 100)
                result = generator.generate_text(prompt, max_length)
            
            # 保存到历史记录
            self.generation_history.append(result)
            
            return result
        
        except Exception as e:
            return GenerationResult("", method, 0.0, {"error": str(e)})
    
    def compare_generation_methods(self, prompt: str = "今天天气"):
        """比较不同生成方法"""
        print(f"\n使用提示词: '{prompt}' 比较不同生成方法")
        print("=" * 80)
        
        for method in self.generators.keys():
            if method == "style_transfer":
                continue  # 风格迁移需要完整文本输入
            
            print(f"\n{method.upper()} 方法:")
            print("-" * 50)
            
            try:
                if method == "template":
                    result = self.generate_text(method, template_category="story_beginning")
                elif method == "transformer" and prompt:
                    result = self.generate_text(method, prompt, max_length=50)
                else:
                    result = self.generate_text(method, prompt, max_length=50)
                
                print(f"生成文本: {result.generated_text}")
                print(f"置信度: {result.confidence:.3f}")
                if result.metadata:
                    print(f"元数据: {result.metadata}")
            
            except Exception as e:
                print(f"生成失败: {e}")
    
    def analyze_generation_quality(self, texts: List[str], reference_text: str = None):
        """分析生成质量"""
        print("\n文本生成质量分析:")
        print("=" * 60)
        
        metrics = {}
        
        for i, text in enumerate(texts):
            metrics[f"text_{i+1}"] = {
                "length": len(text),
                "word_count": len(text.split()),
                "sentence_count": len(re.split(r'[.!?。！？]', text)),
                "avg_word_length": np.mean([len(word) for word in text.split()]) if text.split() else 0,
                "uniqueness": len(set(text.split())) / len(text.split()) if text.split() else 0
            }
        
        # 创建可视化
        df = pd.DataFrame(metrics).T
        
        plt.figure(figsize=(15, 10))
        
        # 文本长度分布
        plt.subplot(2, 3, 1)
        plt.bar(range(len(texts)), df['length'])
        plt.title('文本长度分布')
        plt.xlabel('文本编号')
        plt.ylabel('字符数')
        
        # 词汇数量分布
        plt.subplot(2, 3, 2)
        plt.bar(range(len(texts)), df['word_count'])
        plt.title('词汇数量分布')
        plt.xlabel('文本编号')
        plt.ylabel('词汇数')
        
        # 平均词长分布
        plt.subplot(2, 3, 3)
        plt.bar(range(len(texts)), df['avg_word_length'])
        plt.title('平均词长分布')
        plt.xlabel('文本编号')
        plt.ylabel('平均词长')
        
        # 句子数量分布
        plt.subplot(2, 3, 4)
        plt.bar(range(len(texts)), df['sentence_count'])
        plt.title('句子数量分布')
        plt.xlabel('文本编号')
        plt.ylabel('句子数')
        
        # 词汇独特性
        plt.subplot(2, 3, 5)
        plt.bar(range(len(texts)), df['uniqueness'])
        plt.title('词汇独特性')
        plt.xlabel('文本编号')
        plt.ylabel('独特性比例')
        
        # 综合质量雷达图
        plt.subplot(2, 3, 6)
        
        # 归一化指标
        normalized_metrics = df.copy()
        for col in normalized_metrics.columns:
            max_val = normalized_metrics[col].max()
            if max_val > 0:
                normalized_metrics[col] = normalized_metrics[col] / max_val
        
        # 绘制雷达图（简化版）
        avg_metrics = normalized_metrics.mean()
        categories = list(avg_metrics.index)
        values = list(avg_metrics.values)
        
        plt.bar(range(len(categories)), values)
        plt.xticks(range(len(categories)), categories, rotation=45)
        plt.title('平均质量指标')
        plt.ylabel('归一化分数')
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print(f"\n质量统计:")
        print(f"平均文本长度: {df['length'].mean():.1f} 字符")
        print(f"平均词汇数: {df['word_count'].mean():.1f} 词")
        print(f"平均句子数: {df['sentence_count'].mean():.1f} 句")
        print(f"平均词汇独特性: {df['uniqueness'].mean():.3f}")
    
    def creative_writing_assistant(self, theme: str, style: str = "casual"):
        """创意写作辅助"""
        print(f"\n🎨 创意写作辅助 - 主题: {theme}, 风格: {style}")
        print("=" * 60)
        
        # 生成多种开头
        print("1. 故事开头建议:")
        for i in range(3):
            result = self.generate_text("template", template_category="story_beginning")
            print(f"   选项 {i+1}: {result.generated_text}")
        
        # 生成关键词
        print(f"\n2. 主题相关词汇建议:")
        theme_words = self._generate_theme_words(theme)
        print(f"   {', '.join(theme_words)}")
        
        # 生成情节发展
        print(f"\n3. 情节发展建议:")
        plot_suggestions = self._generate_plot_suggestions(theme)
        for i, suggestion in enumerate(plot_suggestions, 1):
            print(f"   {i}. {suggestion}")
        
        # 风格转换示例
        if style != "casual":
            print(f"\n4. {style} 风格转换示例:")
            sample_text = "这个故事很有趣，主人公的经历让人印象深刻。"
            style_result = self.generate_text("style_transfer", sample_text, target_style=style)
            print(f"   原文: {sample_text}")
            print(f"   转换后: {style_result.generated_text}")
    
    def _generate_theme_words(self, theme: str) -> List[str]:
        """根据主题生成相关词汇"""
        theme_mappings = {
            "科幻": ["未来", "科技", "星球", "机器人", "时空", "探索"],
            "爱情": ["浪漫", "邂逅", "心动", "承诺", "思念", "永恒"],
            "冒险": ["旅程", "挑战", "勇气", "发现", "危险", "成长"],
            "悬疑": ["谜团", "线索", "真相", "隐秘", "调查", "揭露"],
            "奇幻": ["魔法", "精灵", "龙族", "法术", "魔法", "异世界"]
        }
        
        return theme_mappings.get(theme, ["故事", "情节", "人物", "背景", "发展", "结局"])
    
    def _generate_plot_suggestions(self, theme: str) -> List[str]:
        """生成情节建议"""
        plot_templates = {
            "科幻": [
                "主人公发现了来自未来的信息",
                "人工智能开始质疑自己的存在",
                "时间旅行者改变了历史进程"
            ],
            "爱情": [
                "两个人在意外的情况下相遇",
                "误会导致了分离，真相带来和解",
                "距离考验着他们的感情"
            ],
            "冒险": [
                "一张神秘地图引领着冒险之旅",
                "伙伴们在困境中展现真正友谊",
                "最大的敌人其实来自内心恐惧"
            ]
        }
        
        return plot_templates.get(theme, [
            "主人公面临重要选择",
            "意外的转折改变了一切",
            "最终的真相令人震惊"
        ])
    
    def run_interactive_demo(self):
        """运行交互式演示"""
        print("\n✨ 文本生成教育应用")
        print("=" * 50)
        print("可用的生成方法:")
        for i, method in enumerate(self.generators.keys(), 1):
            print(f"  {i}. {method}")
        
        # 训练统计模型
        print("\n正在训练统计模型...")
        self.train_statistical_models()
        
        while True:
            print("\n选择操作:")
            print("1. 单一方法生成")
            print("2. 方法比较")
            print("3. 模板生成")
            print("4. 风格转换")
            print("5. 质量分析")
            print("6. 创意写作辅助")
            print("7. 批量生成")
            print("0. 退出")
            
            choice = input("\n请选择 (0-7): ").strip()
            
            if choice == '0':
                break
            
            elif choice == '1':
                method = input(f"选择方法 ({'/'.join(self.generators.keys())}): ").strip()
                if method in self.generators:
                    if method == "style_transfer":
                        text = input("请输入要转换风格的文本: ").strip()
                        style = input("目标风格 (formal/casual/poetic): ").strip()
                        result = self.generate_text(method, text, target_style=style)
                    elif method == "template":
                        category = input("模板类别 (product_review/news_headline/story_beginning): ").strip()
                        result = self.generate_text(method, template_category=category)
                    else:
                        prompt = input("请输入提示词 (可选): ").strip()
                        max_length = int(input("最大长度 (默认100): ").strip() or "100")
                        result = self.generate_text(method, prompt, max_length=max_length)
                    
                    print(f"\n生成结果:")
                    print(f"文本: {result.generated_text}")
                    print(f"方法: {result.method}")
                    print(f"置信度: {result.confidence:.3f}")
                    if result.metadata:
                        print(f"元数据: {result.metadata}")
                else:
                    print("无效的方法")
            
            elif choice == '2':
                prompt = input("请输入提示词: ").strip()
                self.compare_generation_methods(prompt)
            
            elif choice == '3':
                print("可用模板类别:")
                print("- product_review: 产品评价")
                print("- news_headline: 新闻标题")
                print("- story_beginning: 故事开头")
                
                category = input("选择模板类别: ").strip()
                result = self.generate_text("template", template_category=category)
                print(f"\n生成结果: {result.generated_text}")
            
            elif choice == '4':
                text = input("请输入要转换风格的文本: ").strip()
                if text:
                    print("可用风格: formal(正式), casual(随意), poetic(诗意)")
                    style = input("选择目标风格: ").strip()
                    result = self.generate_text("style_transfer", text, target_style=style)
                    print(f"\n原文: {text}")
                    print(f"转换后: {result.generated_text}")
                    print(f"置信度: {result.confidence:.3f}")
            
            elif choice == '5':
                print("生成多个文本进行质量分析...")
                texts = []
                for i in range(5):
                    method = random.choice(['ngram', 'markov', 'template'])
                    if method == 'template':
                        result = self.generate_text(method, template_category="story_beginning")
                    else:
                        result = self.generate_text(method, "今天", max_length=50)
                    texts.append(result.generated_text)
                
                self.analyze_generation_quality(texts)
            
            elif choice == '6':
                theme = input("请输入创作主题 (如: 科幻, 爱情, 冒险): ").strip()
                style = input("请输入风格 (formal/casual/poetic): ").strip() or "casual"
                self.creative_writing_assistant(theme, style)
            
            elif choice == '7':
                method = input(f"选择生成方法 ({'/'.join([m for m in self.generators.keys() if m != 'style_transfer'])}): ").strip()
                if method in self.generators and method != 'style_transfer':
                    count = int(input("生成数量 (1-10): ").strip() or "3")
                    count = min(max(count, 1), 10)
                    
                    print(f"\n批量生成 {count} 个文本:")
                    for i in range(count):
                        if method == "template":
                            categories = ["product_review", "news_headline", "story_beginning"]
                            category = random.choice(categories)
                            result = self.generate_text(method, template_category=category)
                        else:
                            result = self.generate_text(method, max_length=80)
                        
                        print(f"\n文本 {i+1}: {result.generated_text}")
                        print(f"置信度: {result.confidence:.3f}")
                else:
                    print("无效的方法或不支持批量生成")

def main():
    """主函数"""
    print("初始化文本生成应用...")
    
    app = TextGenerationApp()
    app.run_interactive_demo()

if __name__ == "__main__":
    main()
