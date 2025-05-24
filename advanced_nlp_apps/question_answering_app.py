#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问答系统教育应用
Question Answering Educational Application

这个应用展示了问答系统的各种技术：
- 基于检索的问答 (Retrieval-based QA)
- 基于生成的问答 (Generative QA)
- 阅读理解问答
- 知识图谱问答
- 多轮对话问答
- FAQ匹配系统

This application demonstrates various QA system techniques:
- Retrieval-based QA
- Generative QA
- Reading comprehension QA
- Knowledge graph QA
- Multi-turn conversational QA
- FAQ matching system
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
import json
from collections import defaultdict, Counter
import difflib

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.text_vectorizer import TextVectorizer
from src.utils import load_documents

# 尝试导入机器学习库
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import LatentDirichletAllocation
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("提示: 安装scikit-learn库以使用机器学习功能: pip install scikit-learn")

# 尝试导入深度学习库
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("提示: 安装transformers库以使用深度学习QA: pip install transformers")

# 尝试导入NLTK
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("提示: 安装NLTK库以获得更好的文本处理: pip install nltk")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class QAResult:
    """问答结果"""
    question: str
    answer: str
    confidence: float
    source: str = ""
    method: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Document:
    """文档"""
    title: str
    content: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class RetrievalBasedQA:
    """基于检索的问答系统"""
    
    def __init__(self):
        self.documents = []
        self.vectorizer = None
        self.doc_vectors = None
        self.is_trained = False
    
    def add_documents(self, documents: List[Document]):
        """添加文档"""
        self.documents.extend(documents)
        self._build_index()
    
    def _build_index(self):
        """构建文档索引"""
        if not self.documents or not HAS_SKLEARN:
            return
        
        print("构建文档索引...")
        doc_texts = [doc.content for doc in self.documents]
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.doc_vectors = self.vectorizer.fit_transform(doc_texts)
        self.is_trained = True
        print(f"索引构建完成，包含 {len(self.documents)} 个文档")
    
    def answer_question(self, question: str, top_k: int = 3) -> QAResult:
        """回答问题"""
        if not self.is_trained:
            return QAResult(question, "系统未训练", 0.0, method="retrieval")
        
        # 向量化问题
        question_vector = self.vectorizer.transform([question])
        
        # 计算相似度
        similarities = cosine_similarity(question_vector, self.doc_vectors).flatten()
        
        # 获取最相关的文档
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        if similarities[top_indices[0]] < 0.1:
            return QAResult(question, "抱歉，我找不到相关信息", 0.0, method="retrieval")
        
        # 从最相关的文档中提取答案
        best_doc = self.documents[top_indices[0]]
        answer = self._extract_answer_from_document(question, best_doc.content)
        
        confidence = float(similarities[top_indices[0]])
        
        return QAResult(
            question=question,
            answer=answer,
            confidence=confidence,
            source=best_doc.title,
            method="retrieval",
            metadata={
                "top_documents": [self.documents[i].title for i in top_indices],
                "similarities": [float(similarities[i]) for i in top_indices]
            }
        )
    
    def _extract_answer_from_document(self, question: str, document: str) -> str:
        """从文档中提取答案"""
        # 分句
        if HAS_NLTK:
            sentences = sent_tokenize(document)
        else:
            sentences = re.split(r'[.!?。！？]', document)
        
        if not sentences:
            return document[:200] + "..."
        
        # 找到最相关的句子
        question_words = set(re.findall(r'\w+', question.lower()))
        
        best_sentence = ""
        max_overlap = 0
        
        for sentence in sentences:
            sentence_words = set(re.findall(r'\w+', sentence.lower()))
            overlap = len(question_words & sentence_words)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = sentence.strip()
        
        return best_sentence if best_sentence else sentences[0].strip()

class GenerativeQA:
    """基于生成的问答系统"""
    
    def __init__(self):
        self.qa_pipeline = None
        self.is_available = False
        
        if HAS_TRANSFORMERS:
            try:
                print("加载问答模型...")
                self.qa_pipeline = pipeline("question-answering")
                self.is_available = True
                print("问答模型加载成功")
            except Exception as e:
                print(f"问答模型加载失败: {e}")
    
    def answer_question(self, question: str, context: str) -> QAResult:
        """基于上下文回答问题"""
        if not self.is_available:
            return QAResult(question, "生成式问答不可用", 0.0, method="generative")
        
        try:
            result = self.qa_pipeline(question=question, context=context)
            
            return QAResult(
                question=question,
                answer=result['answer'],
                confidence=result['score'],
                source="generative_model",
                method="generative",
                metadata={
                    "start": result['start'],
                    "end": result['end'],
                    "context_length": len(context)
                }
            )
        
        except Exception as e:
            return QAResult(question, f"生成失败: {str(e)}", 0.0, method="generative")

class FAQMatcher:
    """FAQ匹配系统"""
    
    def __init__(self):
        self.faq_pairs = []
        self.vectorizer = None
        self.question_vectors = None
        self.is_trained = False
    
    def add_faq_pair(self, question: str, answer: str, category: str = "general"):
        """添加FAQ对"""
        self.faq_pairs.append({
            "question": question,
            "answer": answer,
            "category": category
        })
    
    def build_index(self):
        """构建FAQ索引"""
        if not self.faq_pairs or not HAS_SKLEARN:
            return
        
        print("构建FAQ索引...")
        questions = [pair["question"] for pair in self.faq_pairs]
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            analyzer='char_wb'  # 字符级n-gram，对中文更友好
        )
        
        self.question_vectors = self.vectorizer.fit_transform(questions)
        self.is_trained = True
        print(f"FAQ索引构建完成，包含 {len(self.faq_pairs)} 个问答对")
    
    def find_similar_question(self, question: str, threshold: float = 0.3) -> QAResult:
        """找到相似问题"""
        if not self.is_trained:
            return QAResult(question, "FAQ系统未训练", 0.0, method="faq")
        
        # 向量化输入问题
        question_vector = self.vectorizer.transform([question])
        
        # 计算相似度
        similarities = cosine_similarity(question_vector, self.question_vectors).flatten()
        
        # 找到最相似的问题
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity < threshold:
            return QAResult(question, "抱歉，我没找到相关的常见问题", 0.0, method="faq")
        
        best_faq = self.faq_pairs[best_idx]
        
        return QAResult(
            question=question,
            answer=best_faq["answer"],
            confidence=float(best_similarity),
            source=f"FAQ - {best_faq['category']}",
            method="faq",
            metadata={
                "matched_question": best_faq["question"],
                "category": best_faq["category"],
                "similarity": float(best_similarity)
            }
        )

class KnowledgeGraphQA:
    """知识图谱问答"""
    
    def __init__(self):
        self.entities = {}
        self.relations = []
        self.relation_patterns = {}
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """初始化关系模式"""
        self.relation_patterns = {
            "是什么": ["定义", "解释", "含义"],
            "属于": ["类型", "分类", "归属"],
            "位于": ["地点", "位置", "在哪"],
            "时间": ["什么时候", "何时", "时间"],
            "原因": ["为什么", "原因", "因为"],
            "方式": ["怎么", "如何", "方法"]
        }
    
    def add_entity(self, name: str, entity_type: str, properties: Dict[str, str]):
        """添加实体"""
        self.entities[name] = {
            "type": entity_type,
            "properties": properties
        }
    
    def add_relation(self, subject: str, relation: str, object: str):
        """添加关系"""
        self.relations.append({
            "subject": subject,
            "relation": relation,
            "object": object
        })
    
    def answer_question(self, question: str) -> QAResult:
        """基于知识图谱回答问题"""
        # 简单的模式匹配
        question_lower = question.lower()
        
        # 识别问题类型
        question_type = self._identify_question_type(question_lower)
        
        # 提取实体
        mentioned_entities = []
        for entity_name in self.entities.keys():
            if entity_name.lower() in question_lower:
                mentioned_entities.append(entity_name)
        
        if not mentioned_entities:
            return QAResult(question, "抱歉，我不了解您提到的实体", 0.0, method="knowledge_graph")
        
        # 根据问题类型查找答案
        entity = mentioned_entities[0]
        answer = self._find_answer_by_type(entity, question_type)
        
        confidence = 0.8 if answer != "未找到相关信息" else 0.0
        
        return QAResult(
            question=question,
            answer=answer,
            confidence=confidence,
            source="knowledge_graph",
            method="knowledge_graph",
            metadata={
                "entity": entity,
                "question_type": question_type,
                "mentioned_entities": mentioned_entities
            }
        )
    
    def _identify_question_type(self, question: str) -> str:
        """识别问题类型"""
        for pattern_type, keywords in self.relation_patterns.items():
            for keyword in keywords:
                if keyword in question:
                    return pattern_type
        return "其他"
    
    def _find_answer_by_type(self, entity: str, question_type: str) -> str:
        """根据问题类型查找答案"""
        if entity not in self.entities:
            return "未找到相关信息"
        
        entity_info = self.entities[entity]
        
        # 从实体属性中查找答案
        if question_type == "是什么":
            return entity_info["properties"].get("定义", f"{entity}是一个{entity_info['type']}")
        
        elif question_type == "属于":
            return f"{entity}属于{entity_info['type']}"
        
        elif question_type in entity_info["properties"]:
            return entity_info["properties"][question_type]
        
        # 从关系中查找答案
        for relation in self.relations:
            if relation["subject"] == entity:
                if question_type in relation["relation"]:
                    return f"{entity}{relation['relation']}{relation['object']}"
        
        return "未找到相关信息"

class ConversationalQA:
    """多轮对话问答"""
    
    def __init__(self):
        self.conversation_history = []
        self.context_window = 5  # 保留最近5轮对话
        self.user_profile = {}
    
    def answer_question(self, question: str, user_id: str = "default") -> QAResult:
        """在对话上下文中回答问题"""
        # 更新对话历史
        self.conversation_history.append({
            "user_id": user_id,
            "question": question,
            "timestamp": time.time()
        })
        
        # 保持对话历史在窗口大小内
        if len(self.conversation_history) > self.context_window:
            self.conversation_history = self.conversation_history[-self.context_window:]
        
        # 分析对话上下文
        context_info = self._analyze_conversation_context(user_id)
        
        # 生成个性化回答
        answer = self._generate_contextual_answer(question, context_info)
        
        # 更新用户画像
        self._update_user_profile(user_id, question)
        
        # 添加答案到历史
        self.conversation_history[-1]["answer"] = answer
        
        return QAResult(
            question=question,
            answer=answer,
            confidence=0.7,
            source="conversational",
            method="conversational",
            metadata={
                "context_length": len(self.conversation_history),
                "user_profile": self.user_profile.get(user_id, {}),
                "conversation_topics": context_info.get("topics", [])
            }
        )
    
    def _analyze_conversation_context(self, user_id: str) -> Dict[str, Any]:
        """分析对话上下文"""
        user_questions = [
            entry["question"] for entry in self.conversation_history 
            if entry.get("user_id") == user_id
        ]
        
        # 简单的主题分析
        topics = []
        topic_keywords = {
            "技术": ["AI", "人工智能", "机器学习", "算法", "编程"],
            "生活": ["天气", "食物", "健康", "运动", "娱乐"],
            "工作": ["项目", "会议", "报告", "同事", "老板"],
            "学习": ["课程", "考试", "作业", "学校", "老师"]
        }
        
        for topic, keywords in topic_keywords.items():
            for question in user_questions:
                if any(keyword in question for keyword in keywords):
                    topics.append(topic)
                    break
        
        return {
            "topics": list(set(topics)),
            "question_count": len(user_questions),
            "recent_questions": user_questions[-3:]
        }
    
    def _generate_contextual_answer(self, question: str, context_info: Dict[str, Any]) -> str:
        """生成上下文相关的答案"""
        # 检查是否是延续性问题
        continuation_patterns = ["那么", "然后", "还有", "另外", "此外"]
        is_continuation = any(pattern in question for pattern in continuation_patterns)
        
        if is_continuation and context_info["recent_questions"]:
            prev_question = context_info["recent_questions"][-1]
            return f"基于我们刚才讨论的'{prev_question}'，关于'{question}'我的回答是..."
        
        # 根据用户兴趣主题定制回答
        topics = context_info.get("topics", [])
        if "技术" in topics:
            return f"从技术角度来看，{question}涉及到多个方面的考虑..."
        elif "学习" in topics:
            return f"关于{question}，这是一个很好的学习问题..."
        
        # 默认回答
        return f"关于'{question}'，这是一个有趣的问题。让我为您详细解答..."
    
    def _update_user_profile(self, user_id: str, question: str):
        """更新用户画像"""
        if user_id not in self.user_profile:
            self.user_profile[user_id] = {
                "interests": Counter(),
                "question_types": Counter(),
                "interaction_count": 0
            }
        
        profile = self.user_profile[user_id]
        profile["interaction_count"] += 1
        
        # 更新兴趣
        interests = ["技术", "生活", "工作", "学习", "娱乐"]
        for interest in interests:
            if any(keyword in question for keyword in [interest]):
                profile["interests"][interest] += 1

class QASystem:
    """综合问答系统"""
    
    def __init__(self):
        self.retrieval_qa = RetrievalBasedQA()
        self.generative_qa = GenerativeQA()
        self.faq_matcher = FAQMatcher()
        self.kg_qa = KnowledgeGraphQA()
        self.conversational_qa = ConversationalQA()
        
        self.qa_history = []
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """初始化示例数据"""
        # 添加示例文档
        sample_docs = [
            Document(
                "人工智能简介",
                "人工智能(AI)是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。"
                "机器学习是AI的一个子领域，它使计算机能够通过数据学习和改进。"
                "深度学习是机器学习的一种方法，使用多层神经网络来模拟人脑的学习过程。"
            ),
            Document(
                "自然语言处理",
                "自然语言处理(NLP)是人工智能和语言学的交叉领域。"
                "它专注于使计算机能够理解、解释和生成人类语言。"
                "常见的NLP任务包括文本分类、情感分析、机器翻译和问答系统。"
            ),
            Document(
                "机器学习算法",
                "机器学习包括监督学习、无监督学习和强化学习三种主要类型。"
                "监督学习使用标记数据进行训练，如分类和回归问题。"
                "无监督学习从未标记数据中发现模式，如聚类和降维。"
                "强化学习通过与环境交互来学习最优策略。"
            )
        ]
        
        self.retrieval_qa.add_documents(sample_docs)
        
        # 添加FAQ
        faqs = [
            ("什么是人工智能？", "人工智能是使机器能够模拟人类智能行为的技术。", "AI基础"),
            ("机器学习和深度学习有什么区别？", "深度学习是机器学习的一个子集，使用深层神经网络。", "AI基础"),
            ("如何开始学习AI？", "建议从Python编程和数学基础开始，然后学习机器学习算法。", "学习指导"),
            ("AI在哪些领域有应用？", "AI应用于医疗、金融、交通、教育、娱乐等多个领域。", "应用领域")
        ]
        
        for question, answer, category in faqs:
            self.faq_matcher.add_faq_pair(question, answer, category)
        
        self.faq_matcher.build_index()
        
        # 添加知识图谱数据
        self.kg_qa.add_entity("Python", "编程语言", {
            "定义": "Python是一种高级编程语言",
            "特点": "简洁易读，功能强大",
            "应用": "AI、Web开发、数据分析"
        })
        
        self.kg_qa.add_entity("TensorFlow", "机器学习框架", {
            "定义": "Google开发的开源机器学习框架",
            "用途": "构建和训练神经网络",
            "特点": "灵活且可扩展"
        })
        
        self.kg_qa.add_relation("Python", "支持", "TensorFlow")
        self.kg_qa.add_relation("TensorFlow", "用于", "深度学习")
    
    def answer_question(self, question: str, method: str = "auto", context: str = "", user_id: str = "default") -> QAResult:
        """回答问题"""
        if method == "auto":
            method = self._select_best_method(question)
        
        if method == "retrieval":
            result = self.retrieval_qa.answer_question(question)
        elif method == "generative" and context:
            result = self.generative_qa.answer_question(question, context)
        elif method == "faq":
            result = self.faq_matcher.find_similar_question(question)
        elif method == "knowledge_graph":
            result = self.kg_qa.answer_question(question)
        elif method == "conversational":
            result = self.conversational_qa.answer_question(question, user_id)
        else:
            result = QAResult(question, "不支持的问答方法", 0.0, method=method)
        
        # 保存到历史记录
        self.qa_history.append(result)
        
        return result
    
    def _select_best_method(self, question: str) -> str:
        """自动选择最佳问答方法"""
        question_lower = question.lower()
        
        # 对话延续性检测
        continuation_keywords = ["那么", "然后", "还有", "另外"]
        if any(keyword in question for keyword in continuation_keywords):
            return "conversational"
        
        # FAQ检测
        faq_keywords = ["什么是", "如何", "怎么", "为什么"]
        if any(keyword in question for keyword in faq_keywords):
            return "faq"
        
        # 知识图谱检测
        entities = ["Python", "TensorFlow", "AI", "机器学习"]
        if any(entity.lower() in question_lower for entity in entities):
            return "knowledge_graph"
        
        # 默认使用检索式
        return "retrieval"
    
    def compare_qa_methods(self, question: str, context: str = ""):
        """比较不同问答方法"""
        print(f"\n问题: {question}")
        print("=" * 80)
        
        methods = ["retrieval", "faq", "knowledge_graph", "conversational"]
        if context:
            methods.append("generative")
        
        results = []
        for method in methods:
            try:
                if method == "generative" and not context:
                    continue
                
                result = self.answer_question(question, method, context)
                results.append(result)
                
                print(f"\n{method.upper()} 方法:")
                print(f"回答: {result.answer}")
                print(f"置信度: {result.confidence:.3f}")
                print(f"来源: {result.source}")
                
            except Exception as e:
                print(f"\n{method.upper()} 方法: 失败 - {e}")
        
        return results
    
    def analyze_qa_performance(self):
        """分析问答性能"""
        if not self.qa_history:
            print("没有问答历史记录")
            return
        
        print("\n问答系统性能分析:")
        print("=" * 60)
        
        # 按方法统计
        method_stats = Counter(result.method for result in self.qa_history)
        confidence_by_method = defaultdict(list)
        
        for result in self.qa_history:
            confidence_by_method[result.method].append(result.confidence)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 方法使用分布
        methods = list(method_stats.keys())
        counts = list(method_stats.values())
        
        axes[0, 0].pie(counts, labels=methods, autopct='%1.1f%%')
        axes[0, 0].set_title('问答方法使用分布')
        
        # 各方法平均置信度
        avg_confidence = {method: np.mean(confidences) 
                         for method, confidences in confidence_by_method.items()}
        
        axes[0, 1].bar(avg_confidence.keys(), avg_confidence.values())
        axes[0, 1].set_title('各方法平均置信度')
        axes[0, 1].set_ylabel('置信度')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 置信度分布
        all_confidences = [result.confidence for result in self.qa_history]
        axes[1, 0].hist(all_confidences, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('置信度分布')
        axes[1, 0].set_xlabel('置信度')
        axes[1, 0].set_ylabel('频次')
        
        # 时间趋势（简化）
        axes[1, 1].plot(range(len(self.qa_history)), all_confidences, 'o-')
        axes[1, 1].set_title('置信度时间趋势')
        axes[1, 1].set_xlabel('问答序号')
        axes[1, 1].set_ylabel('置信度')
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print(f"总问答次数: {len(self.qa_history)}")
        print(f"平均置信度: {np.mean(all_confidences):.3f}")
        print(f"最高置信度: {np.max(all_confidences):.3f}")
        print(f"最低置信度: {np.min(all_confidences):.3f}")
        
        print(f"\n各方法统计:")
        for method, count in method_stats.most_common():
            avg_conf = np.mean(confidence_by_method[method])
            print(f"  {method}: {count} 次, 平均置信度 {avg_conf:.3f}")
    
    def run_interactive_demo(self):
        """运行交互式演示"""
        print("\n❓ 问答系统教育应用")
        print("=" * 50)
        print("可用的问答方法:")
        print("  1. retrieval - 基于检索")
        print("  2. generative - 基于生成 (需要上下文)")
        print("  3. faq - FAQ匹配")
        print("  4. knowledge_graph - 知识图谱")
        print("  5. conversational - 多轮对话")
        print("  6. auto - 自动选择")
        
        while True:
            print("\n选择操作:")
            print("1. 单一方法问答")
            print("2. 方法比较")
            print("3. 多轮对话")
            print("4. 性能分析")
            print("5. 添加FAQ")
            print("6. 添加知识实体")
            print("7. 批量测试")
            print("0. 退出")
            
            choice = input("\n请选择 (0-7): ").strip()
            
            if choice == '0':
                break
            
            elif choice == '1':
                question = input("请输入问题: ").strip()
                if question:
                    method = input("选择方法 (retrieval/generative/faq/knowledge_graph/conversational/auto): ").strip()
                    
                    context = ""
                    if method == "generative":
                        context = input("请输入上下文: ").strip()
                    
                    result = self.answer_question(question, method, context)
                    
                    print(f"\n问题: {result.question}")
                    print(f"回答: {result.answer}")
                    print(f"方法: {result.method}")
                    print(f"置信度: {result.confidence:.3f}")
                    print(f"来源: {result.source}")
                    if result.metadata:
                        print(f"元数据: {result.metadata}")
            
            elif choice == '2':
                question = input("请输入问题: ").strip()
                if question:
                    context = input("上下文 (可选): ").strip()
                    self.compare_qa_methods(question, context)
            
            elif choice == '3':
                print("进入多轮对话模式 (输入'退出'结束对话):")
                user_id = input("用户ID (默认为 default): ").strip() or "default"
                
                while True:
                    question = input(f"\n[{user_id}] 问题: ").strip()
                    if question.lower() in ['退出', 'exit', 'quit']:
                        break
                    
                    result = self.answer_question(question, "conversational", user_id=user_id)
                    print(f"回答: {result.answer}")
                    print(f"置信度: {result.confidence:.3f}")
            
            elif choice == '4':
                self.analyze_qa_performance()
            
            elif choice == '5':
                question = input("FAQ问题: ").strip()
                answer = input("FAQ答案: ").strip()
                category = input("类别 (可选): ").strip() or "general"
                
                if question and answer:
                    self.faq_matcher.add_faq_pair(question, answer, category)
                    self.faq_matcher.build_index()
                    print("FAQ添加成功!")
            
            elif choice == '6':
                name = input("实体名称: ").strip()
                entity_type = input("实体类型: ").strip()
                
                if name and entity_type:
                    properties = {}
                    print("输入实体属性 (属性名=属性值，输入空行结束):")
                    while True:
                        prop_input = input().strip()
                        if not prop_input:
                            break
                        if '=' in prop_input:
                            key, value = prop_input.split('=', 1)
                            properties[key.strip()] = value.strip()
                    
                    self.kg_qa.add_entity(name, entity_type, properties)
                    print("实体添加成功!")
            
            elif choice == '7':
                test_questions = [
                    "什么是人工智能？",
                    "如何学习机器学习？",
                    "Python有什么特点？",
                    "TensorFlow是什么？",
                    "AI有哪些应用？"
                ]
                
                print("批量测试问答系统...")
                for question in test_questions:
                    result = self.answer_question(question, "auto")
                    print(f"\n问题: {question}")
                    print(f"回答: {result.answer}")
                    print(f"方法: {result.method}")
                    print(f"置信度: {result.confidence:.3f}")

def main():
    """主函数"""
    print("初始化问答系统...")
    
    qa_system = QASystem()
    qa_system.run_interactive_demo()

if __name__ == "__main__":
    main()
