#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命名实体识别教育应用
Named Entity Recognition (NER) Educational Application

这个应用展示了命名实体识别的各种技术：
- 基于规则的实体识别
- 基于机器学习的实体识别
- 基于深度学习的实体识别
- 自定义实体类型
- 实体关系抽取
- 知识图谱构建

This application demonstrates various NER techniques:
- Rule-based entity recognition
- Machine learning-based entity recognition
- Deep learning-based entity recognition
- Custom entity types
- Entity relationship extraction
- Knowledge graph construction
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import time

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.text_vectorizer import TextVectorizer
from src.utils import load_documents

# 尝试导入深度学习库
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("提示: 安装transformers库以使用深度学习NER: pip install transformers")

# 尝试导入spaCy
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    print("提示: 安装spaCy库以获得更好的NER效果: pip install spacy")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class Entity:
    """命名实体"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    context: str = ""

@dataclass
class Relation:
    """实体关系"""
    subject: Entity
    predicate: str
    object: Entity
    confidence: float = 1.0

class RuleBasedNER:
    """基于规则的命名实体识别"""
    
    def __init__(self):
        # 定义实体规则模式
        self.patterns = {
            'PERSON': [
                r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # 英文人名
                r'[王李张刘陈杨黄赵周吴徐孙朱马胡郭林何高梁郑罗宋谢唐韩曹许邓萧冯曾程蔡彭潘袁于董余苏叶吕魏蒋田杜丁沈姜范江傅钟卢汪戴崔任陆廖姚方金邱夏谭韦贾邹石熊孟秦阎薛侯雷白龙段郝孔邵史毛常万顾赖武康贺严尹钱施牛洪龚][A-Za-z\u4e00-\u9fff]{1,3}',  # 中文人名
            ],
            'LOCATION': [
                r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|State|Province|County|District))?',  # 英文地名
                r'[北京上海天津重庆河北山西辽宁吉林黑龙江江苏浙江安徽福建江西山东河南湖北湖南广东广西海南四川贵州云南西藏陕西甘肃青海宁夏新疆内蒙古香港澳门台湾][市省区县镇村]?',  # 中国地名
                r'[东西南北中][A-Za-z\u4e00-\u9fff]{1,8}[市省区县镇村路街道]',  # 方位地名
            ],
            'ORGANIZATION': [
                r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|Ltd|Co|Company|University|College|School|Hospital|Bank))',  # 英文机构
                r'[A-Za-z\u4e00-\u9fff]{2,10}[公司集团银行医院学校大学学院]',  # 中文机构
                r'[A-Za-z\u4e00-\u9fff]{2,10}[有限责任股份]公司',  # 公司类型
            ],
            'DATE': [
                r'\d{4}年\d{1,2}月\d{1,2}日',  # 中文日期
                r'\d{4}-\d{1,2}-\d{1,2}',  # 数字日期
                r'\d{1,2}/\d{1,2}/\d{4}',  # 美式日期
                r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',  # 英文日期
            ],
            'MONEY': [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # 美元
                r'¥\d+(?:,\d{3})*(?:\.\d{2})?',  # 人民币
                r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:美元|人民币|元|万元|亿元)',  # 中文货币
            ],
            'PHONE': [
                r'\d{3}-\d{3}-\d{4}',  # 美式电话
                r'\d{11}',  # 中国手机号
                r'\(\d{3}\)\s*\d{3}-\d{4}',  # 带括号电话
            ],
            'EMAIL': [
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # 邮箱
            ],
            'URL': [
                r'https?://[^\s]+',  # 网址
                r'www\.[^\s]+',  # www网址
            ]
        }
        
        # 编译正则表达式
        self.compiled_patterns = {}
        for label, patterns in self.patterns.items():
            self.compiled_patterns[label] = [re.compile(pattern) for pattern in patterns]
    
    def extract_entities(self, text: str) -> List[Entity]:
        """提取命名实体"""
        entities = []
        
        for label, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity = Entity(
                        text=match.group(),
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0,
                        context=text[max(0, match.start()-20):match.end()+20]
                    )
                    entities.append(entity)
        
        # 去重和合并重叠实体
        entities = self._merge_overlapping_entities(entities)
        
        return entities
    
    def _merge_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """合并重叠的实体"""
        if not entities:
            return entities
        
        # 按位置排序
        entities.sort(key=lambda x: (x.start, x.end))
        
        merged = [entities[0]]
        
        for current in entities[1:]:
            last = merged[-1]
            
            # 检查是否重叠
            if current.start < last.end:
                # 选择更长的实体或置信度更高的实体
                if (current.end - current.start) > (last.end - last.start):
                    merged[-1] = current
                elif (current.end - current.start) == (last.end - last.start) and current.confidence > last.confidence:
                    merged[-1] = current
            else:
                merged.append(current)
        
        return merged

class TransformerNER:
    """基于Transformer的命名实体识别"""
    
    def __init__(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        if not HAS_TRANSFORMERS:
            raise ImportError("需要安装transformers库")
        
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple"
            )
            self.model_name = model_name
        except Exception as e:
            print(f"无法加载模型 {model_name}，尝试使用默认模型")
            try:
                self.ner_pipeline = pipeline("ner", aggregation_strategy="simple")
                self.model_name = "default"
            except Exception as e2:
                raise ImportError(f"无法初始化NER模型: {e2}")
    
    def extract_entities(self, text: str) -> List[Entity]:
        """提取命名实体"""
        try:
            results = self.ner_pipeline(text)
            entities = []
            
            for result in results:
                # 标准化标签
                label = result['entity_group'].replace('B-', '').replace('I-', '')
                
                entity = Entity(
                    text=result['word'],
                    label=label,
                    start=result.get('start', 0),
                    end=result.get('end', len(result['word'])),
                    confidence=result['score'],
                    context=text
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            print(f"Transformer NER失败: {e}")
            return []

class SpacyNER:
    """基于spaCy的命名实体识别"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        if not HAS_SPACY:
            raise ImportError("需要安装spaCy库")
        
        try:
            self.nlp = spacy.load(model_name)
            self.model_name = model_name
        except OSError:
            print(f"无法加载spaCy模型 {model_name}")
            print("请先下载模型: python -m spacy download en_core_web_sm")
            raise
    
    def extract_entities(self, text: str) -> List[Entity]:
        """提取命名实体"""
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0,  # spaCy不提供置信度
                    context=text[max(0, ent.start_char-20):ent.end_char+20]
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            print(f"spaCy NER失败: {e}")
            return []

class CustomNER:
    """自定义命名实体识别"""
    
    def __init__(self):
        self.custom_entities = {
            'PRODUCT': [
                'iPhone', 'iPad', 'MacBook', 'Android', 'Windows',
                'Tesla Model', 'BMW', 'Mercedes', 'Toyota',
                'PlayStation', 'Xbox', 'Nintendo'
            ],
            'TECHNOLOGY': [
                '人工智能', '机器学习', '深度学习', '自然语言处理',
                '计算机视觉', '数据科学', '区块链', '云计算',
                'Python', 'JavaScript', 'Java', 'C++', 'TensorFlow',
                'PyTorch', 'React', 'Vue', 'Angular'
            ],
            'DISEASE': [
                '新冠肺炎', 'COVID-19', '糖尿病', '高血压', '癌症',
                '心脏病', '肺炎', '流感', '感冒', '发烧'
            ]
        }
    
    def add_entity_type(self, entity_type: str, entities: List[str]):
        """添加自定义实体类型"""
        if entity_type not in self.custom_entities:
            self.custom_entities[entity_type] = []
        self.custom_entities[entity_type].extend(entities)
    
    def extract_entities(self, text: str) -> List[Entity]:
        """提取自定义实体"""
        entities = []
        text_lower = text.lower()
        
        for entity_type, entity_list in self.custom_entities.items():
            for entity_text in entity_list:
                # 查找实体在文本中的所有出现位置
                start = 0
                while True:
                    pos = text_lower.find(entity_text.lower(), start)
                    if pos == -1:
                        break
                    
                    entity = Entity(
                        text=text[pos:pos+len(entity_text)],
                        label=entity_type,
                        start=pos,
                        end=pos + len(entity_text),
                        confidence=1.0,
                        context=text[max(0, pos-20):pos+len(entity_text)+20]
                    )
                    entities.append(entity)
                    start = pos + 1
        
        return entities

class RelationExtractor:
    """关系抽取器"""
    
    def __init__(self):
        # 定义关系模式
        self.relation_patterns = [
            (r'(.+)\s*是\s*(.+)的.*', 'is_part_of'),
            (r'(.+)\s*在\s*(.+)', 'located_in'),
            (r'(.+)\s*工作在\s*(.+)', 'works_at'),
            (r'(.+)\s*属于\s*(.+)', 'belongs_to'),
            (r'(.+)\s*创立了\s*(.+)', 'founded'),
            (r'(.+)\s*生产\s*(.+)', 'produces'),
            (r'(.+)\s*与\s*(.+)\s*合作', 'cooperates_with'),
        ]
        
        self.compiled_patterns = [(re.compile(pattern), relation) 
                                for pattern, relation in self.relation_patterns]
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """抽取实体关系"""
        relations = []
        
        # 基于模式的关系抽取
        for pattern, relation_type in self.compiled_patterns:
            for match in pattern.finditer(text):
                subject_text = match.group(1).strip()
                object_text = match.group(2).strip()
                
                # 查找匹配的实体
                subject_entity = self._find_entity_by_text(entities, subject_text)
                object_entity = self._find_entity_by_text(entities, object_text)
                
                if subject_entity and object_entity:
                    relation = Relation(
                        subject=subject_entity,
                        predicate=relation_type,
                        object=object_entity,
                        confidence=0.8
                    )
                    relations.append(relation)
        
        return relations
    
    def _find_entity_by_text(self, entities: List[Entity], text: str) -> Optional[Entity]:
        """根据文本查找实体"""
        for entity in entities:
            if text in entity.text or entity.text in text:
                return entity
        return None

class KnowledgeGraph:
    """知识图谱"""
    
    def __init__(self):
        self.entities = {}  # entity_id -> Entity
        self.relations = []  # List[Relation]
        self.entity_counter = 0
    
    def add_entity(self, entity: Entity) -> str:
        """添加实体"""
        entity_id = f"{entity.label}_{self.entity_counter}"
        self.entities[entity_id] = entity
        self.entity_counter += 1
        return entity_id
    
    def add_relation(self, relation: Relation):
        """添加关系"""
        self.relations.append(relation)
    
    def build_from_entities_and_relations(self, entities: List[Entity], relations: List[Relation]):
        """从实体和关系构建知识图谱"""
        # 添加实体
        entity_map = {}
        for entity in entities:
            entity_id = self.add_entity(entity)
            entity_map[entity.text] = entity_id
        
        # 添加关系
        for relation in relations:
            self.add_relation(relation)
    
    def visualize(self, max_nodes: int = 50):
        """可视化知识图谱"""
        if not self.entities:
            print("知识图谱为空")
            return
        
        G = nx.Graph()
        
        # 添加节点
        for entity_id, entity in list(self.entities.items())[:max_nodes]:
            G.add_node(entity_id, 
                      label=entity.text, 
                      entity_type=entity.label)
        
        # 添加边
        for relation in self.relations:
            subject_id = None
            object_id = None
            
            # 查找实体ID
            for entity_id, entity in self.entities.items():
                if entity.text == relation.subject.text:
                    subject_id = entity_id
                if entity.text == relation.object.text:
                    object_id = entity_id
            
            if subject_id and object_id and subject_id in G.nodes and object_id in G.nodes:
                G.add_edge(subject_id, object_id, 
                          relation=relation.predicate,
                          weight=relation.confidence)
        
        # 绘制图谱
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 根据实体类型设置颜色
        entity_types = set(self.entities[node]['entity_type'] for node in G.nodes())
        colors = plt.cm.Set3(np.linspace(0, 1, len(entity_types)))
        color_map = dict(zip(entity_types, colors))
        
        node_colors = [color_map[self.entities[node]['entity_type']] for node in G.nodes()]
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=500, alpha=0.8)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)
        
        # 绘制标签
        labels = {node: self.entities[node]['label'][:10] + '...' 
                 if len(self.entities[node]['label']) > 10 
                 else self.entities[node]['label'] 
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color_map[entity_type], 
                                    markersize=10, label=entity_type)
                         for entity_type in entity_types]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title('知识图谱可视化')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识图谱统计信息"""
        entity_type_counts = Counter(entity.label for entity in self.entities.values())
        relation_type_counts = Counter(relation.predicate for relation in self.relations)
        
        return {
            'total_entities': len(self.entities),
            'total_relations': len(self.relations),
            'entity_types': dict(entity_type_counts),
            'relation_types': dict(relation_type_counts)
        }

class NERApp:
    """命名实体识别教育应用"""
    
    def __init__(self):
        self.extractors = {}
        self.relation_extractor = RelationExtractor()
        self.knowledge_graph = KnowledgeGraph()
        
        # 初始化实体识别器
        self.extractors['rule_based'] = RuleBasedNER()
        self.extractors['custom'] = CustomNER()
        
        if HAS_TRANSFORMERS:
            try:
                self.extractors['transformer'] = TransformerNER()
            except Exception as e:
                print(f"无法初始化Transformer NER: {e}")
        
        if HAS_SPACY:
            try:
                self.extractors['spacy'] = SpacyNER()
            except Exception as e:
                print(f"无法初始化spaCy NER: {e}")
    
    def extract_entities(self, text: str, methods: List[str] = None) -> Dict[str, List[Entity]]:
        """使用多种方法提取实体"""
        if methods is None:
            methods = list(self.extractors.keys())
        
        results = {}
        for method in methods:
            if method in self.extractors:
                try:
                    entities = self.extractors[method].extract_entities(text)
                    results[method] = entities
                except Exception as e:
                    print(f"方法 {method} 提取实体失败: {e}")
                    results[method] = []
        
        return results
    
    def compare_ner_methods(self, text: str):
        """比较不同NER方法"""
        results = self.extract_entities(text)
        
        print(f"\n文本: {text}")
        print("=" * 80)
        
        for method, entities in results.items():
            print(f"\n{method.upper()} 方法识别出 {len(entities)} 个实体:")
            print("-" * 50)
            
            if entities:
                for entity in entities:
                    print(f"  {entity.text:<15} | {entity.label:<12} | 置信度: {entity.confidence:.3f}")
            else:
                print("  未识别出任何实体")
    
    def analyze_entity_distribution(self, texts: List[str], method: str = 'rule_based'):
        """分析实体分布"""
        if method not in self.extractors:
            print(f"不支持的方法: {method}")
            return
        
        all_entities = []
        entity_type_counts = Counter()
        entity_text_counts = Counter()
        
        for text in texts:
            entities = self.extractors[method].extract_entities(text)
            all_entities.extend(entities)
            
            for entity in entities:
                entity_type_counts[entity.label] += 1
                entity_text_counts[entity.text.lower()] += 1
        
        # 可视化实体类型分布
        plt.figure(figsize=(15, 5))
        
        # 实体类型分布
        plt.subplot(1, 3, 1)
        if entity_type_counts:
            labels, counts = zip(*entity_type_counts.most_common(10))
            plt.pie(counts, labels=labels, autopct='%1.1f%%')
            plt.title('实体类型分布')
        
        # 实体频次分布
        plt.subplot(1, 3, 2)
        if entity_text_counts:
            top_entities = entity_text_counts.most_common(10)
            entities, counts = zip(*top_entities)
            plt.barh(range(len(entities)), counts)
            plt.yticks(range(len(entities)), entities)
            plt.xlabel('频次')
            plt.title('高频实体')
        
        # 置信度分布
        plt.subplot(1, 3, 3)
        confidences = [entity.confidence for entity in all_entities]
        if confidences:
            plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('置信度')
            plt.ylabel('频次')
            plt.title('置信度分布')
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print(f"\n实体分析统计 ({method}):")
        print(f"总实体数: {len(all_entities)}")
        print(f"唯一实体数: {len(entity_text_counts)}")
        print(f"实体类型数: {len(entity_type_counts)}")
        
        print(f"\n最常见实体类型:")
        for entity_type, count in entity_type_counts.most_common(5):
            print(f"  {entity_type}: {count}")
    
    def build_knowledge_graph_from_text(self, text: str, method: str = 'rule_based'):
        """从文本构建知识图谱"""
        # 提取实体
        entities = self.extractors[method].extract_entities(text)
        
        # 提取关系
        relations = self.relation_extractor.extract_relations(text, entities)
        
        # 构建知识图谱
        self.knowledge_graph = KnowledgeGraph()
        self.knowledge_graph.build_from_entities_and_relations(entities, relations)
        
        # 显示统计信息
        stats = self.knowledge_graph.get_statistics()
        print(f"\n知识图谱统计:")
        print(f"实体数: {stats['total_entities']}")
        print(f"关系数: {stats['total_relations']}")
        
        if stats['entity_types']:
            print(f"实体类型分布:")
            for entity_type, count in stats['entity_types'].items():
                print(f"  {entity_type}: {count}")
        
        if stats['relation_types']:
            print(f"关系类型分布:")
            for relation_type, count in stats['relation_types'].items():
                print(f"  {relation_type}: {count}")
        
        # 可视化知识图谱
        if stats['total_entities'] > 0:
            self.knowledge_graph.visualize()
    
    def evaluate_ner_performance(self, text: str, ground_truth: List[Entity], method: str = 'rule_based'):
        """评估NER性能"""
        predicted_entities = self.extractors[method].extract_entities(text)
        
        # 计算精确率、召回率、F1分数
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # 简单的匹配策略：文本和标签都匹配
        predicted_set = {(e.text.lower(), e.label) for e in predicted_entities}
        ground_truth_set = {(e.text.lower(), e.label) for e in ground_truth}
        
        true_positives = len(predicted_set & ground_truth_set)
        false_positives = len(predicted_set - ground_truth_set)
        false_negatives = len(ground_truth_set - predicted_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nNER性能评估 ({method}):")
        print(f"精确率: {precision:.3f}")
        print(f"召回率: {recall:.3f}")
        print(f"F1分数: {f1:.3f}")
        print(f"正确识别: {true_positives}")
        print(f"错误识别: {false_positives}")
        print(f"遗漏识别: {false_negatives}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def run_interactive_demo(self):
        """运行交互式演示"""
        print("\n🏷️ 命名实体识别教育应用")
        print("=" * 50)
        print("可用的NER方法:")
        for i, method in enumerate(self.extractors.keys(), 1):
            print(f"  {i}. {method}")
        
        while True:
            print("\n选择操作:")
            print("1. 单文本实体识别")
            print("2. 方法比较")
            print("3. 实体分布分析")
            print("4. 构建知识图谱")
            print("5. 添加自定义实体")
            print("6. 加载示例文档")
            print("0. 退出")
            
            choice = input("\n请选择 (0-6): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                text = input("请输入要分析的文本: ").strip()
                if text:
                    method = input(f"选择方法 ({'/'.join(self.extractors.keys())}): ").strip()
                    if method in self.extractors:
                        entities = self.extractors[method].extract_entities(text)
                        print(f"\n识别出 {len(entities)} 个实体:")
                        for entity in entities:
                            print(f"  {entity.text} [{entity.label}] (置信度: {entity.confidence:.3f})")
                    else:
                        print("无效的方法")
            
            elif choice == '2':
                text = input("请输入要分析的文本: ").strip()
                if text:
                    self.compare_ner_methods(text)
            
            elif choice == '3':
                print("请输入多个文本，每行一个 (输入空行结束):")
                texts = []
                while True:
                    line = input().strip()
                    if not line:
                        break
                    texts.append(line)
                
                if texts:
                    method = input(f"选择方法 ({'/'.join(self.extractors.keys())}): ").strip()
                    if method in self.extractors:
                        self.analyze_entity_distribution(texts, method)
                    else:
                        print("无效的方法")
            
            elif choice == '4':
                text = input("请输入要构建知识图谱的文本: ").strip()
                if text:
                    method = input(f"选择方法 ({'/'.join(self.extractors.keys())}): ").strip()
                    if method in self.extractors:
                        self.build_knowledge_graph_from_text(text, method)
                    else:
                        print("无效的方法")
            
            elif choice == '5':
                entity_type = input("请输入实体类型: ").strip().upper()
                print("请输入实体列表，每行一个 (输入空行结束):")
                entities = []
                while True:
                    line = input().strip()
                    if not line:
                        break
                    entities.append(line)
                
                if entity_type and entities:
                    self.extractors['custom'].add_entity_type(entity_type, entities)
                    print(f"已添加 {len(entities)} 个 {entity_type} 类型的实体")
            
            elif choice == '6':
                try:
                    documents = load_documents('data/sample_documents.json')
                    texts = [doc['content'] for doc in documents[:5]]
                    print(f"加载了 {len(texts)} 个文档")
                    
                    method = input(f"选择分析方法 ({'/'.join(self.extractors.keys())}): ").strip()
                    if method in self.extractors:
                        print("正在分析...")
                        self.analyze_entity_distribution(texts, method)
                        
                        # 构建知识图谱
                        print("正在构建知识图谱...")
                        combined_text = ' '.join(texts)
                        self.build_knowledge_graph_from_text(combined_text, method)
                    else:
                        print("无效的方法")
                except Exception as e:
                    print(f"加载文档失败: {e}")

def main():
    """主函数"""
    print("初始化命名实体识别应用...")
    
    app = NERApp()
    
    # 示例文本
    sample_text = """
    苹果公司（Apple Inc.）是一家总部位于加利福尼亚州库比蒂诺的美国跨国科技公司。
    该公司由史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩于1976年4月1日创立。
    苹果公司设计、开发和销售消费电子产品、计算机软件和在线服务。
    公司的硬件产品包括iPhone智能手机、iPad平板电脑、Mac个人电脑、iPod便携式媒体播放器、
    Apple Watch智能手表、Apple TV数字媒体播放器和HomePod智能音箱。
    
    苹果公司的CEO蒂姆·库克在2023年宣布公司将投资10亿美元用于人工智能研发。
    公司总部位于苹果园区(Apple Park)，该园区于2017年开放，耗资约50亿美元建设。
    你可以通过info@apple.com联系苹果公司，或访问www.apple.com了解更多信息。
    """
    
    print("\n🎯 演示: 多方法实体识别")
    print("=" * 40)
    
    # 比较不同NER方法
    app.compare_ner_methods(sample_text)
    
    # 构建知识图谱
    print("\n构建知识图谱...")
    app.build_knowledge_graph_from_text(sample_text)
    
    # 运行交互式演示
    app.run_interactive_demo()

if __name__ == "__main__":
    main()
