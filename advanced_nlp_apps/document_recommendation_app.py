#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档推荐系统教育应用
Document Recommendation Educational Application

这个应用展示了文档推荐系统的各种技术：
- 基于内容的推荐 (Content-based Filtering)
- 协同过滤推荐 (Collaborative Filtering)  
- 混合推荐系统 (Hybrid Recommendation)
- 基于知识的推荐 (Knowledge-based)
- 实时推荐系统
- 推荐效果评估

This application demonstrates various document recommendation techniques:
- Content-based filtering
- Collaborative filtering
- Hybrid recommendation systems
- Knowledge-based recommendation
- Real-time recommendation
- Recommendation evaluation
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
import random
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.text_vectorizer import TextVectorizer
from src.utils import load_documents

# 尝试导入机器学习库
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    from sklearn.decomposition import TruncatedSVD, NMF
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("提示: 安装scikit-learn库以使用机器学习功能: pip install scikit-learn")

# 尝试导入scipy
try:
    from scipy.sparse import csr_matrix
    from scipy.spatial.distance import cosine
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("提示: 安装scipy库以获得更好的矩阵运算: pip install scipy")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class Document:
    """文档信息"""
    id: str
    title: str
    content: str
    category: str
    tags: List[str]
    author: str = ""
    publish_date: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.publish_date is None:
            self.publish_date = datetime.now()

@dataclass
class User:
    """用户信息"""
    id: str
    preferences: Dict[str, float]
    viewed_documents: List[str]
    ratings: Dict[str, float]
    profile: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.profile is None:
            self.profile = {}

@dataclass
class Recommendation:
    """推荐结果"""
    document_id: str
    score: float
    reason: str
    method: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ContentBasedRecommender:
    """基于内容的推荐系统"""
    
    def __init__(self):
        self.documents = {}
        self.vectorizer = None
        self.doc_vectors = None
        self.is_trained = False
    
    def add_documents(self, documents: List[Document]):
        """添加文档"""
        for doc in documents:
            self.documents[doc.id] = doc
        self._build_content_model()
    
    def _build_content_model(self):
        """构建内容模型"""
        if not self.documents or not HAS_SKLEARN:
            return
        
        print("构建内容特征向量...")
        
        # 准备文档文本
        doc_texts = []
        doc_ids = []
        
        for doc_id, doc in self.documents.items():
            # 合并标题、内容、标签和分类
            combined_text = f"{doc.title} {doc.content} {' '.join(doc.tags)} {doc.category}"
            doc_texts.append(combined_text)
            doc_ids.append(doc_id)
        
        # 构建TF-IDF向量
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        self.doc_vectors = self.vectorizer.fit_transform(doc_texts)
        self.doc_ids = doc_ids
        self.is_trained = True
        
        print(f"内容模型构建完成，包含 {len(self.documents)} 个文档")
    
    def recommend(self, user: User, top_k: int = 10) -> List[Recommendation]:
        """基于内容推荐"""
        if not self.is_trained:
            return []
        
        # 构建用户兴趣向量
        user_vector = self._build_user_profile_vector(user)
        
        if user_vector is None:
            return self._recommend_popular_documents(top_k)
        
        # 计算文档相似度
        similarities = cosine_similarity(user_vector, self.doc_vectors).flatten()
        
        # 排除用户已查看的文档
        viewed_indices = []
        for i, doc_id in enumerate(self.doc_ids):
            if doc_id in user.viewed_documents:
                viewed_indices.append(i)
        
        # 获取推荐
        recommendations = []
        sorted_indices = np.argsort(similarities)[::-1]
        
        count = 0
        for idx in sorted_indices:
            if idx not in viewed_indices and count < top_k:
                doc_id = self.doc_ids[idx]
                doc = self.documents[doc_id]
                
                recommendations.append(Recommendation(
                    document_id=doc_id,
                    score=float(similarities[idx]),
                    reason=f"基于您对{doc.category}类文档的兴趣",
                    method="content_based",
                    metadata={
                        "similarity": float(similarities[idx]),
                        "category": doc.category,
                        "tags": doc.tags
                    }
                ))
                count += 1
        
        return recommendations
    
    def _build_user_profile_vector(self, user: User) -> Optional[np.ndarray]:
        """构建用户兴趣向量"""
        if not user.viewed_documents:
            return None
        
        # 获取用户查看过的文档
        user_docs = []
        weights = []
        
        for doc_id in user.viewed_documents:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                combined_text = f"{doc.title} {doc.content} {' '.join(doc.tags)} {doc.category}"
                user_docs.append(combined_text)
                
                # 使用评分作为权重，如果没有评分则使用1.0
                weight = user.ratings.get(doc_id, 1.0)
                weights.append(weight)
        
        if not user_docs:
            return None
        
        # 向量化用户文档
        user_doc_vectors = self.vectorizer.transform(user_docs)
        
        # 加权平均
        weights = np.array(weights).reshape(-1, 1)
        user_vector = np.average(user_doc_vectors.toarray(), axis=0, weights=weights.flatten())
        
        return user_vector.reshape(1, -1)
    
    def _recommend_popular_documents(self, top_k: int) -> List[Recommendation]:
        """推荐热门文档（冷启动）"""
        doc_list = list(self.documents.values())
        random.shuffle(doc_list)
        
        recommendations = []
        for i, doc in enumerate(doc_list[:top_k]):
            recommendations.append(Recommendation(
                document_id=doc.id,
                score=0.5,  # 默认分数
                reason="热门推荐",
                method="content_based_popular",
                metadata={"rank": i + 1}
            ))
        
        return recommendations

class CollaborativeFilteringRecommender:
    """协同过滤推荐系统"""
    
    def __init__(self, method: str = "user_based"):
        self.method = method  # "user_based" or "item_based"
        self.users = {}
        self.documents = {}
        self.rating_matrix = None
        self.similarity_matrix = None
        self.is_trained = False
    
    def add_users(self, users: List[User]):
        """添加用户"""
        for user in users:
            self.users[user.id] = user
    
    def add_documents(self, documents: List[Document]):
        """添加文档"""
        for doc in documents:
            self.documents[doc.id] = doc
    
    def build_rating_matrix(self):
        """构建评分矩阵"""
        if not self.users or not HAS_SKLEARN:
            return
        
        print(f"构建{self.method}协同过滤模型...")
        
        # 收集所有用户和文档ID
        user_ids = list(self.users.keys())
        doc_ids = list(self.documents.keys())
        
        # 创建评分矩阵
        rating_data = []
        for i, user_id in enumerate(user_ids):
            user = self.users[user_id]
            row = []
            for j, doc_id in enumerate(doc_ids):
                rating = user.ratings.get(doc_id, 0.0)
                row.append(rating)
            rating_data.append(row)
        
        self.rating_matrix = np.array(rating_data)
        self.user_ids = user_ids
        self.doc_ids = doc_ids
        
        # 计算相似度矩阵
        self._compute_similarity_matrix()
        self.is_trained = True
        
        print(f"协同过滤模型构建完成，{len(user_ids)}个用户，{len(doc_ids)}个文档")
    
    def _compute_similarity_matrix(self):
        """计算相似度矩阵"""
        if self.method == "user_based":
            # 用户相似度
            self.similarity_matrix = cosine_similarity(self.rating_matrix)
        else:
            # 物品相似度
            self.similarity_matrix = cosine_similarity(self.rating_matrix.T)
    
    def recommend(self, user: User, top_k: int = 10) -> List[Recommendation]:
        """协同过滤推荐"""
        if not self.is_trained or user.id not in self.users:
            return []
        
        user_index = self.user_ids.index(user.id)
        
        if self.method == "user_based":
            return self._user_based_recommend(user_index, user, top_k)
        else:
            return self._item_based_recommend(user_index, user, top_k)
    
    def _user_based_recommend(self, user_index: int, user: User, top_k: int) -> List[Recommendation]:
        """基于用户的协同过滤"""
        # 找到相似用户
        user_similarities = self.similarity_matrix[user_index]
        similar_users = np.argsort(user_similarities)[::-1][1:6]  # 排除自己，取前5个相似用户
        
        # 计算推荐分数
        recommendations = []
        user_ratings = self.rating_matrix[user_index]
        
        for doc_idx, doc_id in enumerate(self.doc_ids):
            if user_ratings[doc_idx] > 0:  # 跳过已评分的文档
                continue
            
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_user_idx in similar_users:
                similarity = user_similarities[similar_user_idx]
                rating = self.rating_matrix[similar_user_idx, doc_idx]
                
                if rating > 0 and similarity > 0:
                    weighted_sum += similarity * rating
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                
                recommendations.append(Recommendation(
                    document_id=doc_id,
                    score=predicted_rating,
                    reason="基于相似用户的推荐",
                    method="collaborative_user_based",
                    metadata={
                        "predicted_rating": predicted_rating,
                        "similar_users_count": len([s for s in similar_users if user_similarities[s] > 0])
                    }
                ))
        
        # 排序并返回top_k
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:top_k]
    
    def _item_based_recommend(self, user_index: int, user: User, top_k: int) -> List[Recommendation]:
        """基于物品的协同过滤"""
        user_ratings = self.rating_matrix[user_index]
        recommendations = []
        
        for doc_idx, doc_id in enumerate(self.doc_ids):
            if user_ratings[doc_idx] > 0:  # 跳过已评分的文档
                continue
            
            # 找到与当前文档相似的已评分文档
            doc_similarities = self.similarity_matrix[doc_idx]
            
            weighted_sum = 0
            similarity_sum = 0
            
            for rated_doc_idx in range(len(self.doc_ids)):
                if user_ratings[rated_doc_idx] > 0:
                    similarity = doc_similarities[rated_doc_idx]
                    if similarity > 0:
                        weighted_sum += similarity * user_ratings[rated_doc_idx]
                        similarity_sum += similarity
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                
                recommendations.append(Recommendation(
                    document_id=doc_id,
                    score=predicted_rating,
                    reason="基于相似文档的推荐",
                    method="collaborative_item_based",
                    metadata={
                        "predicted_rating": predicted_rating,
                        "similar_items_count": len([s for s in doc_similarities if s > 0])
                    }
                ))
        
        # 排序并返回top_k
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:top_k]

class HybridRecommender:
    """混合推荐系统"""
    
    def __init__(self):
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeFilteringRecommender()
        self.weights = {"content": 0.6, "collaborative": 0.4}
    
    def set_weights(self, content_weight: float, collaborative_weight: float):
        """设置权重"""
        total = content_weight + collaborative_weight
        self.weights = {
            "content": content_weight / total,
            "collaborative": collaborative_weight / total
        }
    
    def add_documents(self, documents: List[Document]):
        """添加文档"""
        self.content_recommender.add_documents(documents)
        self.collaborative_recommender.add_documents(documents)
    
    def add_users(self, users: List[User]):
        """添加用户"""
        self.collaborative_recommender.add_users(users)
    
    def train(self):
        """训练模型"""
        self.collaborative_recommender.build_rating_matrix()
    
    def recommend(self, user: User, top_k: int = 10) -> List[Recommendation]:
        """混合推荐"""
        # 获取各个方法的推荐
        content_recs = self.content_recommender.recommend(user, top_k * 2)
        collaborative_recs = self.collaborative_recommender.recommend(user, top_k * 2)
        
        # 合并推荐结果
        combined_scores = defaultdict(float)
        all_recommendations = {}
        
        # 处理内容推荐
        for rec in content_recs:
            combined_scores[rec.document_id] += rec.score * self.weights["content"]
            all_recommendations[rec.document_id] = rec
        
        # 处理协同过滤推荐
        for rec in collaborative_recs:
            combined_scores[rec.document_id] += rec.score * self.weights["collaborative"]
            
            if rec.document_id in all_recommendations:
                # 更新已存在的推荐
                existing_rec = all_recommendations[rec.document_id]
                existing_rec.reason += f" + {rec.reason}"
                existing_rec.method = "hybrid"
            else:
                rec.method = "hybrid"
                all_recommendations[rec.document_id] = rec
        
        # 更新分数并排序
        hybrid_recommendations = []
        for doc_id, final_score in combined_scores.items():
            rec = all_recommendations[doc_id]
            rec.score = final_score
            rec.metadata["hybrid_score"] = final_score
            rec.metadata["content_weight"] = self.weights["content"]
            rec.metadata["collaborative_weight"] = self.weights["collaborative"]
            hybrid_recommendations.append(rec)
        
        hybrid_recommendations.sort(key=lambda x: x.score, reverse=True)
        return hybrid_recommendations[:top_k]

class KnowledgeBasedRecommender:
    """基于知识的推荐系统"""
    
    def __init__(self):
        self.documents = {}
        self.rules = []
        self.category_preferences = {}
        self.tag_preferences = {}
    
    def add_documents(self, documents: List[Document]):
        """添加文档"""
        for doc in documents:
            self.documents[doc.id] = doc
    
    def add_rule(self, condition: Dict[str, Any], action: str, weight: float = 1.0):
        """添加推荐规则"""
        self.rules.append({
            "condition": condition,
            "action": action,
            "weight": weight
        })
    
    def set_category_preferences(self, preferences: Dict[str, float]):
        """设置类别偏好"""
        self.category_preferences = preferences
    
    def set_tag_preferences(self, preferences: Dict[str, float]):
        """设置标签偏好"""
        self.tag_preferences = preferences
    
    def recommend(self, user: User, top_k: int = 10) -> List[Recommendation]:
        """基于知识推荐"""
        recommendations = []
        
        for doc_id, doc in self.documents.items():
            if doc_id in user.viewed_documents:
                continue
            
            score = self._calculate_knowledge_score(user, doc)
            reason = self._generate_reason(user, doc)
            
            if score > 0:
                recommendations.append(Recommendation(
                    document_id=doc_id,
                    score=score,
                    reason=reason,
                    method="knowledge_based",
                    metadata={
                        "category_score": self.category_preferences.get(doc.category, 0),
                        "tag_scores": {tag: self.tag_preferences.get(tag, 0) for tag in doc.tags}
                    }
                ))
        
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:top_k]
    
    def _calculate_knowledge_score(self, user: User, doc: Document) -> float:
        """计算知识基础分数"""
        score = 0.0
        
        # 类别偏好分数
        category_score = self.category_preferences.get(doc.category, 0.0)
        score += category_score * 0.5
        
        # 标签偏好分数
        tag_scores = [self.tag_preferences.get(tag, 0.0) for tag in doc.tags]
        if tag_scores:
            score += np.mean(tag_scores) * 0.3
        
        # 应用规则
        for rule in self.rules:
            if self._evaluate_rule(user, doc, rule["condition"]):
                score += rule["weight"] * 0.2
        
        # 时效性分数
        if doc.publish_date:
            days_ago = (datetime.now() - doc.publish_date).days
            freshness_score = max(0, 1 - days_ago / 365)  # 一年内的文档有时效性加分
            score += freshness_score * 0.1
        
        return score
    
    def _evaluate_rule(self, user: User, doc: Document, condition: Dict[str, Any]) -> bool:
        """评估规则条件"""
        for key, value in condition.items():
            if key == "user_profile":
                for profile_key, profile_value in value.items():
                    if user.profile.get(profile_key) != profile_value:
                        return False
            elif key == "document_category":
                if doc.category != value:
                    return False
            elif key == "document_tags":
                if not any(tag in doc.tags for tag in value):
                    return False
        
        return True
    
    def _generate_reason(self, user: User, doc: Document) -> str:
        """生成推荐理由"""
        reasons = []
        
        if doc.category in self.category_preferences:
            reasons.append(f"您偏好{doc.category}类文档")
        
        matching_tags = [tag for tag in doc.tags if tag in self.tag_preferences]
        if matching_tags:
            reasons.append(f"包含您感兴趣的标签: {', '.join(matching_tags)}")
        
        if doc.publish_date and (datetime.now() - doc.publish_date).days < 30:
            reasons.append("最新发布")
        
        return "; ".join(reasons) if reasons else "基于知识规则推荐"

class RecommendationEvaluator:
    """推荐系统评估器"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_recommendations(self, 
                                recommendations: List[Recommendation], 
                                ground_truth: Dict[str, float],
                                k: int = 10) -> Dict[str, float]:
        """评估推荐结果"""
        metrics = {}
        
        # 准备数据
        rec_items = [rec.document_id for rec in recommendations[:k]]
        rec_scores = [rec.score for rec in recommendations[:k]]
        
        # Precision@K
        relevant_items = set(doc_id for doc_id, rating in ground_truth.items() if rating >= 3.0)
        recommended_relevant = set(rec_items) & relevant_items
        
        metrics['precision_at_k'] = len(recommended_relevant) / len(rec_items) if rec_items else 0
        
        # Recall@K
        metrics['recall_at_k'] = len(recommended_relevant) / len(relevant_items) if relevant_items else 0
        
        # F1@K
        if metrics['precision_at_k'] + metrics['recall_at_k'] > 0:
            metrics['f1_at_k'] = 2 * metrics['precision_at_k'] * metrics['recall_at_k'] / \
                                 (metrics['precision_at_k'] + metrics['recall_at_k'])
        else:
            metrics['f1_at_k'] = 0
        
        # NDCG@K
        metrics['ndcg_at_k'] = self._calculate_ndcg(rec_items, ground_truth, k)
        
        # 多样性指标
        metrics['diversity'] = self._calculate_diversity(recommendations[:k])
        
        # 覆盖率
        metrics['coverage'] = len(set(rec_items)) / len(rec_items) if rec_items else 0
        
        return metrics
    
    def _calculate_ndcg(self, rec_items: List[str], ground_truth: Dict[str, float], k: int) -> float:
        """计算NDCG@K"""
        # DCG
        dcg = 0
        for i, item in enumerate(rec_items[:k]):
            rating = ground_truth.get(item, 0)
            if i == 0:
                dcg += rating
            else:
                dcg += rating / np.log2(i + 1)
        
        # IDCG
        ideal_ratings = sorted(ground_truth.values(), reverse=True)[:k]
        idcg = 0
        for i, rating in enumerate(ideal_ratings):
            if i == 0:
                idcg += rating
            else:
                idcg += rating / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0
    
    def _calculate_diversity(self, recommendations: List[Recommendation]) -> float:
        """计算推荐多样性"""
        categories = [rec.metadata.get("category", "") for rec in recommendations if rec.metadata]
        unique_categories = len(set(categories))
        total_items = len(categories)
        
        return unique_categories / total_items if total_items > 0 else 0

class DocumentRecommendationApp:
    """文档推荐系统教育应用"""
    
    def __init__(self):
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeFilteringRecommender()
        self.hybrid_recommender = HybridRecommender()
        self.knowledge_recommender = KnowledgeBasedRecommender()
        self.evaluator = RecommendationEvaluator()
        
        self.users = {}
        self.documents = {}
        self.recommendation_history = []
        
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """初始化示例数据"""
        # 创建示例文档
        sample_docs = [
            Document("doc1", "机器学习入门", "机器学习是人工智能的核心技术...", "技术", ["机器学习", "AI", "入门"]),
            Document("doc2", "深度学习实战", "深度学习在图像识别中的应用...", "技术", ["深度学习", "实战", "图像"]),
            Document("doc3", "数据科学概论", "数据科学是一个跨学科领域...", "技术", ["数据科学", "统计", "分析"]),
            Document("doc4", "Python编程指南", "Python是一种简洁优雅的编程语言...", "编程", ["Python", "编程", "指南"]),
            Document("doc5", "Web开发技术", "现代Web开发技术栈介绍...", "编程", ["Web", "开发", "前端"]),
            Document("doc6", "项目管理方法", "敏捷开发和项目管理最佳实践...", "管理", ["项目管理", "敏捷", "团队"]),
            Document("doc7", "市场营销策略", "数字化时代的营销新思路...", "商业", ["营销", "策略", "数字化"]),
            Document("doc8", "人工智能伦理", "AI发展中的伦理考量...", "伦理", ["AI", "伦理", "社会"]),
            Document("doc9", "区块链技术", "区块链的原理和应用场景...", "技术", ["区块链", "加密", "去中心化"]),
            Document("doc10", "云计算架构", "企业云计算解决方案...", "技术", ["云计算", "架构", "企业"])
        ]
        
        for doc in sample_docs:
            self.documents[doc.id] = doc
        
        # 创建示例用户
        sample_users = [
            User("user1", {"技术": 0.8, "编程": 0.6}, ["doc1", "doc2"], {"doc1": 4.5, "doc2": 4.0}),
            User("user2", {"编程": 0.9, "Web": 0.7}, ["doc4", "doc5"], {"doc4": 5.0, "doc5": 4.5}),
            User("user3", {"管理": 0.8, "商业": 0.6}, ["doc6", "doc7"], {"doc6": 4.0, "doc7": 3.5}),
            User("user4", {"技术": 0.7, "AI": 0.9}, ["doc1", "doc8"], {"doc1": 4.0, "doc8": 4.5}),
            User("user5", {"技术": 0.6, "云计算": 0.8}, ["doc9", "doc10"], {"doc9": 3.5, "doc10": 4.0})
        ]
        
        for user in sample_users:
            self.users[user.id] = user
        
        # 初始化推荐系统
        self._setup_recommenders()
    
    def _setup_recommenders(self):
        """设置推荐系统"""
        documents = list(self.documents.values())
        users = list(self.users.values())
        
        # 设置各个推荐器
        self.content_recommender.add_documents(documents)
        
        self.collaborative_recommender.add_documents(documents)
        self.collaborative_recommender.add_users(users)
        self.collaborative_recommender.build_rating_matrix()
        
        self.hybrid_recommender.add_documents(documents)
        self.hybrid_recommender.add_users(users)
        self.hybrid_recommender.train()
        
        self.knowledge_recommender.add_documents(documents)
        self.knowledge_recommender.set_category_preferences({
            "技术": 0.8, "编程": 0.7, "管理": 0.6, "商业": 0.5, "伦理": 0.4
        })
        self.knowledge_recommender.set_tag_preferences({
            "AI": 0.9, "机器学习": 0.8, "Python": 0.7, "Web": 0.6, "管理": 0.5
        })
    
    def get_recommendations(self, user_id: str, method: str = "hybrid", top_k: int = 5) -> List[Recommendation]:
        """获取推荐"""
        if user_id not in self.users:
            return []
        
        user = self.users[user_id]
        
        if method == "content":
            recommendations = self.content_recommender.recommend(user, top_k)
        elif method == "collaborative":
            recommendations = self.collaborative_recommender.recommend(user, top_k)
        elif method == "hybrid":
            recommendations = self.hybrid_recommender.recommend(user, top_k)
        elif method == "knowledge":
            recommendations = self.knowledge_recommender.recommend(user, top_k)
        else:
            return []
        
        # 保存到历史记录
        self.recommendation_history.extend(recommendations)
        
        return recommendations
    
    def compare_recommendation_methods(self, user_id: str, top_k: int = 5):
        """比较不同推荐方法"""
        if user_id not in self.users:
            print("用户不存在")
            return
        
        print(f"\n用户 {user_id} 的推荐比较:")
        print("=" * 80)
        
        methods = ["content", "collaborative", "hybrid", "knowledge"]
        all_results = {}
        
        for method in methods:
            print(f"\n{method.upper()} 推荐:")
            print("-" * 50)
            
            recommendations = self.get_recommendations(user_id, method, top_k)
            all_results[method] = recommendations
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    doc = self.documents[rec.document_id]
                    print(f"{i}. {doc.title}")
                    print(f"   分数: {rec.score:.3f}")
                    print(f"   理由: {rec.reason}")
                    print(f"   类别: {doc.category}")
            else:
                print("无推荐结果")
        
        return all_results
    
    def analyze_recommendation_diversity(self, recommendations: List[Recommendation]):
        """分析推荐多样性"""
        if not recommendations:
            print("没有推荐结果")
            return
        
        # 统计类别分布
        categories = []
        tags = []
        
        for rec in recommendations:
            doc = self.documents[rec.document_id]
            categories.append(doc.category)
            tags.extend(doc.tags)
        
        category_counts = Counter(categories)
        tag_counts = Counter(tags)
        
        print(f"\n推荐多样性分析:")
        print("=" * 60)
        print(f"总推荐数: {len(recommendations)}")
        print(f"类别数量: {len(category_counts)}")
        print(f"标签数量: {len(tag_counts)}")
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 类别分布
        if category_counts:
            axes[0].pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
            axes[0].set_title('推荐类别分布')
        
        # 标签分布
        if tag_counts:
            top_tags = dict(tag_counts.most_common(10))
            axes[1].barh(range(len(top_tags)), list(top_tags.values()))
            axes[1].set_yticks(range(len(top_tags)))
            axes[1].set_yticklabels(list(top_tags.keys()))
            axes[1].set_title('热门标签分布')
            axes[1].set_xlabel('频次')
        
        plt.tight_layout()
        plt.show()
        
        # 计算多样性指标
        diversity_score = len(category_counts) / len(recommendations)
        print(f"\n多样性分数: {diversity_score:.3f}")
        
        return {
            "diversity_score": diversity_score,
            "category_distribution": dict(category_counts),
            "tag_distribution": dict(tag_counts)
        }
    
    def simulate_user_interaction(self, user_id: str, num_interactions: int = 10):
        """模拟用户交互"""
        if user_id not in self.users:
            print("用户不存在")
            return
        
        user = self.users[user_id]
        print(f"\n模拟用户 {user_id} 的交互:")
        print("=" * 60)
        
        interaction_history = []
        
        for i in range(num_interactions):
            # 获取推荐
            recommendations = self.get_recommendations(user_id, "hybrid", 3)
            
            if not recommendations:
                break
            
            # 模拟用户选择（偏向高分推荐）
            probabilities = np.array([rec.score for rec in recommendations])
            probabilities = probabilities / probabilities.sum()
            
            chosen_idx = np.random.choice(len(recommendations), p=probabilities)
            chosen_rec = recommendations[chosen_idx]
            
            # 模拟评分（基于文档质量和用户偏好）
            doc = self.documents[chosen_rec.document_id]
            base_rating = random.uniform(3.0, 5.0)
            
            # 根据用户偏好调整评分
            if doc.category in user.preferences:
                base_rating += user.preferences[doc.category] * 1.0
            
            rating = min(5.0, max(1.0, base_rating))
            
            # 更新用户数据
            user.viewed_documents.append(chosen_rec.document_id)
            user.ratings[chosen_rec.document_id] = rating
            
            interaction_history.append({
                "step": i + 1,
                "document": doc.title,
                "category": doc.category,
                "rating": rating,
                "recommendation_score": chosen_rec.score
            })
            
            print(f"第{i+1}步: 选择《{doc.title}》，评分 {rating:.1f}")
        
        # 重新训练协同过滤模型
        self.collaborative_recommender.build_rating_matrix()
        self.hybrid_recommender.train()
        
        return interaction_history
    
    def evaluate_recommendation_quality(self, user_id: str, test_interactions: Dict[str, float]):
        """评估推荐质量"""
        if user_id not in self.users:
            print("用户不存在")
            return
        
        print(f"\n评估用户 {user_id} 的推荐质量:")
        print("=" * 60)
        
        methods = ["content", "collaborative", "hybrid", "knowledge"]
        evaluation_results = {}
        
        for method in methods:
            recommendations = self.get_recommendations(user_id, method, 10)
            metrics = self.evaluator.evaluate_recommendations(
                recommendations, test_interactions, k=5
            )
            evaluation_results[method] = metrics
            
            print(f"\n{method.upper()} 方法:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.3f}")
        
        # 可视化评估结果
        self._plot_evaluation_results(evaluation_results)
        
        return evaluation_results
    
    def _plot_evaluation_results(self, evaluation_results: Dict[str, Dict[str, float]]):
        """绘制评估结果"""
        methods = list(evaluation_results.keys())
        metrics = list(evaluation_results[methods[0]].keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [evaluation_results[method][metric] for method in methods]
            
            axes[i].bar(methods, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('分数')
            axes[i].tick_params(axis='x', rotation=45)
        
        # 删除多余的子图
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
    
    def run_interactive_demo(self):
        """运行交互式演示"""
        print("\n📚 文档推荐系统教育应用")
        print("=" * 50)
        print("可用用户:", ", ".join(self.users.keys()))
        print("可用方法: content, collaborative, hybrid, knowledge")
        
        while True:
            print("\n选择操作:")
            print("1. 获取推荐")
            print("2. 比较推荐方法")
            print("3. 多样性分析")
            print("4. 模拟用户交互")
            print("5. 评估推荐质量")
            print("6. 添加新用户")
            print("7. 添加新文档")
            print("8. 查看用户资料")
            print("0. 退出")
            
            choice = input("\n请选择 (0-8): ").strip()
            
            if choice == '0':
                break
            
            elif choice == '1':
                user_id = input("用户ID: ").strip()
                if user_id in self.users:
                    method = input("推荐方法 (content/collaborative/hybrid/knowledge): ").strip()
                    top_k = int(input("推荐数量 (默认5): ").strip() or "5")
                    
                    recommendations = self.get_recommendations(user_id, method, top_k)
                    
                    print(f"\n为用户 {user_id} 推荐 ({method}):")
                    for i, rec in enumerate(recommendations, 1):
                        doc = self.documents[rec.document_id]
                        print(f"{i}. {doc.title}")
                        print(f"   分数: {rec.score:.3f}")
                        print(f"   理由: {rec.reason}")
                        print(f"   类别: {doc.category} | 标签: {', '.join(doc.tags)}")
                else:
                    print("用户不存在")
            
            elif choice == '2':
                user_id = input("用户ID: ").strip()
                if user_id in self.users:
                    top_k = int(input("推荐数量 (默认5): ").strip() or "5")
                    self.compare_recommendation_methods(user_id, top_k)
                else:
                    print("用户不存在")
            
            elif choice == '3':
                user_id = input("用户ID: ").strip()
                if user_id in self.users:
                    method = input("推荐方法: ").strip()
                    recommendations = self.get_recommendations(user_id, method, 10)
                    self.analyze_recommendation_diversity(recommendations)
                else:
                    print("用户不存在")
            
            elif choice == '4':
                user_id = input("用户ID: ").strip()
                if user_id in self.users:
                    num_interactions = int(input("交互次数 (默认10): ").strip() or "10")
                    self.simulate_user_interaction(user_id, num_interactions)
                else:
                    print("用户不存在")
            
            elif choice == '5':
                user_id = input("用户ID: ").strip()
                if user_id in self.users:
                    print("使用随机测试数据进行评估...")
                    test_data = {
                        f"doc{i}": random.uniform(1, 5) 
                        for i in range(1, 11) if random.random() > 0.5
                    }
                    self.evaluate_recommendation_quality(user_id, test_data)
                else:
                    print("用户不存在")
            
            elif choice == '6':
                user_id = input("新用户ID: ").strip()
                if user_id not in self.users:
                    print("输入用户偏好 (类别=权重，用逗号分隔):")
                    prefs_input = input().strip()
                    preferences = {}
                    
                    if prefs_input:
                        for pref in prefs_input.split(','):
                            if '=' in pref:
                                category, weight = pref.split('=')
                                preferences[category.strip()] = float(weight.strip())
                    
                    new_user = User(user_id, preferences, [], {})
                    self.users[user_id] = new_user
                    
                    # 更新推荐系统
                    self.collaborative_recommender.add_users([new_user])
                    self.hybrid_recommender.add_users([new_user])
                    
                    print("用户添加成功!")
                else:
                    print("用户已存在")
            
            elif choice == '7':
                doc_id = input("文档ID: ").strip()
                if doc_id not in self.documents:
                    title = input("标题: ").strip()
                    content = input("内容: ").strip()
                    category = input("类别: ").strip()
                    tags_input = input("标签 (用逗号分隔): ").strip()
                    tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
                    
                    new_doc = Document(doc_id, title, content, category, tags)
                    self.documents[doc_id] = new_doc
                    
                    # 更新推荐系统
                    self.content_recommender.add_documents([new_doc])
                    self.collaborative_recommender.add_documents([new_doc])
                    self.hybrid_recommender.add_documents([new_doc])
                    self.knowledge_recommender.add_documents([new_doc])
                    
                    print("文档添加成功!")
                else:
                    print("文档已存在")
            
            elif choice == '8':
                user_id = input("用户ID: ").strip()
                if user_id in self.users:
                    user = self.users[user_id]
                    print(f"\n用户 {user_id} 资料:")
                    print(f"偏好: {user.preferences}")
                    print(f"查看过的文档: {len(user.viewed_documents)} 个")
                    print(f"评分记录: {len(user.ratings)} 个")
                    
                    if user.ratings:
                        avg_rating = np.mean(list(user.ratings.values()))
                        print(f"平均评分: {avg_rating:.2f}")
                else:
                    print("用户不存在")

def main():
    """主函数"""
    print("初始化文档推荐系统...")
    
    app = DocumentRecommendationApp()
    app.run_interactive_demo()

if __name__ == "__main__":
    main()
