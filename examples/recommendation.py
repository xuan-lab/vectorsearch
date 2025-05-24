#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于向量的推荐系统示例

这个模块演示如何构建基于内容的推荐系统，包括用户画像构建、物品推荐和推荐解释。

作者: Vector Search Learning Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
import numpy as np
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.text_vectorizer import TextVectorizer
from src.advanced_search import FAISSSearch
from src.utils import load_documents, save_json


class ContentRecommendationSystem:
    """
    基于内容的推荐系统
    
    支持多种推荐策略和用户行为分析
    """
    
    def __init__(self, documents: List[Dict[str, Any]], vector_type: str = 'semantic'):
        """
        初始化推荐系统
        
        Args:
            documents: 文档列表
            vector_type: 向量类型 ('tfidf', 'semantic')
        """
        self.documents = documents
        self.vector_type = vector_type
        self.vectorizer = TextVectorizer()
        self.item_vectors = None
        self.faiss_search = None
        self.user_profiles = {}
        self.interaction_history = defaultdict(list)
        self.item_popularity = Counter()
        
        self._build_item_vectors()
        self._build_search_index()
    
    def _build_item_vectors(self) -> None:
        """构建物品向量表示"""
        print(f"构建物品向量 (类型: {self.vector_type})...")
        texts = [doc['content'] for doc in self.documents]
        
        if self.vector_type == 'semantic':
            try:
                self.item_vectors = self.vectorizer.sentence_transformer_vectorize(texts)
                print(f"语义向量维度: {self.item_vectors.shape}")
            except Exception as e:
                print(f"语义向量构建失败，使用TF-IDF: {e}")
                self.vector_type = 'tfidf'
                tfidf_vectors = self.vectorizer.tfidf_vectorize(texts)
                self.item_vectors = normalize(tfidf_vectors.toarray().astype('float32'))
        else:
            tfidf_vectors = self.vectorizer.tfidf_vectorize(texts)
            self.item_vectors = normalize(tfidf_vectors.toarray().astype('float32'))
        
        print(f"物品向量构建完成，形状: {self.item_vectors.shape}")
    
    def _build_search_index(self) -> None:
        """构建搜索索引用于快速检索"""
        print("构建FAISS搜索索引...")
        self.faiss_search = FAISSSearch(vector_dim=self.item_vectors.shape[1])
        self.faiss_search.add_vectors(self.item_vectors)
        print("搜索索引构建完成")
    
    def add_user_interaction(self, user_id: str, item_id: int, interaction_type: str, 
                           rating: Optional[float] = None, timestamp: Optional[float] = None) -> None:
        """
        添加用户交互记录
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            interaction_type: 交互类型 ('view', 'like', 'dislike', 'share', 'rate')
            rating: 评分 (可选)
            timestamp: 时间戳 (可选)
        """
        if item_id >= len(self.documents):
            raise ValueError(f"无效的物品ID: {item_id}")
        
        interaction = {
            'item_id': item_id,
            'type': interaction_type,
            'rating': rating,
            'timestamp': timestamp or time.time()
        }
        
        self.interaction_history[user_id].append(interaction)
        
        # 更新物品流行度
        if interaction_type in ['like', 'share', 'rate']:
            self.item_popularity[item_id] += 1
        
        # 更新用户画像
        self._update_user_profile(user_id)
    
    def _update_user_profile(self, user_id: str) -> None:
        """更新用户画像"""
        interactions = self.interaction_history[user_id]
        
        if not interactions:
            return
        
        # 分析用户偏好
        liked_items = []
        disliked_items = []
        viewed_items = []
        
        for interaction in interactions:
            item_id = interaction['item_id']
            interaction_type = interaction['type']
            rating = interaction.get('rating')
            
            if interaction_type == 'like' or (rating and rating >= 4):
                liked_items.append(item_id)
            elif interaction_type == 'dislike' or (rating and rating <= 2):
                disliked_items.append(item_id)
            elif interaction_type == 'view':
                viewed_items.append(item_id)
        
        # 构建用户向量
        user_vector = self._compute_user_vector(liked_items, disliked_items, viewed_items)
        
        # 分析用户偏好类别
        preferred_categories = self._analyze_category_preferences(interactions)
        
        # 更新用户画像
        self.user_profiles[user_id] = {
            'vector': user_vector,
            'liked_items': set(liked_items),
            'disliked_items': set(disliked_items),
            'viewed_items': set(viewed_items),
            'preferred_categories': preferred_categories,
            'total_interactions': len(interactions),
            'last_updated': time.time()
        }
    
    def _compute_user_vector(self, liked_items: List[int], disliked_items: List[int], 
                           viewed_items: List[int]) -> np.ndarray:
        """计算用户向量表示"""
        if not liked_items and not viewed_items:
            # 新用户，返回零向量
            return np.zeros(self.item_vectors.shape[1])
        
        # 正向偏好（喜欢的物品）
        positive_vectors = []
        if liked_items:
            positive_vectors.extend([self.item_vectors[i] for i in liked_items])
        
        # 如果没有明确喜欢的物品，使用浏览过的物品
        if not positive_vectors and viewed_items:
            positive_vectors.extend([self.item_vectors[i] for i in viewed_items[-5:]])  # 最近5个
        
        if not positive_vectors:
            return np.zeros(self.item_vectors.shape[1])
        
        # 计算正向偏好的平均向量
        user_vector = np.mean(positive_vectors, axis=0)
        
        # 如果有明确不喜欢的物品，减去这些向量的影响
        if disliked_items:
            negative_vectors = [self.item_vectors[i] for i in disliked_items]
            negative_vector = np.mean(negative_vectors, axis=0)
            user_vector = user_vector - 0.3 * negative_vector  # 减少负向影响
        
        return normalize([user_vector])[0]
    
    def _analyze_category_preferences(self, interactions: List[Dict]) -> Dict[str, float]:
        """分析用户类别偏好"""
        category_scores = defaultdict(float)
        
        for interaction in interactions:
            item_id = interaction['item_id']
            interaction_type = interaction['type']
            category = self.documents[item_id]['category']
            
            # 根据交互类型给予不同权重
            if interaction_type == 'like':
                category_scores[category] += 1.0
            elif interaction_type == 'share':
                category_scores[category] += 0.8
            elif interaction_type == 'view':
                category_scores[category] += 0.2
            elif interaction_type == 'dislike':
                category_scores[category] -= 0.5
        
        # 归一化分数
        if category_scores:
            max_score = max(category_scores.values())
            if max_score > 0:
                for category in category_scores:
                    category_scores[category] /= max_score
        
        return dict(category_scores)
    
    def recommend_items(self, user_id: str, num_recommendations: int = 10, 
                       strategy: str = 'hybrid', diversity_factor: float = 0.3) -> List[Dict[str, Any]]:
        """
        为用户推荐物品
        
        Args:
            user_id: 用户ID
            num_recommendations: 推荐数量
            strategy: 推荐策略 ('content', 'popularity', 'hybrid')
            diversity_factor: 多样性因子 (0-1)
        
        Returns:
            推荐结果列表
        """
        if strategy == 'content':
            return self._content_based_recommend(user_id, num_recommendations, diversity_factor)
        elif strategy == 'popularity':
            return self._popularity_based_recommend(user_id, num_recommendations)
        elif strategy == 'hybrid':
            return self._hybrid_recommend(user_id, num_recommendations, diversity_factor)
        else:
            raise ValueError(f"未支持的推荐策略: {strategy}")
    
    def _content_based_recommend(self, user_id: str, num_recommendations: int, 
                               diversity_factor: float = 0.3) -> List[Dict[str, Any]]:
        """基于内容的推荐"""
        if user_id not in self.user_profiles:
            # 新用户，推荐热门物品
            return self._recommend_popular_items(num_recommendations)
        
        user_profile = self.user_profiles[user_id]
        user_vector = user_profile['vector']
        
        # 排除已经交互过的物品
        excluded_items = user_profile['liked_items'] | user_profile['disliked_items']
        
        # 计算与所有物品的相似度
        similarities = cosine_similarity([user_vector], self.item_vectors)[0]
        
        # 创建候选列表
        candidates = []
        for i, similarity in enumerate(similarities):
            if i not in excluded_items:
                candidates.append({
                    'item_id': i,
                    'similarity': similarity,
                    'document': self.documents[i]
                })
        
        # 按相似度排序
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 应用多样性
        if diversity_factor > 0:
            candidates = self._apply_diversity(candidates, diversity_factor)
        
        # 获取top-k推荐
        recommendations = candidates[:num_recommendations]
        
        # 添加推荐原因
        for rec in recommendations:
            rec['reason'] = self._generate_recommendation_reason(user_id, rec['item_id'])
            rec['strategy'] = 'content'
        
        return recommendations
    
    def _popularity_based_recommend(self, user_id: str, num_recommendations: int) -> List[Dict[str, Any]]:
        """基于流行度的推荐"""
        user_profile = self.user_profiles.get(user_id, {})
        excluded_items = user_profile.get('liked_items', set()) | user_profile.get('disliked_items', set())
        
        # 获取热门物品
        popular_items = [(item_id, count) for item_id, count in self.item_popularity.most_common() 
                        if item_id not in excluded_items]
        
        recommendations = []
        for i, (item_id, popularity) in enumerate(popular_items[:num_recommendations]):
            recommendations.append({
                'item_id': item_id,
                'popularity': popularity,
                'similarity': 0.5,  # 默认相似度
                'document': self.documents[item_id],
                'reason': f'热门物品 (流行度: {popularity})',
                'strategy': 'popularity'
            })
        
        return recommendations
    
    def _hybrid_recommend(self, user_id: str, num_recommendations: int, 
                         diversity_factor: float) -> List[Dict[str, Any]]:
        """混合推荐策略"""
        # 获取内容推荐和流行度推荐
        content_recs = self._content_based_recommend(user_id, num_recommendations * 2, diversity_factor)
        popularity_recs = self._popularity_based_recommend(user_id, num_recommendations)
        
        # 合并推荐，内容推荐权重更高
        hybrid_scores = defaultdict(float)
        item_info = {}
        
        # 内容推荐权重
        for rec in content_recs:
            item_id = rec['item_id']
            hybrid_scores[item_id] += 0.7 * rec['similarity']
            item_info[item_id] = rec
        
        # 流行度推荐权重
        for rec in popularity_recs:
            item_id = rec['item_id']
            normalized_popularity = rec['popularity'] / max(self.item_popularity.values()) if self.item_popularity else 0
            hybrid_scores[item_id] += 0.3 * normalized_popularity
            if item_id not in item_info:
                item_info[item_id] = rec
        
        # 排序并返回top-k
        sorted_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for i, (item_id, score) in enumerate(sorted_items[:num_recommendations]):
            rec = item_info[item_id].copy()
            rec['similarity'] = score
            rec['strategy'] = 'hybrid'
            rec['reason'] = self._generate_recommendation_reason(user_id, item_id, 'hybrid')
            recommendations.append(rec)
        
        return recommendations
    
    def _recommend_popular_items(self, num_recommendations: int) -> List[Dict[str, Any]]:
        """为新用户推荐热门物品"""
        if not self.item_popularity:
            # 如果没有流行度数据，随机推荐
            import random
            item_ids = random.sample(range(len(self.documents)), 
                                   min(num_recommendations, len(self.documents)))
            return [{
                'item_id': item_id,
                'similarity': 0.5,
                'document': self.documents[item_id],
                'reason': '新用户推荐',
                'strategy': 'cold_start'
            } for item_id in item_ids]
        
        return self._popularity_based_recommend('', num_recommendations)
    
    def _apply_diversity(self, candidates: List[Dict[str, Any]], 
                        diversity_factor: float) -> List[Dict[str, Any]]:
        """应用多样性策略"""
        if diversity_factor <= 0:
            return candidates
        
        selected = []
        remaining = candidates.copy()
        
        # 选择第一个最相似的
        if remaining:
            selected.append(remaining.pop(0))
        
        # 应用多样性选择
        while remaining and len(selected) < len(candidates):
            best_score = -1
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                # 计算与已选择物品的平均相似度
                selected_vectors = [self.item_vectors[rec['item_id']] for rec in selected]
                candidate_vector = self.item_vectors[candidate['item_id']]
                
                avg_similarity = np.mean([cosine_similarity([candidate_vector], [sv])[0][0] 
                                        for sv in selected_vectors])
                
                # 多样性得分：原始相似度 - 多样性惩罚
                diversity_score = candidate['similarity'] - diversity_factor * avg_similarity
                
                if diversity_score > best_score:
                    best_score = diversity_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _generate_recommendation_reason(self, user_id: str, item_id: int, 
                                      strategy: str = 'content') -> str:
        """生成推荐原因"""
        if user_id not in self.user_profiles:
            return "新用户推荐"
        
        user_profile = self.user_profiles[user_id]
        document = self.documents[item_id]
        
        if strategy == 'content':
            # 找到最相似的用户喜欢的物品
            liked_items = list(user_profile['liked_items'])
            if liked_items:
                item_vector = self.item_vectors[item_id]
                max_similarity = 0
                most_similar_item = None
                
                for liked_id in liked_items:
                    liked_vector = self.item_vectors[liked_id]
                    similarity = cosine_similarity([item_vector], [liked_vector])[0][0]
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_item = liked_id
                
                if most_similar_item is not None:
                    similar_doc = self.documents[most_similar_item]
                    return f"因为与您喜欢的'{similar_doc['title']}'相似"
        
        # 基于类别偏好
        preferred_categories = user_profile.get('preferred_categories', {})
        if document['category'] in preferred_categories:
            return f"基于您对{document['category']}类别的偏好"
        
        return "基于您的浏览历史推荐"


def main():
    """主函数：命令行界面"""
    parser = argparse.ArgumentParser(description="基于内容的推荐系统")
    parser.add_argument('--documents', '-d', required=True, help='文档文件路径')
    parser.add_argument('--user-id', '-u', help='用户ID')
    parser.add_argument('--num-recs', '-n', type=int, default=5, help='推荐数量')
    parser.add_argument('--strategy', '-s', choices=['content', 'popularity', 'hybrid'], 
                       default='hybrid', help='推荐策略')
    parser.add_argument('--vector-type', '-v', choices=['tfidf', 'semantic'], 
                       default='semantic', help='向量类型')
    parser.add_argument('--simulate', action='store_true', help='模拟用户行为')
    parser.add_argument('--interactive', '-i', action='store_true', help='交互式模式')
    
    args = parser.parse_args()
    
    # 加载文档
    documents = load_documents(args.documents)
    
    # 初始化推荐系统
    rec_system = ContentRecommendationSystem(documents, args.vector_type)
    
    # 模拟用户行为（如果指定）
    if args.simulate:
        simulate_user_behavior(rec_system)
    
    if args.interactive:
        # 交互式模式
        run_interactive_recommendation(rec_system)
    elif args.user_id:
        # 为指定用户生成推荐
        recommendations = rec_system.recommend_items(
            args.user_id, args.num_recs, args.strategy
        )
        display_recommendations(args.user_id, recommendations)
    else:
        print("请指定用户ID (--user-id) 或使用交互式模式 (--interactive)")


def simulate_user_behavior(rec_system: ContentRecommendationSystem, 
                          num_users: int = 5, interactions_per_user: int = 10) -> None:
    """模拟用户行为数据"""
    print(f"模拟 {num_users} 个用户的行为数据...")
    
    import random
    
    interaction_types = ['view', 'like', 'dislike', 'share']
    interaction_weights = [0.6, 0.25, 0.05, 0.1]  # 浏览最多，点赞次之
    
    for user_idx in range(num_users):
        user_id = f"user_{user_idx + 1}"
        
        # 为每个用户随机选择偏好类别
        categories = list(set(doc['category'] for doc in rec_system.documents))
        preferred_category = random.choice(categories)
        
        for _ in range(interactions_per_user):
            # 偏向于选择偏好类别的文档
            if random.random() < 0.7:  # 70%概率选择偏好类别
                category_items = [i for i, doc in enumerate(rec_system.documents) 
                                if doc['category'] == preferred_category]
                if category_items:
                    item_id = random.choice(category_items)
                else:
                    item_id = random.randint(0, len(rec_system.documents) - 1)
            else:
                item_id = random.randint(0, len(rec_system.documents) - 1)
            
            interaction_type = random.choices(interaction_types, weights=interaction_weights)[0]
            
            rec_system.add_user_interaction(user_id, item_id, interaction_type)
    
    print("用户行为模拟完成")


def display_recommendations(user_id: str, recommendations: List[Dict[str, Any]]) -> None:
    """显示推荐结果"""
    print(f"\n为用户 {user_id} 的推荐:")
    print("=" * 60)
    
    if not recommendations:
        print("暂无推荐")
        return
    
    for i, rec in enumerate(recommendations):
        doc = rec['document']
        print(f"{i+1}. {doc['title']}")
        print(f"   类别: {doc['category']}")
        print(f"   相似度: {rec['similarity']:.3f}")
        print(f"   推荐原因: {rec['reason']}")
        print(f"   策略: {rec['strategy']}")
        print(f"   内容: {doc['content'][:80]}...")
        print()


def run_interactive_recommendation(rec_system: ContentRecommendationSystem) -> None:
    """交互式推荐模式"""
    print("\n=== 推荐系统 - 交互式模式 ===")
    print("可用命令:")
    print("  recommend <user_id> [strategy] - 获取推荐")
    print("  interact <user_id> <item_id> <type> - 添加交互")
    print("  list - 显示所有文档")
    print("  quit - 退出")
    
    while True:
        try:
            command = input("\n请输入命令: ").strip().split()
            
            if not command:
                continue
            
            if command[0] == 'quit':
                break
            elif command[0] == 'recommend':
                if len(command) < 2:
                    print("用法: recommend <user_id> [strategy]")
                    continue
                
                user_id = command[1]
                strategy = command[2] if len(command) > 2 else 'hybrid'
                
                recommendations = rec_system.recommend_items(user_id, 5, strategy)
                display_recommendations(user_id, recommendations)
                
            elif command[0] == 'interact':
                if len(command) < 4:
                    print("用法: interact <user_id> <item_id> <type>")
                    continue
                
                user_id = command[1]
                item_id = int(command[2])
                interaction_type = command[3]
                
                rec_system.add_user_interaction(user_id, item_id, interaction_type)
                print(f"已添加交互: {user_id} {interaction_type} 物品 {item_id}")
                
            elif command[0] == 'list':
                print("\n文档列表:")
                for i, doc in enumerate(rec_system.documents):
                    print(f"{i}. {doc['title']} [{doc['category']}]")
                
            else:
                print(f"未知命令: {command[0]}")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"命令执行出错: {e}")


if __name__ == "__main__":
    main()
