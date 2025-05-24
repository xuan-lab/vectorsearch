#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced NLP Applications Demo
高级自然语言处理应用演示

这个演示脚本展示了三个新的高级NLP应用：
1. 文本生成应用 (Text Generation)
2. 问答系统应用 (Question Answering)
3. 文档推荐应用 (Document Recommendation)

This demo script showcases three new advanced NLP applications:
1. Text Generation Application
2. Question Answering System
3. Document Recommendation System
"""

import os
import sys
import time
import importlib.util
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

class AdvancedNLPDemo:
    """高级NLP应用演示系统"""
    
    def __init__(self):
        self.apps = {}
        self.load_applications()
    
    def load_applications(self):
        """加载所有NLP应用"""
        print("🚀 加载高级NLP应用...")
        
        # Load Text Generation App
        try:
            spec = importlib.util.spec_from_file_location(
                "text_generation_app", 
                "advanced_nlp_apps/text_generation_app.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.apps['text_generation'] = getattr(module, 'TextGenerationApp')()
            print("✅ 文本生成应用加载成功")
        except Exception as e:
            print(f"❌ 文本生成应用加载失败: {e}")
        
        # Load Question Answering System
        try:
            spec = importlib.util.spec_from_file_location(
                "question_answering_app", 
                "advanced_nlp_apps/question_answering_app.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.apps['qa_system'] = getattr(module, 'QASystem')()
            print("✅ 问答系统加载成功")
        except Exception as e:
            print(f"❌ 问答系统加载失败: {e}")
        
        # Load Document Recommendation App
        try:
            spec = importlib.util.spec_from_file_location(
                "document_recommendation_app", 
                "advanced_nlp_apps/document_recommendation_app.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.apps['doc_recommendation'] = getattr(module, 'DocumentRecommendationApp')()
            print("✅ 文档推荐应用加载成功")
        except Exception as e:
            print(f"❌ 文档推荐应用加载失败: {e}")
        
        print(f"\n📊 成功加载 {len(self.apps)} 个应用\n")
    
    def demo_text_generation(self):
        """演示文本生成功能"""
        print("=" * 60)
        print("🎯 文本生成应用演示 (Text Generation Demo)")
        print("=" * 60)
        
        if 'text_generation' not in self.apps:
            print("❌ 文本生成应用未加载")
            return
        
        app = self.apps['text_generation']
        
        # Demo different generation methods
        demo_texts = [
            ("今天天气很好", "ngram"),
            ("人工智能技术", "markov"),
            ("科幻故事", "template")
        ]
        
        for seed_text, method in demo_texts:
            print(f"\n🔸 使用 {method.upper()} 方法生成文本")
            print(f"种子文本: {seed_text}")
            
            if method == "template":
                result = app.generate_text(method, template_category="story_beginning")
            else:
                result = app.generate_text(method, seed_text, max_length=50)
            
            print(f"生成结果: {result.generated_text}")
            print(f"置信度: {result.confidence:.3f}")
            print("-" * 40)
    
    def demo_question_answering(self):
        """演示问答系统功能"""
        print("=" * 60)
        print("🎯 问答系统演示 (Question Answering Demo)")
        print("=" * 60)
        
        if 'qa_system' not in self.apps:
            print("❌ 问答系统未加载")
            return
        
        app = self.apps['qa_system']
        
        # Demo questions and contexts
        demo_qa_pairs = [
            {
                "question": "什么是机器学习？",
                "context": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。机器学习算法通过分析数据来构建数学模型，以便对新数据进行预测或决策。"
            },
            {
                "question": "Python有什么优点？",
                "context": "Python是一种高级编程语言，具有简洁的语法和强大的功能。它广泛用于数据科学、人工智能、Web开发和自动化脚本。Python拥有丰富的库生态系统。"
            },
            {
                "question": "深度学习如何工作？",
                "context": "深度学习是机器学习的子集，使用多层神经网络来学习数据的复杂模式。通过反向传播算法，神经网络可以自动学习特征表示，无需手动特征工程。"
            }
        ]
        
        for qa in demo_qa_pairs:
            print(f"\n🔸 问题: {qa['question']}")
            print(f"上下文: {qa['context'][:50]}...")
            
            result = app.answer_question(qa['question'], qa['context'])
            print(f"回答: {result.answer}")
            print(f"方法: {result.method}")
            print(f"置信度: {result.confidence:.3f}")
            print("-" * 40)
    
    def demo_document_recommendation(self):
        """演示文档推荐功能"""
        print("=" * 60)
        print("🎯 文档推荐系统演示 (Document Recommendation Demo)")
        print("=" * 60)
        
        if 'doc_recommendation' not in self.apps:
            print("❌ 文档推荐应用未加载")
            return
        
        app = self.apps['doc_recommendation']
        
        # Demo different recommendation methods
        user_profiles = [
            {
                "name": "AI研究者",
                "preferences": {"interests": ["machine learning", "artificial intelligence", "deep learning"]}
            },
            {
                "name": "Python开发者", 
                "preferences": {"interests": ["python programming", "web development", "data science"]}
            },
            {
                "name": "数据科学家",
                "preferences": {"interests": ["data analysis", "statistics", "visualization"]}
            }
        ]
        
        methods = ["content_based", "collaborative", "hybrid"]
        
        for profile in user_profiles:
            print(f"\n🔸 用户: {profile['name']}")
            print(f"兴趣: {', '.join(profile['preferences']['interests'])}")
            
            for method in methods:
                print(f"\n  📋 {method.upper()} 推荐:")
                try:
                    recommendations = app.get_recommendations(
                        method, profile['preferences'], top_k=3
                    )
                    
                    for i, rec in enumerate(recommendations, 1):
                        print(f"    {i}. {rec.content[:60]}... (分数: {rec.score:.3f})")
                        
                except Exception as e:
                    print(f"    ❌ {method} 推荐失败: {e}")
            
            print("-" * 40)
    
    def run_performance_analysis(self):
        """运行性能分析"""
        print("=" * 60)
        print("📊 性能分析 (Performance Analysis)")
        print("=" * 60)
        
        # Analyze response times
        analysis_results = {}
        
        # Text Generation Performance
        if 'text_generation' in self.apps:
            start_time = time.time()
            app = self.apps['text_generation']
            app.generate_text('ngram', '测试文本', max_length=20)
            analysis_results['text_generation'] = time.time() - start_time
        
        # QA System Performance
        if 'qa_system' in self.apps:
            start_time = time.time()
            app = self.apps['qa_system']
            app.answer_question("测试问题", "这是一个测试上下文")
            analysis_results['qa_system'] = time.time() - start_time
        
        # Recommendation System Performance
        if 'doc_recommendation' in self.apps:
            start_time = time.time()
            app = self.apps['doc_recommendation']
            app.get_recommendations('content_based', {'interests': ['test']}, top_k=3)
            analysis_results['doc_recommendation'] = time.time() - start_time
        
        print("⏱️ 响应时间分析:")
        for app_name, response_time in analysis_results.items():
            print(f"  {app_name}: {response_time:.3f}秒")
        
        print(f"\n📈 平均响应时间: {sum(analysis_results.values())/len(analysis_results):.3f}秒")
    
    def show_menu(self):
        """显示主菜单"""
        print("\n" + "=" * 60)
        print("🎓 高级NLP应用教育演示系统")
        print("Advanced NLP Applications Educational Demo")
        print("=" * 60)
        
        menu_options = [
            "1. 🎨 文本生成演示 (Text Generation Demo)",
            "2. ❓ 问答系统演示 (Question Answering Demo)", 
            "3. 📚 文档推荐演示 (Document Recommendation Demo)",
            "4. 📊 性能分析 (Performance Analysis)",
            "5. 🔄 重新加载应用 (Reload Applications)",
            "6. ❌ 退出 (Exit)"
        ]
        
        for option in menu_options:
            print(option)
        
        print("\n📋 应用状态:")
        for app_name in ['text_generation', 'qa_system', 'doc_recommendation']:
            status = "✅ 已加载" if app_name in self.apps else "❌ 未加载"
            print(f"  {app_name}: {status}")
    
    def run_interactive_demo(self):
        """运行交互式演示"""
        print("🎉 欢迎使用高级NLP应用演示系统!")
        print("Welcome to Advanced NLP Applications Demo!")
        
        while True:
            self.show_menu()
            
            try:
                choice = input("\n请选择操作 (Enter your choice): ").strip()
                
                if choice == '1':
                    self.demo_text_generation()
                elif choice == '2':
                    self.demo_question_answering()
                elif choice == '3':
                    self.demo_document_recommendation()
                elif choice == '4':
                    self.run_performance_analysis()
                elif choice == '5':
                    self.load_applications()
                elif choice == '6':
                    print("\n👋 感谢使用! Thank you for using the demo!")
                    break
                else:
                    print("❌ 无效选择，请重试。Invalid choice, please try again.")
                
                input("\n按 Enter 继续... (Press Enter to continue...)")
                
            except KeyboardInterrupt:
                print("\n\n👋 演示已中断。Demo interrupted.")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                input("按 Enter 继续... (Press Enter to continue...)")

def main():
    """主函数"""
    demo = AdvancedNLPDemo()
    demo.run_interactive_demo()

if __name__ == "__main__":
    main()
