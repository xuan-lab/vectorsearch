#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for advanced NLP applications
测试高级NLP应用的简单脚本
"""

import os
import sys
import importlib.util

def test_all_applications():
    """测试所有应用"""
    print("🧪 Testing Advanced NLP Applications")
    print("=" * 50)
    
    results = {}
    
    # Test Text Generation App
    print("\n1. Testing Text Generation App...")
    try:
        spec = importlib.util.spec_from_file_location(
            "text_generation_app", 
            "advanced_nlp_apps/text_generation_app.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        app = getattr(module, 'TextGenerationApp')()
        result = app.generate_text('ngram', 'Hello world', max_length=20)
        
        print(f"✅ Text Generation: {result.generated_text}")
        results['text_generation'] = True
        
    except Exception as e:
        print(f"❌ Text Generation failed: {e}")
        results['text_generation'] = False
    
    # Test Question Answering System
    print("\n2. Testing Question Answering System...")
    try:
        spec = importlib.util.spec_from_file_location(
            "question_answering_app", 
            "advanced_nlp_apps/question_answering_app.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        app = getattr(module, 'QASystem')()
        result = app.answer_question(
            "What is AI?", 
            "Artificial Intelligence is a branch of computer science."
        )
        
        print(f"✅ Question Answering: {result.answer}")
        results['qa_system'] = True
        
    except Exception as e:
        print(f"❌ Question Answering failed: {e}")
        results['qa_system'] = False
    
    # Test Document Recommendation App
    print("\n3. Testing Document Recommendation App...")
    try:
        spec = importlib.util.spec_from_file_location(
            "document_recommendation_app", 
            "advanced_nlp_apps/document_recommendation_app.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        app = getattr(module, 'DocumentRecommendationApp')()
        recommendations = app.get_recommendations(
            'content_based', 
            {'interests': ['machine learning']}, 
            top_k=3
        )
        
        print(f"✅ Document Recommendation: Found {len(recommendations)} recommendations")
        results['doc_recommendation'] = True
        
    except Exception as e:
        print(f"❌ Document Recommendation failed: {e}")
        results['doc_recommendation'] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(results.values())
    total = len(results)
    
    for app_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {app_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} applications working")
    
    if passed == total:
        print("🎉 All applications are working correctly!")
    else:
        print("⚠️ Some applications need attention.")
    
    return results

if __name__ == "__main__":
    test_all_applications()
