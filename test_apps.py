#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the advanced NLP applications
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_text_generation_app():
    """Test the text generation application"""
    try:
        print("Testing Text Generation App...")
        
        # Import with explicit encoding handling
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "text_generation_app", 
            "advanced_nlp_apps/text_generation_app.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["text_generation_app"] = module
        spec.loader.exec_module(module)
        
        # Get the class
        TextGenerationApp = getattr(module, 'TextGenerationApp')
        
        # Create instance
        app = TextGenerationApp()
        print("✓ TextGenerationApp instance created successfully")
        
        # Test basic generation
        result = app.generate_text('ngram', 'Hello world', max_length=20)
        print(f"✓ Generated text: {result.generated_text}")
        print(f"✓ Confidence: {result.confidence}")
        
        return True
        
    except Exception as e:
        print(f"✗ Text Generation App test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_question_answering_app():
    """Test the question answering application"""
    try:
        print("\nTesting Question Answering App...")
        
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "question_answering_app", 
            "advanced_nlp_apps/question_answering_app.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        QuestionAnsweringApp = getattr(module, 'QuestionAnsweringApp')
        app = QuestionAnsweringApp()
        print("✓ QuestionAnsweringApp instance created successfully")
        
        # Test basic QA
        context = "Python is a programming language. It is widely used for data science."
        question = "What is Python?"
        answer = app.answer_question(question, context)
        print(f"✓ Answer: {answer}")
        
        return True
        
    except Exception as e:
        print(f"✗ Question Answering App test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_document_recommendation_app():
    """Test the document recommendation application"""
    try:
        print("\nTesting Document Recommendation App...")
        
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "document_recommendation_app", 
            "advanced_nlp_apps/document_recommendation_app.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        DocumentRecommendationApp = getattr(module, 'DocumentRecommendationApp')
        app = DocumentRecommendationApp()
        print("✓ DocumentRecommendationApp instance created successfully")
        
        # Test basic recommendation
        sample_docs = [
            "Machine learning is a subset of artificial intelligence",
            "Python programming language is popular for data science",
            "Natural language processing deals with text analysis"
        ]
        
        recommendations = app.get_content_based_recommendations(
            "artificial intelligence", sample_docs, top_k=2
        )
        print(f"✓ Recommendations: {recommendations}")
        
        return True
        
    except Exception as e:
        print(f"✗ Document Recommendation App test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== Advanced NLP Applications Test Suite ===\n")
    
    results = []
    results.append(test_text_generation_app())
    results.append(test_question_answering_app())
    results.append(test_document_recommendation_app())
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed")

if __name__ == "__main__":
    main()
