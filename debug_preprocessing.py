#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug preprocessing to understand why similarity scores are 0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_vectorizer import TextVectorizer
from src.utils import load_json_data

def debug_preprocessing():
    print("Starting debug...")
    try:
        # Load sample data
        documents = load_json_data("data/sample_documents.json")
        print(f"Loaded {len(documents)} documents")
        
        vectorizer = TextVectorizer()
        print("Created vectorizer")
        
        # Test preprocessing on first document
        doc = documents[0]
        print(f"Original: {doc['content']}")
        
        processed = vectorizer.preprocess_text(doc['content'])
        print(f"Processed: '{processed}'")
        
        # Test TF-IDF with different settings
        texts = [doc['content'] for doc in documents[:3]]
        processed_texts = [vectorizer.preprocess_text(text) for text in texts]
        
        print(f"\nProcessed texts:")
        for i, text in enumerate(processed_texts):
            print(f"{i+1}: '{text}'")
        
        # Test TF-IDF matrix directly
        from sklearn.feature_extraction.text import TfidfVectorizer
        print(f"\nTesting TF-IDF configurations:")
        
        # Test 1: Default settings with Chinese text
        tfidf1 = TfidfVectorizer(max_features=1000)
        matrix1 = tfidf1.fit_transform(processed_texts)
        print(f"Default TF-IDF: {matrix1.shape}, non-zero: {matrix1.nnz}")
        vocab1 = tfidf1.get_feature_names_out()
        print(f"Vocabulary sample: {list(vocab1[:10])}")
        
        # Test 2: Custom token pattern for Chinese
        tfidf2 = TfidfVectorizer(
            max_features=1000, 
            token_pattern=r'(?u)\b\w+\b',  # Unicode aware pattern
            analyzer='char_wb',  # Character n-grams within word boundaries
            ngram_range=(1, 3)
        )
        matrix2 = tfidf2.fit_transform(processed_texts)
        print(f"Chinese TF-IDF: {matrix2.shape}, non-zero: {matrix2.nnz}")
        vocab2 = tfidf2.get_feature_names_out()
        print(f"Vocabulary sample: {list(vocab2[:10])}")
        
        # Test query transformation
        query = "机器学习"
        query_processed = vectorizer.preprocess_text(query)
        print(f"\nQuery: '{query}' -> '{query_processed}'")
        
        query_vec1 = tfidf1.transform([query_processed])
        query_vec2 = tfidf2.transform([query_processed])
        print(f"Query vector 1 non-zero: {query_vec1.nnz}")
        print(f"Query vector 2 non-zero: {query_vec2.nnz}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_preprocessing()
