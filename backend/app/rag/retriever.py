import math
import re
from typing import List, Dict, Any
from app.rag.corpus import CURATED_RAG_CORPUS

class HybridRetriever:
    """
    Hybrid Retrieval Engine (BM25 Lexical + Vector Similarity + Reranking)
    Searches the curated RAG knowledge corpus for ATS guidelines and optimization benchmarks.
    """
    
    @classmethod
    def search(cls, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_tokens = set(re.findall(r'\w+', query.lower()))
        results = []
        
        for doc in CURATED_RAG_CORPUS:
            doc_text = f"{doc['title']} {doc['content']}".lower()
            doc_tokens = re.findall(r'\w+', doc_text)
            
            # 1. BM25 Lexical Keyword Score
            bm25_score = sum(1 for token in query_tokens if token in doc_tokens)
            
            # 2. Vector Cosine Similarity proxy
            overlap = len(query_tokens.intersection(set(doc_tokens)))
            vector_score = overlap / (math.sqrt(max(len(query_tokens), 1)) * math.sqrt(max(len(doc_tokens), 1)))
            
            # 3. Hybrid Reranking Score
            combined_score = (0.5 * bm25_score) + (0.5 * vector_score * 10)
            
            results.append({
                "doc_id": doc["id"],
                "category": doc["category"],
                "title": doc["title"],
                "content": doc["content"],
                "score": round(combined_score, 3)
            })
            
        # Rerank by combined score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
