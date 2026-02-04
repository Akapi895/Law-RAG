# -*- coding: utf-8 -*-
"""
fusion.py - Fusion Retriever

Kết hợp Vector Search (semantic) + BM25 (keyword)
để có kết quả tìm kiếm tốt hơn.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from llama_index.core import Document, VectorStoreIndex

from src.retrievers.bm25 import BM25Retriever
from config import FUSION_ALPHA, SIMILARITY_TOP_K


class FusionRetriever:
    """
    Fusion Retriever: kết hợp Vector Search + BM25.
    
    Công thức:
        combined_score = α × vector_score + (1-α) × bm25_score
        
    α = FUSION_ALPHA (0.0 - 1.0)
    - α = 1.0: chỉ dùng vector search (semantic)
    - α = 0.0: chỉ dùng BM25 (keyword)
    - α = 0.5: cân bằng cả hai (khuyên dùng)
    """
    
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        bm25_retriever: BM25Retriever,
        alpha: float = None
    ):
        """
        Khởi tạo Fusion Retriever.
        
        Args:
            vector_index: LlamaIndex VectorStoreIndex
            bm25_retriever: BM25Retriever instance
            alpha: Trọng số cho vector search (mặc định từ config)
        """
        self.vector_index = vector_index
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha if alpha is not None else FUSION_ALPHA
    
    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        Thực hiện Fusion Retrieval.
        
        Args:
            query: Câu truy vấn
            top_k: Số lượng kết quả (mặc định = SIMILARITY_TOP_K)
            
        Returns:
            List documents đã được xếp hạng
        """
        if top_k is None:
            top_k = SIMILARITY_TOP_K
        
        # Lấy nhiều hơn để có đủ candidates cho fusion
        fetch_k = top_k * 3
        
        # =====================================================
        # BƯỚC 1: VECTOR SEARCH (Semantic)
        # =====================================================
        vector_results = []
        if self.vector_index is not None:
            retriever = self.vector_index.as_retriever(similarity_top_k=fetch_k)
            vector_nodes = retriever.retrieve(query)
            
            for node in vector_nodes:
                node_id = getattr(node.node, 'doc_id', None) or getattr(node.node, 'id_', str(hash(node.node.text)))
                vector_results.append({
                    'doc_id': node_id,
                    'text': node.node.text,
                    'metadata': node.node.metadata,
                    'vector_score': node.score if node.score else 0.0
                })
        
        # =====================================================
        # BƯỚC 2: BM25 SEARCH (Keyword)
        # =====================================================
        bm25_results = {}
        if self.bm25_retriever.is_ready:
            bm25_docs = self.bm25_retriever.search(query, top_k=fetch_k)
            
            for doc, score in bm25_docs:
                doc_id = getattr(doc, 'doc_id', None) or getattr(doc, 'id_', str(hash(doc.text)))
                bm25_results[doc_id] = {
                    'text': doc.text,
                    'metadata': doc.metadata,
                    'bm25_score': score
                }
        
        # =====================================================
        # BƯỚC 3: FUSION - Kết hợp scores
        # =====================================================
        all_docs: Dict[str, Dict[str, Any]] = {}
        
        # Thêm từ vector results
        for r in vector_results:
            doc_id = r['doc_id']
            all_docs[doc_id] = {
                'text': r['text'],
                'metadata': r['metadata'],
                'vector_score': r['vector_score'],
                'bm25_score': bm25_results.get(doc_id, {}).get('bm25_score', 0.0)
            }
        
        # Thêm từ BM25 results (những cái chưa có)
        for doc_id, r in bm25_results.items():
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    'text': r['text'],
                    'metadata': r['metadata'],
                    'vector_score': 0.0,
                    'bm25_score': r['bm25_score']
                }
        
        if not all_docs:
            return []
        
        # =====================================================
        # BƯỚC 4: NORMALIZE SCORES (0-1)
        # =====================================================
        epsilon = 1e-8
        
        # Normalize vector scores
        vector_scores = np.array([d['vector_score'] for d in all_docs.values()])
        if vector_scores.max() - vector_scores.min() > epsilon:
            vector_scores_norm = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + epsilon)
        else:
            vector_scores_norm = np.ones_like(vector_scores) * 0.5
        
        # Normalize BM25 scores
        bm25_scores = np.array([d['bm25_score'] for d in all_docs.values()])
        if bm25_scores.max() - bm25_scores.min() > epsilon:
            bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + epsilon)
        else:
            bm25_scores_norm = np.zeros_like(bm25_scores)
        
        # =====================================================
        # BƯỚC 5: TÍNH COMBINED SCORE
        # =====================================================
        combined_scores = self.alpha * vector_scores_norm + (1 - self.alpha) * bm25_scores_norm
        
        # Gán combined score cho mỗi doc
        for i, (doc_id, doc_data) in enumerate(all_docs.items()):
            doc_data['combined_score'] = combined_scores[i]
        
        # =====================================================
        # BƯỚC 6: SẮP XẾP VÀ TRẢ VỀ TOP_K
        # =====================================================
        sorted_docs = sorted(all_docs.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        
        # Chuyển về Document objects
        result_docs = []
        for doc_id, doc_data in sorted_docs[:top_k]:
            doc = Document(
                text=doc_data['text'],
                metadata=doc_data['metadata'],
                doc_id=doc_id
            )
            result_docs.append(doc)
        
        return result_docs
