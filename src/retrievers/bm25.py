# -*- coding: utf-8 -*-
"""
bm25.py - BM25 Keyword Retriever

BM25 là thuật toán ranking dựa trên keyword matching,
giúp tìm documents chứa từ khóa chính xác.
"""

from typing import List, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from llama_index.core import Document


class BM25Retriever:
    """
    BM25 Keyword-based Retriever.
    
    Sử dụng thuật toán BM25 để tìm kiếm documents
    dựa trên keyword matching (khác với semantic search).
    """
    
    def __init__(self):
        """Khởi tạo BM25 Retriever."""
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[Document] = []
    
    def build_index(self, documents: List[Document]):
        """
        Xây dựng BM25 index từ documents.
        
        Args:
            documents: Danh sách LlamaIndex Documents
        """
        if not documents:
            print("[WARNING] Không có documents để build BM25 index")
            return
        
        print("[INFO] Đang xây dựng BM25 index cho keyword search...")
        
        # Lưu documents gốc
        self._documents = documents
        
        # Tokenize documents (chia theo khoảng trắng)
        tokenized_docs = [doc.text.split() for doc in documents]
        
        # Tạo BM25 index
        self._bm25 = BM25Okapi(tokenized_docs)
        
        print(f"[SUCCESS] Đã xây dựng BM25 index với {len(documents)} documents!")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Tìm kiếm documents bằng BM25.
        
        Args:
            query: Câu truy vấn
            top_k: Số lượng kết quả trả về
            
        Returns:
            List tuples (document, score)
        """
        if self._bm25 is None or not self._documents:
            return []
        
        # Tokenize query
        query_tokens = query.split()
        
        # Lấy BM25 scores
        scores = self._bm25.get_scores(query_tokens)
        
        # Sắp xếp và lấy top_k
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in sorted_indices:
            if scores[idx] > 0:  # Chỉ lấy kết quả có score > 0
                results.append((self._documents[idx], float(scores[idx])))
        
        return results
    
    @property
    def is_ready(self) -> bool:
        """Kiểm tra BM25 index đã được build chưa."""
        return self._bm25 is not None and len(self._documents) > 0
    
    @property
    def documents(self) -> List[Document]:
        """Trả về danh sách documents."""
        return self._documents
