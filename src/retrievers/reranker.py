# -*- coding: utf-8 -*-
"""
reranker.py - Cross-Encoder Reranking cho Vietnamese Legal RAG

Sử dụng Cross-Encoder để re-score và xếp hạng lại kết quả retrieval.
Cross-Encoder đánh giá cặp (query, document) chính xác hơn bi-encoder.
"""

from typing import List, Optional
from llama_index.core import Document

# Import cross-encoder
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("[WARNING] sentence-transformers không được cài đặt. Reranking sẽ bị tắt.")


class CrossEncoderReranker:
    """
    Reranker sử dụng Cross-Encoder model.
    
    Cross-Encoder nhận cặp (query, document) và trả về relevance score.
    Chính xác hơn bi-encoder nhưng chậm hơn → chỉ dùng để rerank top-k.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Khởi tạo Cross-Encoder Reranker.
        
        Args:
            model_name: Tên model cross-encoder từ HuggingFace
                - "cross-encoder/ms-marco-MiniLM-L-6-v2" (fast, multilingual)
                - "BAAI/bge-reranker-base" (tốt cho tiếng Việt)
        """
        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load cross-encoder model."""
        if not CROSS_ENCODER_AVAILABLE:
            print("[WARNING] Cross-Encoder không khả dụng. Bỏ qua reranking.")
            return
        
        print(f"[INFO] Đang tải Cross-Encoder model: {self.model_name}")
        try:
            self.model = CrossEncoder(self.model_name)
            print(f"[SUCCESS] Đã tải Cross-Encoder model!")
        except Exception as e:
            print(f"[ERROR] Không thể tải Cross-Encoder model: {e}")
            self.model = None
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Rerank documents theo relevance score từ Cross-Encoder.
        
        Args:
            query: Câu hỏi của user
            documents: Danh sách documents cần rerank
            top_k: Số lượng documents trả về (None = trả về tất cả)
            
        Returns:
            Danh sách documents đã được rerank theo relevance score
        """
        if not documents:
            return []
        
        if self.model is None:
            print("[WARNING] Cross-Encoder không khả dụng, trả về documents gốc.")
            return documents[:top_k] if top_k else documents
        
        # Tạo pairs (query, document_text) cho cross-encoder
        pairs = [(query, doc.text) for doc in documents]
        
        # Tính relevance scores
        scores = self.model.predict(pairs)
        
        # Kết hợp documents với scores
        doc_scores = list(zip(documents, scores))
        
        # Sắp xếp theo score giảm dần
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Lấy top_k documents
        if top_k:
            doc_scores = doc_scores[:top_k]
        
        # Trả về documents đã rerank
        reranked_docs = [doc for doc, score in doc_scores]
        
        # Log scores cho debugging
        print(f"[RERANK] Reranked {len(documents)} → {len(reranked_docs)} documents")
        for i, (doc, score) in enumerate(doc_scores[:3], 1):
            article_id = doc.metadata.get('article_id', 'N/A')
            doc_name = doc.metadata.get('doc_name', 'N/A')
            print(f"  #{i}: {doc_name} - {article_id} (score: {score:.4f})")
        
        return reranked_docs
    
    def is_available(self) -> bool:
        """Kiểm tra Cross-Encoder có sẵn sàng không."""
        return self.model is not None


# =====================================================
# TEST MODULE
# =====================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TEST: Cross-Encoder Reranker")
    print("=" * 50)
    
    reranker = CrossEncoderReranker()
    
    if reranker.is_available():
        # Tạo test documents
        test_docs = [
            Document(text="Điều 5 quy định về tư cách hợp lệ của nhà thầu", 
                    metadata={"article_id": "Điều 5", "doc_name": "Luật Đấu thầu"}),
            Document(text="Điều 16 quy định về các hành vi bị cấm trong đấu thầu",
                    metadata={"article_id": "Điều 16", "doc_name": "Luật Đấu thầu"}),
            Document(text="Nghĩa vụ của nhà thầu thi công xây dựng công trình",
                    metadata={"article_id": "Điều 113", "doc_name": "Luật Xây dựng"}),
        ]
        
        query = "Nhà thầu cần điều kiện gì để hợp lệ?"
        
        reranked = reranker.rerank(query, test_docs, top_k=2)
        
        print("\n" + "=" * 50)
        print(f"Query: {query}")
        print("=" * 50)
        print(f"Reranked documents: {len(reranked)}")
    else:
        print("Cross-Encoder không khả dụng!")
