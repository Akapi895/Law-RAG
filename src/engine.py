# -*- coding: utf-8 -*-
"""
engine.py - RAG Engine chính cho Hệ thống Luật Việt Nam

Orchestrates all components:
- Embedding Model (BKAI Vietnamese)
- LLM (Google Gemini)
- Vector Store (Qdrant)
- Retrievers (BM25, Fusion)
"""

from typing import List, Optional, Dict, Any

from llama_index.core import Settings, Document

from src.embeddings import VietnameseEmbedding
from src.llm import GeminiLLM
from src.vector_store import QdrantStore
from src.retrievers import BM25Retriever, FusionRetriever, MetadataFilterParser, filter_documents

from config import (
    SIMILARITY_TOP_K,
    SYSTEM_PROMPT,
    FUSION_ALPHA,
    USE_FUSION_RETRIEVAL,
    USE_METADATA_FILTERING
)


class LegalRAGSystem:
    """
    Hệ thống RAG (Retrieval-Augmented Generation) cho Luật Việt Nam.
    
    Kết hợp:
    - Vector Search (semantic) để tìm kiếm theo ngữ nghĩa
    - BM25 (keyword) để tìm kiếm theo từ khóa
    - Fusion Retrieval: kết hợp cả hai
    - LLM để sinh câu trả lời
    """
    
    def __init__(self):
        """Khởi tạo hệ thống RAG."""
        print("[INFO] Đang khởi tạo hệ thống RAG...")
        
        # Khởi tạo Embedding Model
        self._embedding = VietnameseEmbedding()
        self.embed_model = self._embedding.load()
        
        # Khởi tạo LLM
        self._llm_wrapper = GeminiLLM()
        self.llm = self._llm_wrapper.load()
        
        # Thiết lập mặc định cho LlamaIndex
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        
        # Khởi tạo Vector Store
        self.vector_store = QdrantStore()
        
        # Khởi tạo BM25 Retriever
        self.bm25_retriever = BM25Retriever()
        
        # Fusion Retriever (sẽ được tạo sau khi build index)
        self.fusion_retriever: Optional[FusionRetriever] = None
        
        # Query engine
        self.query_engine = None
        
        if USE_FUSION_RETRIEVAL:
            print(f"[INFO] Fusion Retrieval đã bật (alpha={FUSION_ALPHA})")
        
        print("[SUCCESS] Khởi tạo hệ thống RAG hoàn tất!")
    
    def build_index(self, documents: List[Document]) -> None:
        """
        Xây dựng Vector Index + BM25 Index.
        
        Args:
            documents: Danh sách LlamaIndex Documents
        """
        if not documents:
            raise ValueError("Danh sách documents trống!")
        
        # Build Vector Index
        self.vector_store.build_index(documents)
        
        # Build BM25 Index
        if USE_FUSION_RETRIEVAL:
            self.bm25_retriever.build_index(documents)
            
            # Tạo Fusion Retriever
            self.fusion_retriever = FusionRetriever(
                vector_index=self.vector_store.index,
                bm25_retriever=self.bm25_retriever,
                alpha=FUSION_ALPHA
            )
    
    def sync_index(self, documents: List[Document]) -> dict:
        """
        Đồng bộ hóa index với documents mới.
        
        Args:
            documents: Danh sách LlamaIndex Documents
            
        Returns:
            Dict với thống kê
        """
        print("[INFO] Đang đồng bộ hóa Vector Index...")
        self.build_index(documents)
        
        return {
            "added": len(documents),
            "removed": 0,
            "unchanged": 0
        }
    
    def get_query_engine(self):
        """Tạo Query Engine."""
        if self.vector_store.index is None:
            raise RuntimeError("Index chưa được xây dựng!")
        
        print(f"[INFO] Đang tạo Query Engine (top_k={SIMILARITY_TOP_K})...")
        
        if USE_FUSION_RETRIEVAL:
            print(f"[INFO] Sử dụng Fusion Retrieval (alpha={FUSION_ALPHA})")
        
        self.query_engine = self.vector_store.index.as_query_engine(
            similarity_top_k=SIMILARITY_TOP_K,
            system_prompt=SYSTEM_PROMPT
        )
        
        print("[SUCCESS] Query Engine đã sẵn sàng!")
        return self.query_engine
    
    def query(self, question: str) -> str:
        """
        Thực hiện truy vấn.
        
        Args:
            question: Câu hỏi của người dùng
            
        Returns:
            Câu trả lời từ LLM
        """
        print(f"\n[QUERY] Câu hỏi: {question}")
        print("[INFO] Đang tìm kiếm thông tin liên quan...")
        
        # Parse metadata filter từ câu hỏi
        metadata_filter = None
        if USE_METADATA_FILTERING:
            metadata_filter = MetadataFilterParser.parse(question)
            if not metadata_filter.is_empty():
                print(f"[INFO] Metadata Filter: {metadata_filter}")
        
        # Sử dụng Fusion Retrieval nếu được bật
        if USE_FUSION_RETRIEVAL and self.fusion_retriever is not None:
            print(f"[INFO] Fusion Retrieval: α={FUSION_ALPHA} (vector={FUSION_ALPHA:.0%}, BM25={1-FUSION_ALPHA:.0%})")
            
            # Lấy documents bằng Fusion Retrieval
            # Lấy nhiều hơn nếu có metadata filter để còn lọc
            fetch_k = SIMILARITY_TOP_K * 3 if metadata_filter and not metadata_filter.is_empty() else SIMILARITY_TOP_K
            retrieved_docs = self.fusion_retriever.retrieve(question, top_k=fetch_k)
            
            # Áp dụng metadata filter
            if metadata_filter and not metadata_filter.is_empty():
                retrieved_docs = filter_documents(retrieved_docs, metadata_filter)
                print(f"[INFO] Sau khi lọc: {len(retrieved_docs)} documents")
                # Lấy top_k sau khi filter
                retrieved_docs = retrieved_docs[:SIMILARITY_TOP_K]
            
            if not retrieved_docs:
                return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
            
            # Tạo context
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                meta = doc.metadata
                source_info = f"[Nguồn {i}: {meta.get('doc_name', 'N/A')} - {meta.get('article_id', 'N/A')}]"
                context_parts.append(f"{source_info}\n{doc.text}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Tạo prompt
            full_prompt = f"""{SYSTEM_PROMPT}

Ngữ cảnh (Context):
{context}

Câu hỏi: {question}

Câu trả lời:"""
            
            # Gọi LLM
            response = self.llm.complete(full_prompt)
            return str(response)
        
        # Fallback: Query Engine mặc định
        else:
            if self.query_engine is None:
                self.get_query_engine()
            
            response = self.query_engine.query(question)
            return str(response)


# =====================================================
# TEST MODULE
# =====================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TEST: LegalRAGSystem")
    print("=" * 60)
    
    from src.data import generate_mock_data, load_and_process_data
    
    generate_mock_data()
    documents = load_and_process_data()
    
    rag_system = LegalRAGSystem()
    rag_system.build_index(documents)
    
    test_question = "Điều kiện để nhà thầu được coi là hợp lệ là gì?"
    answer = rag_system.query(test_question)
    
    print("\n" + "=" * 60)
    print("CÂU TRẢ LỜI:")
    print("=" * 60)
    print(answer)
