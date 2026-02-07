# -*- coding: utf-8 -*-
"""
engine.py - RAG Engine chính cho Hệ thống Luật Việt Nam

Orchestrates all components:
- Embedding Model (BKAI Vietnamese)
- LLM (Google Gemini)
- Vector Store (Qdrant)
- Retrievers (BM25, Fusion)
- Query Enhancement (Phase 2)
"""

from typing import List, Optional, Dict, Any

from llama_index.core import Settings, Document

from src.embeddings import VietnameseEmbedding
from src.llm import GeminiLLM
from src.vector_store import QdrantStore
from src.retrievers import BM25Retriever, FusionRetriever, MetadataFilterParser, filter_documents, CrossEncoderReranker

# Phase 2: Query Enhancement imports
from src.query import (
    QueryEnhancer,
    QueryType,
    IterativeRetriever,
    RelevanceEvaluator,
)

from config import (
    SIMILARITY_TOP_K,
    SYSTEM_PROMPT,
    FUSION_ALPHA,
    USE_FUSION_RETRIEVAL,
    USE_METADATA_FILTERING,
    USE_RERANKING,
    RERANKER_MODEL,
    RERANK_TOP_K,
    # Phase 2 config
    USE_QUERY_ENHANCEMENT,
    USE_ITERATIVE_RETRIEVAL,
    USE_HYDE,
    MAX_RETRIEVAL_ITERATIONS,
)


class LegalRAGSystem:
    """
    Hệ thống RAG (Retrieval-Augmented Generation) cho Luật Việt Nam.
    
    Kết hợp:
    - Vector Search (semantic) để tìm kiếm theo ngữ nghĩa
    - BM25 (keyword) để tìm kiếm theo từ khóa
    - Fusion Retrieval: kết hợp cả hai
    - Query Enhancement: phân loại và phân tách câu hỏi (Phase 2)
    - Iterative Retrieval: nhiều vòng retrieval với gap analysis (Phase 2)
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
        
        # Khởi tạo Cross-Encoder Reranker
        self.reranker: Optional[CrossEncoderReranker] = None
        if USE_RERANKING:
            self.reranker = CrossEncoderReranker(model_name=RERANKER_MODEL)
        
        # Phase 2: Query Enhancement
        self.query_enhancer: Optional[QueryEnhancer] = None
        if USE_QUERY_ENHANCEMENT:
            # Dùng LLM nếu cần decomposition, ngược lại dùng rule-based
            self.query_enhancer = QueryEnhancer(
                llm=self.llm,
                use_llm_classification=False  # Rule-based classification (nhanh hơn)
            )
            print("[INFO] Query Enhancement đã bật (classification + decomposition)")
        
        # Phase 2: Relevance Evaluator
        self.relevance_evaluator = RelevanceEvaluator(llm=self.llm)
        
        # Log status
        if USE_FUSION_RETRIEVAL:
            print(f"[INFO] Fusion Retrieval đã bật (alpha={FUSION_ALPHA})")
        
        if USE_RERANKING and self.reranker and self.reranker.is_available():
            print(f"[INFO] Cross-Encoder Reranking đã bật (model={RERANKER_MODEL})")
        
        if USE_ITERATIVE_RETRIEVAL:
            print("[INFO] Iterative Retrieval đã bật")
        
        if USE_HYDE:
            print("[INFO] HyDE đã bật")
        
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
        Thực hiện truy vấn với Query Enhancement (Phase 2).
        
        Pipeline:
        1. Query Enhancement: Classification + Decomposition
        2. (Optional) HyDE: Generate hypothetical document
        3. Retrieval với adaptive strategy
        4. (Optional) Iterative Retrieval: Gap analysis + re-retrieve
        5. Metadata filtering + Reranking
        6. Generate answer với Legal CoT
        
        Args:
            question: Câu hỏi của người dùng
            
        Returns:
            Câu trả lời từ LLM
        """
        print(f"\n[QUERY] Câu hỏi: {question}")
        
        # ============================================================
        # PHASE 2: Query Enhancement
        # ============================================================
        query_analysis = None
        retrieval_top_k = SIMILARITY_TOP_K
        
        if USE_QUERY_ENHANCEMENT and self.query_enhancer is not None:
            query_analysis = self.query_enhancer.analyze(question)
            print(f"[INFO] Query Type: {query_analysis.query_type.value} (confidence: {query_analysis.confidence:.2f})")
            
            # Adaptive retrieval config
            retrieval_top_k = query_analysis.retrieval_config.get("top_k", SIMILARITY_TOP_K)
            
            # Log decomposition nếu có
            if query_analysis.is_complex and len(query_analysis.sub_queries) > 1:
                print(f"[INFO] Query phức tạp → {len(query_analysis.sub_queries)} sub-queries")
                for i, sq in enumerate(query_analysis.sub_queries, 1):
                    print(f"  {i}. {sq}")
        
        # Parse metadata filter từ câu hỏi
        metadata_filter = None
        if USE_METADATA_FILTERING:
            metadata_filter = MetadataFilterParser.parse(question)
            if not metadata_filter.is_empty():
                print(f"[INFO] Metadata Filter: {metadata_filter}")
        
        # ============================================================
        # RETRIEVAL
        # ============================================================
        print("[INFO] Đang tìm kiếm thông tin liên quan...")
        
        if USE_FUSION_RETRIEVAL and self.fusion_retriever is not None:
            print(f"[INFO] Fusion Retrieval: α={FUSION_ALPHA} (vector={FUSION_ALPHA:.0%}, BM25={1-FUSION_ALPHA:.0%})")
            
            # Determine fetch size
            need_more = (USE_RERANKING and self.reranker and self.reranker.is_available()) or \
                        (metadata_filter and not metadata_filter.is_empty())
            fetch_k = RERANK_TOP_K if need_more else retrieval_top_k
            
            # ============================================================
            # PHASE 2: HyDE (if enabled)
            # ============================================================
            search_query = question
            if USE_HYDE and self.query_enhancer is not None:
                hyde_doc = self.query_enhancer.generate_hyde_query(question)
                if hyde_doc:
                    print("[INFO] HyDE: Đã tạo hypothetical document")
                    # Combine original query with HyDE for hybrid search
                    search_query = f"{question}\n\n{hyde_doc[:500]}"
            
            # ============================================================
            # Retrieve (with sub-queries if complex)
            # ============================================================
            retrieved_docs = []
            
            if query_analysis and query_analysis.is_complex and len(query_analysis.sub_queries) > 1:
                # Retrieve cho từng sub-query và merge
                seen_doc_ids = set()
                sub_top_k = max(3, fetch_k // len(query_analysis.sub_queries))
                
                for sq in query_analysis.sub_queries:
                    sub_docs = self.fusion_retriever.retrieve(sq, top_k=sub_top_k)
                    for doc in sub_docs:
                        doc_id = getattr(doc, 'doc_id', None) or hash(doc.text)
                        if doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            retrieved_docs.append(doc)
                
                print(f"[INFO] Multi-query retrieval: {len(retrieved_docs)} unique documents")
            else:
                retrieved_docs = self.fusion_retriever.retrieve(search_query, top_k=fetch_k)
            
            # ============================================================
            # PHASE 2: Iterative Retrieval (if enabled)
            # ============================================================
            if USE_ITERATIVE_RETRIEVAL and query_analysis:
                should_iterate = query_analysis.retrieval_config.get("use_iterative", False)
                max_iters = query_analysis.retrieval_config.get("max_iterations", MAX_RETRIEVAL_ITERATIONS)
                
                if should_iterate and max_iters > 1:
                    # Create retriever function for iterator
                    def retrieve_fn(q: str, k: int) -> List[Document]:
                        return self.fusion_retriever.retrieve(q, top_k=k)
                    
                    from src.query import IterativeRetriever
                    iterative = IterativeRetriever(
                        retriever_fn=retrieve_fn,
                        llm=self.llm,
                        max_iterations=max_iters
                    )
                    
                    iter_result = iterative.retrieve(
                        question, 
                        top_k=retrieval_top_k,
                        initial_documents=retrieved_docs
                    )
                    retrieved_docs = iter_result.all_documents
                    
                    if iter_result.total_iterations > 1:
                        print(f"[INFO] Iterative Retrieval: {iter_result.total_iterations} iterations, "
                              f"sufficient={iter_result.is_sufficient}")
            
            # ============================================================
            # Metadata Filtering
            # ============================================================
            if metadata_filter and not metadata_filter.is_empty():
                retrieved_docs = filter_documents(retrieved_docs, metadata_filter)
                print(f"[INFO] Sau khi lọc metadata: {len(retrieved_docs)} documents")
            
            # ============================================================
            # Reranking
            # ============================================================
            if USE_RERANKING and self.reranker and self.reranker.is_available():
                retrieved_docs = self.reranker.rerank(question, retrieved_docs, top_k=retrieval_top_k)
            else:
                retrieved_docs = retrieved_docs[:retrieval_top_k]
            
            if not retrieved_docs:
                return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
            
            # ============================================================
            # Build Context
            # ============================================================
            context_parts = []
            max_context_chars = 8000
            current_chars = 0
            
            for i, doc in enumerate(retrieved_docs, 1):
                meta = doc.metadata
                source_info = f"[{meta.get('doc_name', 'N/A')} - {meta.get('article_id', 'N/A')}]"
                
                text = doc.text
                if "---\n" in text:
                    content = text.split("---\n", 1)[-1].strip()
                else:
                    content = text
                
                part_text = f"{source_info}\n{content}"
                if current_chars + len(part_text) > max_context_chars:
                    remaining = max_context_chars - current_chars - len(source_info) - 50
                    if remaining > 200:
                        content = content[:remaining] + "..."
                        part_text = f"{source_info}\n{content}"
                    else:
                        break
                
                context_parts.append(part_text)
                current_chars += len(part_text)
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Log summary
            print(f"[INFO] Legal CoT với {len(context_parts)} nguồn ({current_chars} chars)")
            
            # ============================================================
            # Generate Answer với Legal CoT (giữ nguyên prompt)
            # ============================================================
            full_prompt = f"""{SYSTEM_PROMPT}

## Context (Quy phạm pháp luật liên quan):

{context}

---
## Câu hỏi: {question}

Hãy phân tích và trả lời:"""
            
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
