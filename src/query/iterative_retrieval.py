# -*- coding: utf-8 -*-
"""
iterative_retrieval.py - Iterative Retrieval Module cho Vietnamese Legal RAG

Cải tiến Phase 2:
- Nhiều vòng retrieval để bổ sung context còn thiếu
- Gap Analysis để xác định thông tin cần bổ sung
- Relevance Evaluation để đánh giá chất lượng retrieval

Tham khảo: rag_improvement_strategy.md
"""

from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from llama_index.core import Document


@dataclass
class RetrievalIteration:
    """Kết quả của một vòng retrieval."""
    iteration: int
    query: str
    documents: List[Document]
    gap_analysis: Optional[str] = None
    is_sufficient: bool = False


@dataclass 
class IterativeRetrievalResult:
    """Kết quả tổng hợp của iterative retrieval."""
    original_query: str
    all_documents: List[Document]
    iterations: List[RetrievalIteration] = field(default_factory=list)
    total_iterations: int = 0
    is_sufficient: bool = False
    final_gap: Optional[str] = None


class GapAnalyzer:
    """
    Phân tích thiếu sót trong context hiện có.
    
    Xác định xem context đã đủ để trả lời câu hỏi chưa,
    nếu chưa thì cần bổ sung thông tin gì.
    """
    
    GAP_ANALYSIS_PROMPT = """Phân tích xem context hiện có đã đủ để trả lời câu hỏi pháp lý chưa.

Câu hỏi: {query}

Context hiện có:
{context_summary}

Hướng dẫn phân tích:
1. Xác định các KHÍA CẠNH PHÁP LÝ cần trả lời trong câu hỏi
2. Kiểm tra từng khía cạnh đã có đủ thông tin chưa
3. Nếu thiếu, nêu cụ thể cần tra cứu thêm gì

Trả lời theo format:
- Nếu ĐỦ thông tin: bắt đầu bằng "SUFFICIENT:" rồi giải thích ngắn gọn
- Nếu THIẾU thông tin: bắt đầu bằng "MISSING:" rồi liệt kê cần tra cứu thêm gì

Phân tích:"""
    
    def __init__(self, llm=None):
        """
        Khởi tạo Gap Analyzer.
        
        Args:
            llm: LLM instance
        """
        self.llm = llm
    
    def analyze(self, query: str, documents: List[Document]) -> tuple[bool, str]:
        """
        Phân tích gap trong context.
        
        Args:
            query: Câu hỏi gốc
            documents: Các documents đã retrieve
            
        Returns:
            Tuple[is_sufficient, gap_description]
        """
        if not documents:
            return False, "Chưa có tài liệu nào được tìm thấy. Cần tìm kiếm tài liệu pháp luật liên quan."
        
        # Nếu không có LLM, dùng heuristics đơn giản
        if self.llm is None:
            return self._analyze_with_heuristics(query, documents)
        
        return self._analyze_with_llm(query, documents)
    
    def _analyze_with_heuristics(self, query: str, documents: List[Document]) -> tuple[bool, str]:
        """Phân tích bằng heuristics đơn giản."""
        # Đủ nếu có >= 3 documents với độ dài hợp lý
        good_docs = [d for d in documents if len(d.text) > 100]
        
        if len(good_docs) >= 3:
            return True, "Đủ tài liệu để trả lời (heuristic check)"
        
        return False, f"Chỉ tìm được {len(good_docs)} tài liệu có nội dung. Cần tìm thêm tài liệu liên quan."
    
    def _analyze_with_llm(self, query: str, documents: List[Document]) -> tuple[bool, str]:
        """Phân tích bằng LLM."""
        # Tạo context summary
        context_parts = []
        for i, doc in enumerate(documents[:5], 1):  # Limit to 5 docs for summary
            meta = doc.metadata
            source = f"{meta.get('doc_name', 'N/A')} - {meta.get('article_id', 'N/A')}"
            # Lấy 200 ký tự đầu của mỗi doc
            preview = doc.text[:200].replace('\n', ' ')
            context_parts.append(f"{i}. [{source}] {preview}...")
        
        context_summary = '\n'.join(context_parts)
        
        prompt = self.GAP_ANALYSIS_PROMPT.format(
            query=query,
            context_summary=context_summary
        )
        
        try:
            response = str(self.llm.complete(prompt)).strip()
            
            # Parse response
            if response.upper().startswith("SUFFICIENT"):
                return True, response
            else:
                # Extract the gap description
                gap = response.replace("MISSING:", "").strip()
                return False, gap
                
        except Exception as e:
            print(f"[WARNING] Gap analysis with LLM failed: {e}")
            return self._analyze_with_heuristics(query, documents)
    
    def create_refined_query(self, original_query: str, gap: str) -> str:
        """
        Tạo query mới để bổ sung thông tin thiếu.
        
        Args:
            original_query: Câu hỏi gốc
            gap: Mô tả thông tin còn thiếu
            
        Returns:
            Query mới để tìm kiếm bổ sung
        """
        # Kết hợp câu hỏi gốc với gap
        return f"{original_query} - Bổ sung: {gap}"


class IterativeRetriever:
    """
    Iterative Retrieval với Gap Analysis.
    
    Thực hiện nhiều vòng retrieval:
    1. Retrieve ban đầu
    2. Phân tích gap
    3. Nếu thiếu → refine query → retrieve thêm
    4. Lặp lại cho đến khi đủ hoặc hết iterations
    """
    
    def __init__(
        self,
        retriever_fn: Callable[[str, int], List[Document]],
        llm=None,
        max_iterations: int = 3
    ):
        """
        Khởi tạo Iterative Retriever.
        
        Args:
            retriever_fn: Function để retrieve documents, nhận (query, top_k)
            llm: LLM instance cho gap analysis
            max_iterations: Số vòng lặp tối đa
        """
        self.retriever_fn = retriever_fn
        self.gap_analyzer = GapAnalyzer(llm=llm)
        self.max_iterations = max_iterations
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        initial_documents: Optional[List[Document]] = None
    ) -> IterativeRetrievalResult:
        """
        Thực hiện iterative retrieval.
        
        Args:
            query: Câu hỏi gốc
            top_k: Số documents mỗi lần retrieve
            initial_documents: Documents đã có sẵn (nếu có)
            
        Returns:
            IterativeRetrievalResult với tất cả documents thu thập được
        """
        result = IterativeRetrievalResult(
            original_query=query,
            all_documents=[],
            iterations=[],
            total_iterations=0
        )
        
        # Thêm initial documents nếu có
        if initial_documents:
            result.all_documents.extend(initial_documents)
        
        current_query = query
        seen_doc_ids = set()
        
        # Đánh dấu doc_id của initial documents
        for doc in result.all_documents:
            doc_id = getattr(doc, 'doc_id', None) or hash(doc.text)
            seen_doc_ids.add(doc_id)
        
        for iteration in range(self.max_iterations):
            # Step 1: Retrieve
            new_docs = self.retriever_fn(current_query, top_k)
            
            # Lọc bỏ documents đã có
            unique_new_docs = []
            for doc in new_docs:
                doc_id = getattr(doc, 'doc_id', None) or hash(doc.text)
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    unique_new_docs.append(doc)
            
            # Thêm vào kết quả
            result.all_documents.extend(unique_new_docs)
            
            # Step 2: Gap Analysis
            is_sufficient, gap = self.gap_analyzer.analyze(query, result.all_documents)
            
            # Lưu iteration result
            iteration_result = RetrievalIteration(
                iteration=iteration + 1,
                query=current_query,
                documents=unique_new_docs,
                gap_analysis=gap,
                is_sufficient=is_sufficient
            )
            result.iterations.append(iteration_result)
            result.total_iterations = iteration + 1
            
            # Log progress
            print(f"[ITERATIVE] Iteration {iteration + 1}: +{len(unique_new_docs)} docs, "
                  f"total={len(result.all_documents)}, sufficient={is_sufficient}")
            
            # Step 3: Check if sufficient
            if is_sufficient:
                result.is_sufficient = True
                break
            
            # Step 4: Refine query for next iteration
            if iteration < self.max_iterations - 1:
                current_query = self.gap_analyzer.create_refined_query(query, gap)
                result.final_gap = gap
        
        return result


class RelevanceEvaluator:
    """
    Đánh giá mức độ liên quan của documents với query.
    
    Dùng để:
    - Lọc bỏ documents không liên quan
    - Scoring documents cho ranking
    """
    
    RELEVANCE_PROMPT = """Đánh giá mức độ liên quan của đoạn văn bản pháp luật sau với câu hỏi:

Câu hỏi: {query}

Văn bản:
{document}

Tiêu chí đánh giá:
- HIGH: Trực tiếp trả lời câu hỏi hoặc chứa quy phạm pháp luật áp dụng
- MEDIUM: Liên quan gián tiếp, cung cấp context bổ sung hoặc định nghĩa
- LOW: Không liên quan hoặc off-topic

Chỉ trả lời một từ: HIGH, MEDIUM, hoặc LOW"""
    
    def __init__(self, llm=None):
        """
        Khởi tạo Relevance Evaluator.
        
        Args:
            llm: LLM instance
        """
        self.llm = llm
    
    def evaluate(self, query: str, document: Document) -> str:
        """
        Đánh giá relevance của một document.
        
        Args:
            query: Câu hỏi
            document: Document cần đánh giá
            
        Returns:
            "HIGH", "MEDIUM", hoặc "LOW"
        """
        if self.llm is None:
            return self._evaluate_with_heuristics(query, document)
        
        return self._evaluate_with_llm(query, document)
    
    def _evaluate_with_heuristics(self, query: str, document: Document) -> str:
        """Đánh giá bằng heuristics."""
        query_words = set(query.lower().split())
        doc_text = document.text.lower()
        
        # Đếm số từ trong query xuất hiện trong document
        matches = sum(1 for word in query_words if word in doc_text)
        match_ratio = matches / len(query_words) if query_words else 0
        
        if match_ratio > 0.5:
            return "HIGH"
        elif match_ratio > 0.2:
            return "MEDIUM"
        return "LOW"
    
    def _evaluate_with_llm(self, query: str, document: Document) -> str:
        """Đánh giá bằng LLM."""
        # Truncate document để tránh vượt token limit
        doc_text = document.text[:1000]
        
        prompt = self.RELEVANCE_PROMPT.format(
            query=query,
            document=doc_text
        )
        
        try:
            response = str(self.llm.complete(prompt)).strip().upper()
            
            if "HIGH" in response:
                return "HIGH"
            elif "MEDIUM" in response:
                return "MEDIUM"
            return "LOW"
            
        except Exception as e:
            print(f"[WARNING] Relevance evaluation failed: {e}")
            return self._evaluate_with_heuristics(query, document)
    
    def filter_documents(
        self, 
        query: str, 
        documents: List[Document],
        min_relevance: str = "MEDIUM"
    ) -> List[Document]:
        """
        Lọc documents theo ngưỡng relevance.
        
        Args:
            query: Câu hỏi
            documents: Danh sách documents
            min_relevance: Ngưỡng tối thiểu ("HIGH", "MEDIUM", "LOW")
            
        Returns:
            Danh sách documents đạt ngưỡng
        """
        relevance_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        min_score = relevance_order.get(min_relevance, 2)
        
        filtered = []
        for doc in documents:
            relevance = self.evaluate(query, doc)
            score = relevance_order.get(relevance, 0)
            
            if score >= min_score:
                filtered.append(doc)
        
        return filtered


# =====================================================
# TEST MODULE
# =====================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Iterative Retrieval Module")
    print("=" * 60)
    
    # Mock retriever function
    def mock_retriever(query: str, top_k: int) -> List[Document]:
        """Mock retriever for testing."""
        return [
            Document(
                text="Điều 5: Nhà thầu có tư cách hợp lệ khi đáp ứng các điều kiện...",
                metadata={"doc_name": "Luật Đấu thầu", "article_id": "Điều 5"},
                doc_id="D5"
            ),
            Document(
                text="Điều 16: Các hành vi bị cấm trong đấu thầu...",
                metadata={"doc_name": "Luật Đấu thầu", "article_id": "Điều 16"},
                doc_id="D16"
            ),
        ]
    
    # Test Gap Analyzer (rule-based)
    gap_analyzer = GapAnalyzer()
    
    test_docs = mock_retriever("test", 5)
    is_sufficient, gap = gap_analyzer.analyze(
        "Điều kiện nhà thầu hợp lệ?",
        test_docs
    )
    
    print(f"\nGap Analysis:")
    print(f"  Sufficient: {is_sufficient}")
    print(f"  Gap: {gap}")
    
    # Test Iterative Retriever
    iterative = IterativeRetriever(
        retriever_fn=mock_retriever,
        max_iterations=2
    )
    
    result = iterative.retrieve("Điều kiện nhà thầu hợp lệ?", top_k=3)
    
    print(f"\nIterative Retrieval:")
    print(f"  Total iterations: {result.total_iterations}")
    print(f"  Total documents: {len(result.all_documents)}")
    print(f"  Is sufficient: {result.is_sufficient}")
    
    # Test Relevance Evaluator (rule-based)
    evaluator = RelevanceEvaluator()
    
    for doc in test_docs:
        relevance = evaluator.evaluate("Điều kiện nhà thầu hợp lệ?", doc)
        print(f"  {doc.metadata['article_id']}: {relevance}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED!")
    print("=" * 60)
