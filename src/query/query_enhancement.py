# -*- coding: utf-8 -*-
"""
query_enhancement.py - Query Enhancement Module cho Vietnamese Legal RAG

Cải tiến Phase 2:
1. Query Classification - Phân loại câu hỏi pháp lý
2. Query Decomposition - Phân tách câu hỏi phức tạp
3. HyDE - Hypothetical Document Embedding
4. Adaptive Retrieval Strategy

Tham khảo: rag_improvement_strategy.md
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class QueryType(Enum):
    """Phân loại câu hỏi pháp lý theo mục đích."""
    FACTUAL = "factual"           # Hỏi về nội dung cụ thể điều luật
    COMPARATIVE = "comparative"   # So sánh, đối chiếu nhiều quy định
    PROCEDURAL = "procedural"     # Hỏi về quy trình, thủ tục
    ANALYTICAL = "analytical"     # Phân tích, suy luận, hậu quả pháp lý


@dataclass
class QueryAnalysis:
    """Kết quả phân tích câu hỏi."""
    original_query: str
    query_type: QueryType
    sub_queries: List[str] = field(default_factory=list)
    is_complex: bool = False
    confidence: float = 0.0
    retrieval_config: Dict = field(default_factory=dict)


@dataclass
class RetrievalStrategy:
    """Chiến lược retrieval theo loại câu hỏi."""
    top_k: int = 5
    use_fusion: bool = True
    use_reranking: bool = True
    use_hyde: bool = False
    use_iterative: bool = False
    max_iterations: int = 1


# =====================================================
# CẤU HÌNH RETRIEVAL STRATEGY THEO QUERY TYPE
# =====================================================
RETRIEVAL_STRATEGIES: Dict[QueryType, RetrievalStrategy] = {
    QueryType.FACTUAL: RetrievalStrategy(
        top_k=3,
        use_fusion=True,
        use_reranking=True,
        use_hyde=False,
        use_iterative=False,
        max_iterations=1
    ),
    QueryType.COMPARATIVE: RetrievalStrategy(
        top_k=6,
        use_fusion=True,
        use_reranking=True,
        use_hyde=True,
        use_iterative=True,
        max_iterations=2
    ),
    QueryType.PROCEDURAL: RetrievalStrategy(
        top_k=5,
        use_fusion=True,
        use_reranking=True,
        use_hyde=False,
        use_iterative=True,
        max_iterations=2
    ),
    QueryType.ANALYTICAL: RetrievalStrategy(
        top_k=8,
        use_fusion=True,
        use_reranking=True,
        use_hyde=True,
        use_iterative=True,
        max_iterations=3
    ),
}


class QueryClassifier:
    """
    Phân loại câu hỏi pháp lý để chọn chiến lược retrieval phù hợp.
    
    Hỗ trợ 2 mode:
    - rule_based: Dựa trên pattern matching (nhanh, không cần LLM)
    - llm_based: Dựa trên LLM classification (chính xác hơn)
    """
    
    # Patterns cho rule-based classification
    PATTERNS = {
        QueryType.FACTUAL: [
            r'điều\s+\d+.*quy định',      # "Điều 5 quy định gì"
            r'nội dung.*điều',             # "nội dung điều X"
            r'theo.*điều\s+\d+',           # "theo Điều 5"
            r'quy định (về|tại)',          # "quy định về/tại"
            r'là gì\??$',                  # "... là gì?"
        ],
        QueryType.COMPARATIVE: [
            r'so sánh',                    # "so sánh"
            r'khác (nhau|biệt)',           # "khác nhau/biệt"
            r'giống (nhau)?',              # "giống nhau"
            r'hay là',                     # "A hay là B"
            r'và.*khác.*như thế nào',      # "A và B khác nhau như thế nào"
        ],
        QueryType.PROCEDURAL: [
            r'quy trình',                  # "quy trình"
            r'thủ tục',                    # "thủ tục"
            r'các bước',                   # "các bước"
            r'trình tự',                   # "trình tự"
            r'làm thế nào để',             # "làm thế nào để"
            r'cách thức',                  # "cách thức"
        ],
        QueryType.ANALYTICAL: [
            r'nếu.*thì',                   # "nếu X thì Y"
            r'hậu quả',                    # "hậu quả"
            r'vi phạm',                    # "vi phạm"
            r'xử lý',                      # "xử lý"
            r'chế tài',                    # "chế tài"
            r'trường hợp',                 # "trường hợp"
            r'khi nào',                    # "khi nào"
        ],
    }
    
    # Prompt cho LLM classification
    CLASSIFICATION_PROMPT = """Phân loại câu hỏi pháp lý sau vào MỘT trong các loại:

- factual: Hỏi về nội dung cụ thể của điều luật (VD: "Điều 5 quy định gì?")
- comparative: So sánh, đối chiếu nhiều quy định (VD: "So sánh quyền nhà thầu trong 2 luật")
- procedural: Hỏi về quy trình, thủ tục (VD: "Quy trình đấu thầu gồm những bước nào?")
- analytical: Phân tích, suy luận, hậu quả pháp lý (VD: "Nếu vi phạm Điều 16 sẽ bị xử lý thế nào?")

Câu hỏi: {query}

Chỉ trả lời MỘT từ: factual, comparative, procedural, hoặc analytical"""
    
    def __init__(self, llm=None, mode: str = "rule_based"):
        """
        Khởi tạo Query Classifier.
        
        Args:
            llm: LLM instance (cần thiết cho mode llm_based)
            mode: "rule_based" hoặc "llm_based"
        """
        self.llm = llm
        self.mode = mode
    
    def classify(self, query: str) -> Tuple[QueryType, float]:
        """
        Phân loại câu hỏi.
        
        Args:
            query: Câu hỏi cần phân loại
            
        Returns:
            Tuple[QueryType, confidence_score]
        """
        if self.mode == "llm_based" and self.llm is not None:
            return self._classify_with_llm(query)
        return self._classify_with_rules(query)
    
    def _classify_with_rules(self, query: str) -> Tuple[QueryType, float]:
        """Phân loại dựa trên pattern matching."""
        query_lower = query.lower()
        
        # Đếm số lượng pattern match cho mỗi loại
        scores = {}
        for query_type, patterns in self.PATTERNS.items():
            matches = sum(1 for p in patterns if re.search(p, query_lower))
            scores[query_type] = matches
        
        # Tìm loại có nhiều match nhất
        if max(scores.values()) == 0:
            # Default to FACTUAL nếu không match pattern nào
            return QueryType.FACTUAL, 0.5
        
        best_type = max(scores, key=scores.get)
        total_patterns = sum(len(p) for p in self.PATTERNS.values())
        confidence = min(scores[best_type] / 3, 1.0)  # Normalize confidence
        
        return best_type, confidence
    
    def _classify_with_llm(self, query: str) -> Tuple[QueryType, float]:
        """Phân loại dựa trên LLM."""
        prompt = self.CLASSIFICATION_PROMPT.format(query=query)
        
        try:
            response = str(self.llm.complete(prompt)).strip().lower()
            
            # Map response to QueryType
            type_map = {
                "factual": QueryType.FACTUAL,
                "comparative": QueryType.COMPARATIVE,
                "procedural": QueryType.PROCEDURAL,
                "analytical": QueryType.ANALYTICAL,
            }
            
            for key, qtype in type_map.items():
                if key in response:
                    return qtype, 0.9  # High confidence for LLM
            
            # Fallback nếu LLM trả về không hợp lệ
            return self._classify_with_rules(query)
            
        except Exception as e:
            print(f"[WARNING] LLM classification failed: {e}")
            return self._classify_with_rules(query)
    
    def get_strategy(self, query_type: QueryType) -> RetrievalStrategy:
        """Lấy retrieval strategy cho loại câu hỏi."""
        return RETRIEVAL_STRATEGIES.get(query_type, RETRIEVAL_STRATEGIES[QueryType.FACTUAL])


class QueryDecomposer:
    """
    Phân tách câu hỏi pháp lý phức tạp thành các sub-queries.
    
    Ví dụ:
    "Nhà thầu cần điều kiện gì để tham gia đấu thầu và có nghĩa vụ gì khi thi công?"
    →
    1. "Điều kiện để nhà thầu có tư cách hợp lệ tham gia đấu thầu?"
    2. "Nghĩa vụ của nhà thầu khi thi công xây dựng?"
    """
    
    # Prompt cho phân tách câu hỏi
    DECOMPOSITION_PROMPT = """Phân tách câu hỏi pháp lý sau thành các câu hỏi con độc lập.

Câu hỏi gốc: {question}

Hướng dẫn:
1. Tách theo CHỦ THỂ pháp luật (nhà thầu, chủ đầu tư, cơ quan...)
2. Tách theo LOẠI QUY PHẠM (điều kiện, quyền, nghĩa vụ, thủ tục, chế tài)
3. Tách theo LĨNH VỰC nếu câu hỏi liên quan nhiều luật
4. Mỗi câu hỏi con phải có thể trả lời bằng 1-2 điều luật
5. Nếu câu hỏi đã đơn giản, trả về chính câu hỏi đó

Liệt kê các câu hỏi con (mỗi dòng một câu, bắt đầu bằng "-"):"""
    
    # Patterns để detect câu hỏi phức tạp
    COMPLEXITY_PATTERNS = [
        r'\bvà\b.*\?',           # "A và B?"
        r'\bnhưng\b',            # "nhưng"
        r'\bhay\b',              # "hay"
        r'\bcũng như\b',         # "cũng như"
        r'\bngoài ra\b',         # "ngoài ra"
        r',\s*\w+.*\?',          # "A, B?"
        r'\?\s*.*\?',            # Nhiều dấu ?
    ]
    
    def __init__(self, llm=None):
        """
        Khởi tạo Query Decomposer.
        
        Args:
            llm: LLM instance để phân tách câu hỏi
        """
        self.llm = llm
    
    def is_complex(self, query: str) -> bool:
        """
        Kiểm tra câu hỏi có phức tạp không.
        
        Args:
            query: Câu hỏi cần kiểm tra
            
        Returns:
            True nếu câu hỏi phức tạp cần phân tách
        """
        query_lower = query.lower()
        
        # Check patterns
        for pattern in self.COMPLEXITY_PATTERNS:
            if re.search(pattern, query_lower):
                return True
        
        # Check độ dài câu hỏi (> 100 ký tự có thể là phức tạp)
        if len(query) > 100:
            return True
        
        return False
    
    def decompose(self, query: str) -> List[str]:
        """
        Phân tách câu hỏi thành các sub-queries.
        
        Args:
            query: Câu hỏi gốc
            
        Returns:
            List các sub-queries (hoặc [query] nếu đơn giản)
        """
        # Kiểm tra độ phức tạp
        if not self.is_complex(query):
            return [query]
        
        # Nếu có LLM, dùng LLM để phân tách
        if self.llm is not None:
            return self._decompose_with_llm(query)
        
        # Fallback: rule-based decomposition
        return self._decompose_with_rules(query)
    
    def _decompose_with_llm(self, query: str) -> List[str]:
        """Phân tách bằng LLM."""
        prompt = self.DECOMPOSITION_PROMPT.format(question=query)
        
        try:
            response = str(self.llm.complete(prompt))
            
            # Parse response
            sub_queries = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Loại bỏ marker đầu dòng
                if line.startswith('-'):
                    line = line[1:].strip()
                elif line.startswith('•'):
                    line = line[1:].strip()
                elif re.match(r'^\d+[\.\)]\s*', line):
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)
                
                if line and len(line) > 10:  # Bỏ qua dòng quá ngắn
                    sub_queries.append(line)
            
            if sub_queries:
                return sub_queries
            
            # Fallback nếu parse thất bại
            return [query]
            
        except Exception as e:
            print(f"[WARNING] LLM decomposition failed: {e}")
            return self._decompose_with_rules(query)
    
    def _decompose_with_rules(self, query: str) -> List[str]:
        """Phân tách bằng rules đơn giản."""
        sub_queries = []
        
        # Tách theo " và " hoặc ", "
        parts = re.split(r'\s+và\s+|\s*,\s*', query)
        
        for part in parts:
            part = part.strip()
            if len(part) > 10:
                # Đảm bảo kết thúc bằng dấu ?
                if not part.endswith('?'):
                    part = part + '?'
                sub_queries.append(part)
        
        if len(sub_queries) <= 1:
            return [query]
        
        return sub_queries


class HyDEGenerator:
    """
    HyDE - Hypothetical Document Embedding Generator.
    
    Tạo câu trả lời giả định để cải thiện semantic search.
    Văn bản pháp luật có ngôn ngữ đặc thù, HyDE giúp bridge
    giữa câu hỏi informal và legal documents.
    """
    
    HYDE_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam. Hãy viết một đoạn văn bản pháp lý MẪU có thể trả lời câu hỏi sau. 

Đoạn văn phải:
1. Có cấu trúc như một điều luật thực tế
2. Sử dụng ngôn ngữ pháp lý chuẩn
3. Đề cập đến các khái niệm pháp lý liên quan
4. Không cần chính xác, chỉ cần giống format văn bản pháp luật

Câu hỏi: {query}

Đoạn văn pháp lý mẫu (giả định):"""
    
    def __init__(self, llm=None):
        """
        Khởi tạo HyDE Generator.
        
        Args:
            llm: LLM instance
        """
        self.llm = llm
    
    def generate(self, query: str) -> Optional[str]:
        """
        Tạo hypothetical document cho query.
        
        Args:
            query: Câu hỏi của user
            
        Returns:
            Hypothetical document hoặc None nếu thất bại
        """
        if self.llm is None:
            return None
        
        prompt = self.HYDE_PROMPT.format(query=query)
        
        try:
            response = str(self.llm.complete(prompt)).strip()
            
            # Validate response
            if len(response) < 50:
                return None
            
            return response
            
        except Exception as e:
            print(f"[WARNING] HyDE generation failed: {e}")
            return None


class QueryEnhancer:
    """
    Orchestrator cho Query Enhancement.
    
    Kết hợp:
    - Query Classification
    - Query Decomposition
    - HyDE Generation
    - Adaptive Retrieval Strategy
    """
    
    def __init__(self, llm=None, use_llm_classification: bool = False):
        """
        Khởi tạo Query Enhancer.
        
        Args:
            llm: LLM instance
            use_llm_classification: Dùng LLM cho classification (chậm hơn nhưng chính xác hơn)
        """
        self.llm = llm
        
        # Initialize components
        mode = "llm_based" if use_llm_classification and llm else "rule_based"
        self.classifier = QueryClassifier(llm=llm, mode=mode)
        self.decomposer = QueryDecomposer(llm=llm)
        self.hyde_generator = HyDEGenerator(llm=llm)
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Phân tích và enhance câu hỏi.
        
        Args:
            query: Câu hỏi gốc
            
        Returns:
            QueryAnalysis với thông tin đầy đủ
        """
        # Step 1: Classify query
        query_type, confidence = self.classifier.classify(query)
        
        # Step 2: Get retrieval strategy
        strategy = self.classifier.get_strategy(query_type)
        
        # Step 3: Check complexity và decompose nếu cần
        is_complex = self.decomposer.is_complex(query)
        sub_queries = self.decomposer.decompose(query) if is_complex else [query]
        
        # Step 4: Build analysis result
        analysis = QueryAnalysis(
            original_query=query,
            query_type=query_type,
            sub_queries=sub_queries,
            is_complex=is_complex,
            confidence=confidence,
            retrieval_config={
                "top_k": strategy.top_k,
                "use_fusion": strategy.use_fusion,
                "use_reranking": strategy.use_reranking,
                "use_hyde": strategy.use_hyde,
                "use_iterative": strategy.use_iterative,
                "max_iterations": strategy.max_iterations,
            }
        )
        
        return analysis
    
    def generate_hyde_query(self, query: str) -> Optional[str]:
        """
        Tạo HyDE query nếu cần.
        
        Args:
            query: Câu hỏi gốc
            
        Returns:
            Hypothetical document hoặc None
        """
        return self.hyde_generator.generate(query)


# =====================================================
# TEST MODULE
# =====================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Query Enhancement Module")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "Điều 5 Luật Đấu thầu quy định gì?",  # FACTUAL
        "So sánh quyền nhà thầu trong Luật Đấu thầu và Luật Xây dựng",  # COMPARATIVE
        "Quy trình đấu thầu rộng rãi gồm những bước nào?",  # PROCEDURAL
        "Nếu nhà thầu vi phạm Điều 16 sẽ bị xử lý thế nào?",  # ANALYTICAL
        "Nhà thầu cần điều kiện gì để tham gia đấu thầu và có nghĩa vụ gì khi thi công?",  # COMPLEX
    ]
    
    # Test classifier (rule-based, no LLM needed)
    classifier = QueryClassifier(mode="rule_based")
    decomposer = QueryDecomposer()
    
    print("\n--- Query Classification (Rule-based) ---")
    for query in test_queries:
        query_type, confidence = classifier.classify(query)
        is_complex = decomposer.is_complex(query)
        strategy = classifier.get_strategy(query_type)
        
        print(f"\nQuery: {query}")
        print(f"  Type: {query_type.value} (confidence: {confidence:.2f})")
        print(f"  Complex: {is_complex}")
        print(f"  Strategy: top_k={strategy.top_k}, hyde={strategy.use_hyde}, iterative={strategy.use_iterative}")
    
    print("\n--- Query Decomposition (Rule-based) ---")
    complex_query = "Nhà thầu cần điều kiện gì để tham gia đấu thầu và có nghĩa vụ gì khi thi công?"
    sub_queries = decomposer.decompose(complex_query)
    
    print(f"\nOriginal: {complex_query}")
    print(f"Sub-queries ({len(sub_queries)}):")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED!")
    print("=" * 60)
