# -*- coding: utf-8 -*-
"""
metadata.py - Metadata Filter cho Vietnamese Legal RAG

Cho phép lọc kết quả theo:
- doc_name: Tên văn bản (Luật Đấu thầu, Luật Xây dựng...)
- topic: Lĩnh vực (Đấu thầu, Xây dựng...)
- article_id: Số điều (Điều 5, Điều 16...)
- doc_type: Loại văn bản (Luật, Nghị định...)
"""

import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from llama_index.core import Document


@dataclass
class MetadataFilter:
    """
    Bộ lọc metadata cho documents.
    
    Attributes:
        doc_name: Lọc theo tên văn bản (partial match)
        topic: Lọc theo lĩnh vực (exact match)
        article_id: Lọc theo số điều (partial match)
        doc_type: Lọc theo loại văn bản (exact match)
    """
    doc_name: Optional[str] = None
    topic: Optional[str] = None
    article_id: Optional[str] = None
    doc_type: Optional[str] = None
    
    def is_empty(self) -> bool:
        """Kiểm tra filter có trống không."""
        return all(v is None for v in [self.doc_name, self.topic, self.article_id, self.doc_type])
    
    def matches(self, metadata: Dict[str, Any]) -> bool:
        """
        Kiểm tra metadata có khớp với filter không.
        
        Args:
            metadata: Metadata của document
            
        Returns:
            True nếu khớp tất cả điều kiện
        """
        if self.doc_name:
            doc_name = str(metadata.get('doc_name', '')).lower()
            if self.doc_name.lower() not in doc_name:
                return False
        
        if self.topic:
            topic = str(metadata.get('topic', '')).lower()
            if self.topic.lower() != topic:
                return False
        
        if self.article_id:
            article_id = str(metadata.get('article_id', '')).lower()
            if self.article_id.lower() not in article_id:
                return False
        
        if self.doc_type:
            doc_type = str(metadata.get('doc_type', '')).lower()
            if self.doc_type.lower() != doc_type:
                return False
        
        return True
    
    def __str__(self) -> str:
        parts = []
        if self.doc_name:
            parts.append(f"doc_name='{self.doc_name}'")
        if self.topic:
            parts.append(f"topic='{self.topic}'")
        if self.article_id:
            parts.append(f"article='{self.article_id}'")
        if self.doc_type:
            parts.append(f"type='{self.doc_type}'")
        return f"Filter({', '.join(parts)})" if parts else "Filter(none)"


class MetadataFilterParser:
    """
    Parser để trích xuất metadata filter từ câu hỏi tự nhiên.
    
    Ví dụ:
    - "Điều 5 Luật Đấu thầu quy định gì?" → article_id="Điều 5", doc_name="Đấu thầu"
    - "Trong Luật Xây dựng, nghĩa vụ nhà thầu là gì?" → doc_name="Xây dựng"
    """
    
    # Patterns để nhận diện metadata trong câu hỏi
    PATTERNS = {
        'article_id': [
            r'điều\s+(\d+)',                    # "Điều 5", "điều 16"
            r'khoản\s+(\d+)\s+điều\s+(\d+)',    # "khoản 2 điều 5"
        ],
        'doc_name': [
            r'luật\s+(đấu\s*thầu)',             # "Luật Đấu thầu"
            r'luật\s+(xây\s*dựng)',             # "Luật Xây dựng"
            r'nghị\s+định\s+(\d+)',             # "Nghị định 123"
        ],
        'topic': [
            r'\b(đấu\s*thầu)\b',                # "đấu thầu"
            r'\b(xây\s*dựng)\b',                # "xây dựng"
        ]
    }
    
    @classmethod
    def parse(cls, query: str) -> MetadataFilter:
        """
        Parse câu hỏi để trích xuất metadata filter.
        
        Args:
            query: Câu hỏi của user
            
        Returns:
            MetadataFilter object
        """
        query_lower = query.lower()
        filter_obj = MetadataFilter()
        
        # Tìm article_id
        for pattern in cls.PATTERNS['article_id']:
            match = re.search(pattern, query_lower)
            if match:
                if len(match.groups()) == 2:  # khoản X điều Y
                    filter_obj.article_id = f"Điều {match.group(2)}"
                else:
                    filter_obj.article_id = f"Điều {match.group(1)}"
                break
        
        # Tìm doc_name
        for pattern in cls.PATTERNS['doc_name']:
            match = re.search(pattern, query_lower)
            if match:
                name = match.group(1).strip()
                # Normalize tên
                if 'đấu' in name and 'thầu' in name:
                    filter_obj.doc_name = "Đấu thầu"
                elif 'xây' in name and 'dựng' in name:
                    filter_obj.doc_name = "Xây dựng"
                else:
                    filter_obj.doc_name = name.title()
                break
        
        # Tìm topic (nếu chưa có doc_name)
        if not filter_obj.doc_name:
            for pattern in cls.PATTERNS['topic']:
                match = re.search(pattern, query_lower)
                if match:
                    topic = match.group(1).strip()
                    if 'đấu' in topic and 'thầu' in topic:
                        filter_obj.topic = "Đấu thầu"
                    elif 'xây' in topic and 'dựng' in topic:
                        filter_obj.topic = "Xây dựng"
                    break
        
        return filter_obj


def filter_documents(
    documents: List[Document],
    metadata_filter: MetadataFilter
) -> List[Document]:
    """
    Lọc documents theo metadata filter.
    
    Args:
        documents: Danh sách documents
        metadata_filter: Bộ lọc metadata
        
    Returns:
        Danh sách documents đã được lọc
    """
    if metadata_filter.is_empty():
        return documents
    
    filtered = [
        doc for doc in documents
        if metadata_filter.matches(doc.metadata)
    ]
    
    return filtered


# =====================================================
# TEST MODULE
# =====================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TEST: Metadata Filter Parser")
    print("=" * 50)
    
    test_queries = [
        "Điều 5 Luật Đấu thầu quy định gì?",
        "Trong Luật Xây dựng, nghĩa vụ nhà thầu là gì?",
        "Các hành vi bị cấm trong đấu thầu?",
        "Điều 16 nói về vấn đề gì?",
        "Quy định về xây dựng công trình",
    ]
    
    for query in test_queries:
        filter_obj = MetadataFilterParser.parse(query)
        print(f"\nQuery: {query}")
        print(f"  → {filter_obj}")
