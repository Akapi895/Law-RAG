# -*- coding: utf-8 -*-
"""
src - Package chính của Hệ thống RAG Luật Việt Nam

Modules:
- data: Xử lý và load dữ liệu
- embeddings: Embedding models
- llm: Large Language Models
- retrievers: Các chiến lược retrieval
- vector_store: Vector database
- engine: RAG Engine chính
"""

from src.engine import LegalRAGSystem

__all__ = ['LegalRAGSystem']
