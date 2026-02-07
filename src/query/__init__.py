# -*- coding: utf-8 -*-
"""
src/query/__init__.py - Query Enhancement Package

Phase 2 Enhancement:
- Query Classification & Decomposition
- HyDE (Hypothetical Document Embedding)
- Iterative Retrieval with Gap Analysis
"""

from src.query.query_enhancement import (
    QueryType,
    QueryAnalysis,
    RetrievalStrategy,
    QueryClassifier,
    QueryDecomposer,
    HyDEGenerator,
    QueryEnhancer,
    RETRIEVAL_STRATEGIES,
)

from src.query.iterative_retrieval import (
    RetrievalIteration,
    IterativeRetrievalResult,
    GapAnalyzer,
    IterativeRetriever,
    RelevanceEvaluator,
)

__all__ = [
    # Query Enhancement
    "QueryType",
    "QueryAnalysis", 
    "RetrievalStrategy",
    "QueryClassifier",
    "QueryDecomposer",
    "HyDEGenerator",
    "QueryEnhancer",
    "RETRIEVAL_STRATEGIES",
    # Iterative Retrieval
    "RetrievalIteration",
    "IterativeRetrievalResult",
    "GapAnalyzer",
    "IterativeRetriever",
    "RelevanceEvaluator",
]
