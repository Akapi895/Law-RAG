# -*- coding: utf-8 -*-
"""
vietnamese.py - Vietnamese Embedding Model

Wrapper cho BKAI Vietnamese Bi-Encoder model.
Model này được huấn luyện đặc biệt cho ngữ nghĩa tiếng Việt.
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import EMBEDDING_MODEL_NAME


class VietnameseEmbedding:
    """
    Wrapper cho BKAI Vietnamese Bi-Encoder embedding model.
    
    Model này được huấn luyện đặc biệt cho tiếng Việt,
    hiệu quả hơn các model đa ngôn ngữ cho văn bản luật Việt Nam.
    """
    
    def __init__(self, model_name: str = None):
        """
        Khởi tạo Vietnamese Embedding Model.
        
        Args:
            model_name: Tên model trên HuggingFace (mặc định từ config)
        """
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self._model = None
    
    def load(self) -> HuggingFaceEmbedding:
        """
        Tải model từ HuggingFace.
        
        Returns:
            HuggingFaceEmbedding instance
        """
        if self._model is None:
            print(f"[INFO] Đang tải Embedding Model: {self.model_name}")
            print("[INFO] (Lần đầu chạy sẽ tải model từ HuggingFace, có thể mất vài phút)")
            
            self._model = HuggingFaceEmbedding(
                model_name=self.model_name,
                trust_remote_code=True
            )
            print("[SUCCESS] Đã tải Embedding Model thành công!")
        
        return self._model
    
    @property
    def model(self) -> HuggingFaceEmbedding:
        """Trả về model đã được load."""
        return self.load()
