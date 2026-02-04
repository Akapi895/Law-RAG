# -*- coding: utf-8 -*-
"""
qdrant.py - Qdrant Vector Store Management

Quản lý Qdrant client và collection.
"""

import os
from typing import Optional, List

from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import (
    QDRANT_COLLECTION_NAME,
    QDRANT_IN_MEMORY,
    QDRANT_STORAGE_PATH
)


class QdrantStore:
    """
    Quản lý Qdrant Vector Database.
    
    Hỗ trợ cả in-memory và persistent storage modes.
    """
    
    def __init__(
        self,
        collection_name: str = None,
        in_memory: bool = None,
        storage_path: str = None
    ):
        """
        Khởi tạo Qdrant Store.
        
        Args:
            collection_name: Tên collection (mặc định từ config)
            in_memory: Sử dụng in-memory mode (mặc định từ config)
            storage_path: Đường dẫn lưu trữ (mặc định từ config)
        """
        self.collection_name = collection_name or QDRANT_COLLECTION_NAME
        self.in_memory = in_memory if in_memory is not None else QDRANT_IN_MEMORY
        self.storage_path = storage_path or QDRANT_STORAGE_PATH
        
        self._client: Optional[QdrantClient] = None
        self._vector_store: Optional[QdrantVectorStore] = None
        self._index: Optional[VectorStoreIndex] = None
    
    def _init_client(self) -> QdrantClient:
        """Khởi tạo Qdrant client."""
        if self._client is None:
            if self.in_memory:
                print("[INFO] Sử dụng Qdrant in-memory mode")
                self._client = QdrantClient(":memory:")
            else:
                if not os.path.exists(self.storage_path):
                    os.makedirs(self.storage_path)
                    print(f"[INFO] Đã tạo thư mục storage: {self.storage_path}")
                
                print(f"[INFO] Sử dụng Qdrant persistent mode: {self.storage_path}")
                self._client = QdrantClient(path=self.storage_path)
        
        return self._client
    
    @property
    def client(self) -> QdrantClient:
        """Trả về Qdrant client."""
        return self._init_client()
    
    def collection_exists(self) -> bool:
        """Kiểm tra collection đã tồn tại chưa."""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception:
            return False
    
    def delete_collection(self):
        """Xóa collection hiện tại."""
        if self.collection_exists():
            print(f"[INFO] Xóa collection cũ: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            self._vector_store = None
            self._index = None
    
    def get_vector_store(self) -> QdrantVectorStore:
        """Tạo hoặc lấy vector store."""
        if self._vector_store is None:
            print(f"[INFO] Đang tạo collection: {self.collection_name}")
            self._vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name
            )
        return self._vector_store
    
    def build_index(self, documents: List[Document], show_progress: bool = True) -> VectorStoreIndex:
        """
        Xây dựng Vector Index từ documents.
        
        Args:
            documents: Danh sách LlamaIndex Documents
            show_progress: Hiển thị progress bar
            
        Returns:
            VectorStoreIndex
        """
        if not documents:
            raise ValueError("Danh sách documents trống!")
        
        print(f"[INFO] Đang xây dựng Vector Index với {len(documents)} documents...")
        
        # Xóa collection cũ
        self.delete_collection()
        
        # Tạo storage context
        storage_context = StorageContext.from_defaults(
            vector_store=self.get_vector_store()
        )
        
        # Build index
        print("[INFO] Đang embedding và đánh chỉ mục documents...")
        self._index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=show_progress
        )
        
        print(f"[SUCCESS] Đã xây dựng Vector Index với {len(documents)} documents!")
        return self._index
    
    def load_index(self) -> Optional[VectorStoreIndex]:
        """
        Load index từ storage (nếu có).
        
        Returns:
            VectorStoreIndex nếu tồn tại, None nếu không
        """
        if not self.collection_exists():
            print("[INFO] Chưa có index được lưu trước đó")
            return None
        
        print(f"[INFO] Đang load index từ storage...")
        
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self.get_vector_store()
        )
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            doc_count = collection_info.points_count
            print(f"[SUCCESS] Đã load index với {doc_count} documents!")
        except Exception:
            print("[SUCCESS] Đã load index từ storage!")
        
        return self._index
    
    @property
    def index(self) -> Optional[VectorStoreIndex]:
        """Trả về index hiện tại."""
        return self._index
    
    def get_documents_count(self) -> int:
        """Đếm số documents trong collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception:
            return 0
