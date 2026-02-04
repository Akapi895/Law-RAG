# -*- coding: utf-8 -*-
"""
gemini.py - Google Gemini LLM Integration

Wrapper cho Google Gemini API.
"""

from llama_index.llms.gemini import Gemini
from config import GOOGLE_API_KEY, LLM_MODEL_NAME


class GeminiLLM:
    """
    Wrapper cho Google Gemini LLM.
    
    Cung cấp interface thống nhất để sử dụng Gemini
    với các cấu hình mặc định phù hợp cho hệ thống luật.
    """
    
    def __init__(self, model_name: str = None, temperature: float = 0.3):
        """
        Khởi tạo Gemini LLM.
        
        Args:
            model_name: Tên model Gemini (mặc định từ config)
            temperature: Độ sáng tạo của model (0.0 - 1.0)
        """
        self.model_name = model_name or LLM_MODEL_NAME
        self.temperature = temperature
        self._llm = None
    
    def load(self) -> Gemini:
        """
        Khởi tạo và trả về Gemini LLM instance.
        
        Returns:
            Gemini LLM instance
            
        Raises:
            ValueError: Nếu không tìm thấy API Key
        """
        if self._llm is None:
            if not GOOGLE_API_KEY:
                raise ValueError(
                    "Không tìm thấy GOOGLE_API_KEY!\n"
                    "Vui lòng tạo file .env với nội dung:\n"
                    "GOOGLE_API_KEY=your_api_key_here"
                )
            
            print(f"[INFO] Đang khởi tạo LLM: {self.model_name}")
            self._llm = Gemini(
                model=self.model_name,
                api_key=GOOGLE_API_KEY,
                temperature=self.temperature
            )
            print("[SUCCESS] Đã khởi tạo LLM thành công!")
        
        return self._llm
    
    @property
    def llm(self) -> Gemini:
        """Trả về LLM đã được load."""
        return self.load()
