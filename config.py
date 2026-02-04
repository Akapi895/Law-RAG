# -*- coding: utf-8 -*-
"""
config.py - File cấu hình cho Hệ thống RAG Luật Việt Nam

Chứa các biến cấu hình:
- API Key (lấy từ biến môi trường)
- Tên các Model AI
- System Prompt cho LLM
- Đường dẫn file dữ liệu
"""

import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# =====================================================
# CẤU HÌNH API KEY
# =====================================================
# Lấy Google API Key từ biến môi trường
# Đảm bảo bạn đã tạo file .env với nội dung: GOOGLE_API_KEY=your_api_key_here
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# =====================================================
# CẤU HÌNH MODEL
# =====================================================
# Model embedding tiếng Việt từ BKAI (HuggingFace)
# Model này được huấn luyện đặc biệt cho ngữ nghĩa tiếng Việt
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

# Model LLM từ Google Gemini (phiên bản miễn phí)
# Gemini 2.0 Flash là model mới nhất (thay thế 1.5 Flash từ Jan 2025)
# Các model có sẵn: gemini-2.0-flash, gemini-2.0-pro, gemini-1.5-pro
LLM_MODEL_NAME = "gemini-2.5-flash"

# =====================================================
# CẤU HÌNH RETRIEVAL
# =====================================================
# Số lượng document tương tự nhất sẽ được truy xuất
SIMILARITY_TOP_K = 5

# Kích thước chunk khi chia nhỏ văn bản (nếu cần)
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

# =====================================================
# CẤU HÌNH FUSION RETRIEVAL
# =====================================================
# Kết hợp Vector Search (semantic) + BM25 (keyword)
# FUSION_ALPHA: trọng số cho vector search (0.0 - 1.0)
# - 1.0 = chỉ dùng vector search
# - 0.0 = chỉ dùng BM25 (keyword)
# - 0.5 = cân bằng cả hai (khuyên dùng)
FUSION_ALPHA = 0.5

# Bật/tắt Fusion Retrieval
USE_FUSION_RETRIEVAL = True

# =====================================================
# CẤU HÌNH METADATA FILTERING
# =====================================================
# Tự động lọc kết quả theo metadata từ câu hỏi
# Ví dụ: "Điều 5 Luật Đấu thầu" → filter article_id="Điều 5", doc_name="Đấu thầu"
USE_METADATA_FILTERING = True

# =====================================================
# CẤU HÌNH CROSS-ENCODER RERANKING
# =====================================================
# Sử dụng Cross-Encoder để rerank kết quả retrieval
# Cross-Encoder đánh giá cặp (query, document) chính xác hơn bi-encoder
USE_RERANKING = True

# Model cross-encoder (từ HuggingFace)
# - "cross-encoder/ms-marco-MiniLM-L-6-v2" (fast, multilingual)
# - "BAAI/bge-reranker-base" (tốt cho tiếng Việt)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Số documents để fetch trước khi rerank (nhiều hơn top_k để có dữ liệu rerank)
RERANK_TOP_K = 15

# =====================================================
# CẤU HÌNH ĐƯỜNG DẪN FILE
# =====================================================
# Đường dẫn đến file Excel chứa dữ liệu luật
# Sử dụng Excel thay vì CSV để tránh lỗi encoding tiếng Việt
DATA_FILE_PATH = "legal_data.xlsx"

# =====================================================
# SYSTEM PROMPT CHO LLM
# =====================================================
# Prompt này hướng dẫn AI cách trả lời câu hỏi về luật
SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên về Luật Việt Nam, đặc biệt là Luật Đấu thầu và Luật Xây dựng.

NHIỆM VỤ CỦA BẠN:
1. Trả lời câu hỏi của người dùng dựa trên thông tin được cung cấp trong phần Ngữ cảnh (Context).
2. Câu trả lời phải chính xác, rõ ràng và chuyên nghiệp.
3. Sử dụng ngôn ngữ pháp lý phù hợp nhưng dễ hiểu.

QUY TẮC BẮT BUỘC:
- BẮT BUỘC phải trích dẫn nguồn ở cuối câu trả lời theo format: "(Theo Điều X, Luật Y)"
- Nếu thông tin từ nhiều điều luật, hãy liệt kê tất cả các nguồn.
- Nếu KHÔNG tìm thấy thông tin liên quan trong Context, hãy nói rõ: "Tôi không tìm thấy thông tin về vấn đề này trong cơ sở dữ liệu hiện có."
- KHÔNG được bịa đặt hoặc suy luận ngoài phạm vi Context.

FORMAT TRẢ LỜI:
1. Trả lời trực tiếp câu hỏi
2. Giải thích chi tiết nếu cần
3. Trích dẫn nguồn (Điều luật, Văn bản)
"""

# =====================================================
# CẤU HÌNH QDRANT (Vector Database)
# =====================================================
# Tên collection trong Qdrant
QDRANT_COLLECTION_NAME = "vietnamese_legal_documents"

# Chế độ chạy: 
# - True = in-memory (nhanh nhưng mất khi tắt)
# - False = persistent (lưu ra disk, giữ lại khi tắt)
QDRANT_IN_MEMORY = False

# Đường dẫn lưu trữ Qdrant (chỉ dùng khi QDRANT_IN_MEMORY = False)
QDRANT_STORAGE_PATH = "./qdrant_storage"
