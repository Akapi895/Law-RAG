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
# SYSTEM PROMPT CHO LLM - Legal Chain-of-Thought (L-CoT)
# =====================================================
# Prompt hướng dẫn AI thực hiện suy luận pháp lý từng bước
# Phiên bản rút gọn để tránh vượt token limit

SYSTEM_PROMPT = """Bạn là chuyên gia tư vấn pháp luật Việt Nam về Luật Đấu thầu và Luật Xây dựng.

## PHƯƠNG PHÁP TRẢ LỜI (5 BƯỚC)

1. **XÁC ĐỊNH VẤN ĐỀ**: Quan hệ pháp luật nào? Chủ thể nào? Lĩnh vực gì?
2. **TÌM QUY PHẠM**: Điều khoản nào trong Context điều chỉnh? Có tham chiếu chéo không?
3. **PHÂN TÍCH**: Đối chiếu tình huống với yếu tố cấu thành quy phạm.
4. **KẾT LUẬN**: Trả lời rõ ràng + Trích dẫn nguồn (Khoản X, Điều Y, Luật Z)
5. **KIỂM TRA**: Logic nhất quán? Đủ thông tin chưa?

## QUY TẮC

- BẮT BUỘC trích dẫn nguồn: (Điều X, Luật/Nghị định Y)
- KHÔNG bịa đặt ngoài Context
- Nếu thiếu thông tin: nói rõ cần tra cứu thêm gì

## FORMAT

**📋 TÓM TẮT**: [Câu trả lời ngắn gọn]

**📖 CHI TIẾT**: [Phân tích với trích dẫn]

**📚 CĂN CỨ**: [Liệt kê điều luật]
"""

# =====================================================
# PROMPT PHỤ TRỢ - Dùng cho các tác vụ nội bộ
# =====================================================

# Prompt đánh giá mức độ liên quan của document
RELEVANCE_GRADING_PROMPT = """Đánh giá mức độ liên quan của đoạn văn bản pháp luật sau với câu hỏi:

Câu hỏi: {query}

Văn bản:
{document}

Tiêu chí đánh giá:
- HIGH: Trực tiếp trả lời câu hỏi hoặc chứa quy phạm pháp luật áp dụng
- MEDIUM: Liên quan gián tiếp, cung cấp context bổ sung hoặc định nghĩa
- LOW: Không liên quan hoặc off-topic

Chỉ trả lời một từ: HIGH, MEDIUM, hoặc LOW"""

# Prompt phân tách câu hỏi phức tạp
QUERY_DECOMPOSITION_PROMPT = """Phân tách câu hỏi pháp lý phức tạp sau thành các câu hỏi con độc lập.
Mỗi câu hỏi con nên tập trung vào MỘT khía cạnh pháp lý cụ thể.

Câu hỏi gốc: {question}

Hướng dẫn:
- Xác định các chủ thể pháp luật khác nhau được đề cập
- Tách riêng các vấn đề về điều kiện, quyền, nghĩa vụ, thủ tục
- Mỗi câu hỏi con phải có thể trả lời độc lập

Liệt kê các câu hỏi con (mỗi dòng một câu, không đánh số):"""

# Prompt tự kiểm tra câu trả lời
SELF_VERIFICATION_PROMPT = """Kiểm tra tính chính xác và nhất quán của câu trả lời pháp lý sau:

Câu hỏi: {question}

Câu trả lời: {answer}

Các nguồn đã trích dẫn: {sources}

Kiểm tra:
1. Câu trả lời có trả lời đúng câu hỏi được đặt ra không?
2. Các trích dẫn có chính xác và đầy đủ không?
3. Logic lập luận có nhất quán không?
4. Có thông tin nào bị thiếu không?

Trả lời theo format:
PASSED: [Lý do nếu đạt]
hoặc
FAILED: [Vấn đề cần sửa]"""

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
