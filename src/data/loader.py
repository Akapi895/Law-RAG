# -*- coding: utf-8 -*-
"""
loader.py - Module xử lý dữ liệu cho Hệ thống RAG Luật Việt Nam

Chức năng:
1. Tạo dữ liệu giả (mock data) nếu file Excel chưa tồn tại
2. Đọc và xử lý file Excel (.xlsx)
3. Chuyển đổi dữ liệu thành LlamaIndex Documents
"""

import os
import pandas as pd
from llama_index.core import Document
from config import DATA_FILE_PATH


def generate_mock_data(file_path: str = DATA_FILE_PATH) -> None:
    """
    Tạo file Excel mẫu với dữ liệu giả về Luật Đấu thầu và Luật Xây dựng.
    
    Args:
        file_path: Đường dẫn đến file Excel sẽ được tạo
    """
    if os.path.exists(file_path):
        print(f"[INFO] File '{file_path}' đã tồn tại. Bỏ qua bước tạo dữ liệu mẫu.")
        return
    
    print(f"[INFO] Đang tạo file dữ liệu mẫu: {file_path}")
    
    mock_data = [
        {
            "doc_id": "22/2023/QH15",
            "doc_name": "Luật Đấu thầu năm 2023",
            "doc_type": "Luật",
            "topic": "Đấu thầu",
            "article_id": "Điều 5",
            "title": "Tư cách hợp lệ của nhà thầu, nhà đầu tư",
            "content": """Nhà thầu, nhà đầu tư là tổ chức có tư cách hợp lệ khi đáp ứng đủ các điều kiện sau đây:
1. Là doanh nghiệp, hợp tác xã được thành lập theo quy định của pháp luật Việt Nam.
2. Hạch toán tài chính độc lập.
3. Không đang trong quá trình chấm dứt hoạt động.
4. Không đang trong thời gian bị cấm tham dự thầu.
5. Có tên trong danh sách ngắn (nếu có).
6. Bảo đảm cạnh tranh trong đấu thầu.
7. Đã đăng ký trên Hệ thống mạng đấu thầu quốc gia.""",
            "status": "Còn hiệu lực"
        },
        {
            "doc_id": "22/2023/QH15",
            "doc_name": "Luật Đấu thầu năm 2023",
            "doc_type": "Luật",
            "topic": "Đấu thầu",
            "article_id": "Điều 16",
            "title": "Các hành vi bị cấm trong hoạt động đấu thầu",
            "content": """1. Đưa, nhận, môi giới hối lộ.
2. Lợi dụng chức vụ, quyền hạn để can thiệp bất hợp pháp vào hoạt động đấu thầu.
3. Thông thầu.""",
            "status": "Còn hiệu lực"
        },
        {
            "doc_id": "50/2014/QH13",
            "doc_name": "Luật Xây dựng năm 2014",
            "doc_type": "Luật",
            "topic": "Xây dựng",
            "article_id": "Điều 113",
            "title": "Nghĩa vụ của nhà thầu thi công xây dựng công trình",
            "content": """1. Chỉ được nhận thầu thi công phù hợp với điều kiện năng lực.
2. Thực hiện đúng hợp đồng đã ký kết.
3. Lập và trình chủ đầu tư phê duyệt biện pháp thi công.
4. Thi công theo đúng thiết kế, tiêu chuẩn áp dụng.""",
            "status": "Còn hiệu lực"
        }
    ]
    
    df = pd.DataFrame(mock_data)
    df.to_excel(file_path, index=False, engine='openpyxl')
    
    print(f"[SUCCESS] Đã tạo file '{file_path}' với {len(mock_data)} điều luật mẫu.")


def create_contextual_header(row: dict) -> str:
    """
    Tạo Contextual Chunk Header cho document.
    
    Header này được prepend vào content để embedding model
    "hiểu" được context đầy đủ của mỗi chunk.
    
    Args:
        row: Dictionary chứa metadata của document
        
    Returns:
        Header string được format
    """
    # Xử lý hierarchy info nếu có
    chapter = row.get('chapter', '')
    section = row.get('section', '')
    
    # Build hierarchy path
    hierarchy_parts = []
    if chapter:
        hierarchy_parts.append(chapter)
    if section:
        hierarchy_parts.append(section) 
    hierarchy_parts.append(str(row.get('article_id', '')))
    hierarchy_path = ' > '.join(filter(None, hierarchy_parts))
    
    # Build header
    header_lines = [
        f"[VĂN BẢN: {row.get('doc_name', 'N/A')} ({row.get('doc_id', 'N/A')})]",
        f"[LOẠI: {row.get('doc_type', 'N/A')}]",
        f"[LĨNH VỰC: {row.get('topic', 'N/A')}]",
    ]
    
    if hierarchy_path:
        header_lines.append(f"[VỊ TRÍ: {hierarchy_path}]")
    
    header_lines.extend([
        f"[TIÊU ĐỀ: {row.get('title', 'N/A')}]",
        f"[HIỆU LỰC: {row.get('status', 'N/A')}]",
        "---"
    ])
    
    return '\n'.join(header_lines)


def compute_content_hash(content: str) -> str:
    """
    Tính hash SHA256 của content để tracking version.
    
    Args:
        content: Nội dung văn bản
        
    Returns:
        Hash string (16 ký tự đầu)
    """
    import hashlib
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def load_and_process_data(file_path: str = DATA_FILE_PATH) -> list:
    """
    Đọc file Excel và chuyển đổi thành danh sách LlamaIndex Documents.
    
    Cải tiến Phase 1:
    - Contextual Chunk Headers cho mỗi document
    - Enhanced metadata schema với hierarchy info
    - Content hash cho version tracking
    
    Args:
        file_path: Đường dẫn đến file Excel
        
    Returns:
        Danh sách các LlamaIndex Document objects
    """
    print(f"[INFO] Đang đọc dữ liệu từ file: {file_path}")
    
    df = pd.read_excel(file_path, engine='openpyxl')
    print(f"[INFO] Đã đọc được {len(df)} dòng dữ liệu")
    
    # Lọc bỏ các điều luật đã hết hiệu lực (chỉ giữ "Hiệu lực" hoặc "Còn hiệu lực")
    if 'status' in df.columns:
        original_count = len(df)
        valid_status = ['Hiệu lực', 'Còn hiệu lực']
        df = df[df['status'].isin(valid_status)]
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            print(f"[INFO] Đã lọc bỏ {filtered_count} điều luật hết hiệu lực")
    
    documents = []
    
    for idx, row in df.iterrows():
        # Chuyển row thành dict để xử lý
        row_dict = row.to_dict()
        
        # Tạo contextual header
        contextual_header = create_contextual_header(row_dict)
        
        # Tạo nội dung gốc (không có header)
        original_content = str(row.get('content', ''))
        
        # Tạo semantic text với contextual header
        # Header + Title + Content để embedding hiểu context
        semantic_text = f"{contextual_header}\n{row['title']}:\n{original_content}"
        
        # Tính content hash cho version tracking
        content_hash = compute_content_hash(original_content)
        
        # Enhanced metadata schema
        metadata = {
            # Thông tin văn bản
            "doc_id": str(row.get('doc_id', '')),
            "doc_name": str(row.get('doc_name', '')),
            "doc_type": str(row.get('doc_type', '')),
            
            # Phân loại
            "topic": str(row.get('topic', '')),
            
            # Vị trí trong văn bản (hierarchy)
            "chapter": str(row.get('chapter', '')),
            "section": str(row.get('section', '')),
            "article_id": str(row.get('article_id', '')),
            "title": str(row.get('title', '')),
            
            # Trạng thái và version
            "status": str(row.get('status', 'Hiệu lực')),
            "content_hash": content_hash,
            "effective_date": str(row.get('effective_date', '')),
            
            # Tracking
            "source_file": file_path,
            "row_index": idx,
        }
        
        # Tạo unique doc_id
        unique_doc_id = f"{row.get('doc_id', '')}_{row.get('article_id', '')}".replace(' ', '_')
        
        doc = Document(
            text=semantic_text,
            metadata=metadata,
            doc_id=unique_doc_id
        )
        
        documents.append(doc)
    
    print(f"[SUCCESS] Đã tạo {len(documents)} LlamaIndex Documents với Contextual Headers")
    
    # Thống kê theo lĩnh vực
    if 'topic' in df.columns:
        topics = df['topic'].value_counts()
        print("[INFO] Thống kê theo lĩnh vực:")
        for topic, count in topics.items():
            print(f"  - {topic}: {count} điều")
    
    # Thống kê theo loại văn bản
    if 'doc_type' in df.columns:
        doc_types = df['doc_type'].value_counts()
        print("[INFO] Thống kê theo loại văn bản:")
        for doc_type, count in doc_types.items():
            print(f"  - {doc_type}: {count} điều")
    
    return documents


# =====================================================
# TEST MODULE
# =====================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TEST: Data Loader Module")
    print("=" * 50)
    
    generate_mock_data()
    docs = load_and_process_data()
    
    if docs:
        print("\n" + "=" * 50)
        print("MẪU DOCUMENT ĐẦU TIÊN:")
        print("=" * 50)
        print(f"Text (100 ký tự đầu): {docs[0].text[:100]}...")
        print(f"Metadata: {docs[0].metadata}")
