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


def load_and_process_data(file_path: str = DATA_FILE_PATH) -> list:
    """
    Đọc file Excel và chuyển đổi thành danh sách LlamaIndex Documents.
    
    Args:
        file_path: Đường dẫn đến file Excel
        
    Returns:
        Danh sách các LlamaIndex Document objects
    """
    print(f"[INFO] Đang đọc dữ liệu từ file: {file_path}")
    
    df = pd.read_excel(file_path, engine='openpyxl')
    print(f"[INFO] Đã đọc được {len(df)} dòng dữ liệu")
    
    # Lọc bỏ các điều luật đã hết hiệu lực
    if 'status' in df.columns:
        original_count = len(df)
        df = df[df['status'] == 'Hiệu lực']
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            print(f"[INFO] Đã lọc bỏ {filtered_count} điều luật hết hiệu lực")
    
    documents = []
    
    for idx, row in df.iterrows():
        # Tạo chuỗi ngữ nghĩa đầy đủ cho embedding
        semantic_text = f"{row['doc_name']} - {row['article_id']} - {row['title']}: {row['content']}"
        
        # Metadata để hỗ trợ trích dẫn nguồn
        metadata = {
            "doc_id": str(row['doc_id']),
            "doc_name": str(row['doc_name']),
            "doc_type": str(row['doc_type']),
            "topic": str(row['topic']),
            "article_id": str(row['article_id']),
            "title": str(row['title']),
            "status": str(row.get('status', 'Hiệu lực'))
        }
        
        doc = Document(
            text=semantic_text,
            metadata=metadata,
            doc_id=f"{row['doc_id']}_{row['article_id']}"
        )
        
        documents.append(doc)
    
    print(f"[SUCCESS] Đã tạo {len(documents)} LlamaIndex Documents")
    
    # Thống kê theo lĩnh vực
    if 'topic' in df.columns:
        topics = df['topic'].value_counts()
        print("[INFO] Thống kê theo lĩnh vực:")
        for topic, count in topics.items():
            print(f"  - {topic}: {count} điều")
    
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
