# -*- coding: utf-8 -*-
"""
main.py - File chạy chính của Hệ thống RAG Luật Việt Nam

Chương trình này:
1. Kiểm tra API Key
2. Tạo dữ liệu mẫu (nếu cần)
3. Load và xử lý dữ liệu
4. Khởi tạo hệ thống RAG
5. Chạy vòng lặp hỏi đáp với người dùng
"""

import sys
import os

# Thêm thư mục hiện tại vào path để import các module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

from config import GOOGLE_API_KEY, DATA_FILE_PATH
from src.data import generate_mock_data, load_and_process_data
from src.engine import LegalRAGSystem


def print_banner():
    """In banner chào mừng."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     HỆ THỐNG TRẢ LỜI CÂU HỎI VỀ LUẬT VIỆT NAM (RAG)              ║
║     ─────────────────────────────────────────────────            ║
║     Chuyên: Luật Đấu thầu & Luật Xây dựng                        ║
║                                                                  ║
║     Powered by: LlamaIndex + Qdrant + Google Gemini              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_api_key() -> bool:
    """Kiểm tra xem GOOGLE_API_KEY đã được cấu hình chưa."""
    if not GOOGLE_API_KEY:
        print("=" * 60)
        print("❌ LỖI: Không tìm thấy GOOGLE_API_KEY!")
        print("=" * 60)
        print()
        print("Vui lòng thực hiện các bước sau:")
        print()
        print("1. Truy cập: https://aistudio.google.com/app/apikey")
        print("2. Tạo API Key mới (miễn phí)")
        print("3. Tạo file .env trong thư mục project với nội dung:")
        print()
        print("   GOOGLE_API_KEY=your_api_key_here")
        print()
        print("4. Chạy lại chương trình")
        print()
        return False
    
    print(f"[OK] Đã tìm thấy GOOGLE_API_KEY: {GOOGLE_API_KEY[:10]}...")
    return True


def main():
    """Hàm chính chạy chương trình."""
    
    # In banner
    print_banner()
    
    # =====================================================
    # BƯỚC 1: KIỂM TRA API KEY
    # =====================================================
    print("\n" + "=" * 60)
    print("BƯỚC 1: Kiểm tra cấu hình")
    print("=" * 60)
    
    if not check_api_key():
        return
    
    # =====================================================
    # BƯỚC 2: TẠO DỮ LIỆU MẪU (NẾU CẦN)
    # =====================================================
    print("\n" + "=" * 60)
    print("BƯỚC 2: Chuẩn bị dữ liệu")
    print("=" * 60)
    
    generate_mock_data(DATA_FILE_PATH)
    
    # =====================================================
    # BƯỚC 3: LOAD VÀ XỬ LÝ DỮ LIỆU
    # =====================================================
    print("\n" + "=" * 60)
    print("BƯỚC 3: Load và xử lý dữ liệu")
    print("=" * 60)
    
    documents = load_and_process_data(DATA_FILE_PATH)
    
    if not documents:
        print("❌ Không có dữ liệu để xử lý!")
        return
    
    # =====================================================
    # BƯỚC 4: KHỞI TẠO HỆ THỐNG RAG
    # =====================================================
    print("\n" + "=" * 60)
    print("BƯỚC 4: Khởi tạo hệ thống RAG")
    print("=" * 60)
    
    try:
        rag_system = LegalRAGSystem()
    except Exception as e:
        print(f"❌ Lỗi khởi tạo hệ thống: {e}")
        return
    
    # =====================================================
    # BƯỚC 5: ĐỒNG BỘ VECTOR INDEX
    # =====================================================
    print("\n" + "=" * 60)
    print("BƯỚC 5: Đồng bộ Vector Index")
    print("=" * 60)
    
    try:
        stats = rag_system.sync_index(documents)
        print(f"[INFO] Thống kê: +{stats['added']} mới, -{stats['removed']} xóa, ={stats['unchanged']} giữ nguyên")
    except Exception as e:
        print(f"❌ Lỗi đồng bộ index: {e}")
        return
    
    # Khởi tạo query engine
    rag_system.get_query_engine()
    
    # =====================================================
    # BƯỚC 6: VÒNG LẶP HỎI ĐÁP
    # =====================================================
    print("\n" + "=" * 60)
    print("HỆ THỐNG ĐÃ SẴN SÀNG!")
    print("=" * 60)
    print()
    print("Bạn có thể đặt câu hỏi về Luật Đấu thầu và Luật Xây dựng.")
    print("Gõ 'exit' hoặc 'quit' để thoát chương trình.")
    print()
    print("Câu hỏi mẫu:")
    print("  • Điều kiện để nhà thầu được coi là hợp lệ là gì?")
    print("  • Nghĩa vụ của nhà thầu thi công xây dựng là gì?")
    print("  • Các hành vi bị cấm trong đấu thầu gồm những gì?")
    print()
    
    while True:
        try:
            print("-" * 60)
            question = input("📝 Câu hỏi của bạn: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q', 'thoat', 'thoát']:
                print("\n👋 Cảm ơn bạn đã sử dụng hệ thống!")
                print("   Goodbye!")
                break
            
            if not question:
                print("⚠️  Vui lòng nhập câu hỏi!")
                continue
            
            print()
            answer = rag_system.query(question)
            
            print("\n" + "=" * 60)
            print("🤖 CÂU TRẢ LỜI:")
            print("=" * 60)
            print(answer)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Đã dừng chương trình!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")
            print("Vui lòng thử lại với câu hỏi khác.")
            continue


if __name__ == "__main__":
    main()
