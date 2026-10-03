"""
Script kiểm tra xem hệ thống chatbot triết học đã sẵn sàng chưa
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Kiểm tra các thư viện cần thiết"""
    print("📦 Kiểm tra thư viện...")
    
    required_packages = [
        'flask',
        'pandas',
        'sentence_transformers',
        'chromadb',
        'PyPDF2',
        'docx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'docx':
                __import__('docx')
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - CHƯA CÀI ĐẶT")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Cần cài đặt: pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ Tất cả thư viện đã được cài đặt\n")
    return True


def check_env_file():
    """Kiểm tra file .env"""
    print("🔐 Kiểm tra file .env...")
    
    env_file = Path('.env')
    
    if not env_file.exists():
        print("  ❌ File .env không tồn tại")
        print("  💡 Chạy: Copy-Item .env.example .env")
        return False
    
    # Đọc file .env
    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Kiểm tra có API key nào không
    api_keys = [
        'GEMINI_API_KEY',
        'OPENAI_API_KEY',
        'TOGETHER_API_KEY',
        'OLLAMA_BASE_URL',
        'VLLM_BASE_URL'
    ]
    
    found_key = False
    for key in api_keys:
        if key in content and 'your_' not in content.split(key)[1].split('\n')[0]:
            print(f"  ✅ Tìm thấy {key}")
            found_key = True
            break
    
    if not found_key:
        print("  ⚠️  Chưa cấu hình API key")
        print("  💡 Mở file .env và thêm API key của bạn")
        print("  💡 Lấy Gemini API key miễn phí tại: https://makersuite.google.com/app/apikey")
        return False
    
    print("✅ File .env đã được cấu hình\n")
    return True


def check_data():
    """Kiểm tra dữ liệu"""
    print("📊 Kiểm tra dữ liệu...")
    
    data_folder = Path('data')
    
    if not data_folder.exists():
        print("  ⚠️  Thư mục data không tồn tại")
        print("  💡 Sẽ tự động tạo khi chạy serve.py")
        return True
    
    csv_files = list(data_folder.glob('*.csv'))
    
    if not csv_files:
        print("  ⚠️  Chưa có file CSV nào")
        print("  💡 Chạy một trong hai lệnh sau:")
        print("     - python create_sample_data.py (dữ liệu mẫu)")
        print("     - python insert_data/process_philosophy_documents.py --input_folder philosophy_books")
        return False
    
    print(f"  ✅ Tìm thấy {len(csv_files)} file CSV:")
    for csv_file in csv_files:
        print(f"     - {csv_file.name}")
    
    print("✅ Dữ liệu đã sẵn sàng\n")
    return True


def check_chromadb():
    """Kiểm tra ChromaDB"""
    print("🗄️  Kiểm tra ChromaDB...")
    
    chroma_folder = Path('chroma_db')
    
    if not chroma_folder.exists():
        print("  ⚠️  Chưa có ChromaDB")
        print("  💡 Sẽ tự động tạo khi chạy serve.py lần đầu")
        return True
    
    print("  ✅ Thư mục ChromaDB đã tồn tại")
    
    # Thử đếm số documents
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = client.list_collections()
        
        if collections:
            print(f"  ✅ Có {len(collections)} collection:")
            for col in collections:
                try:
                    count = col.count()
                    print(f"     - {col.name}: {count} documents")
                except:
                    print(f"     - {col.name}")
        else:
            print("  ⚠️  Chưa có collection nào")
            print("  💡 Sẽ tự động tạo khi chạy serve.py")
    except Exception as e:
        print(f"  ⚠️  Không thể kiểm tra chi tiết: {e}")
    
    print("✅ ChromaDB đã sẵn sàng\n")
    return True


def main():
    print("=" * 60)
    print("🔍 KIỂM TRA HỆ THỐNG CHATBOT TRIẾT HỌC")
    print("=" * 60)
    print()
    
    results = []
    
    # Kiểm tra thư viện
    results.append(check_requirements())
    
    # Kiểm tra .env
    results.append(check_env_file())
    
    # Kiểm tra dữ liệu
    results.append(check_data())
    
    # Kiểm tra ChromaDB
    results.append(check_chromadb())
    
    # Tổng kết
    print("=" * 60)
    
    if all(results):
        print("🎉 HỆ THỐNG ĐÃ SẴN SÀNG!")
        print()
        print("Bạn có thể chạy chatbot bằng lệnh:")
        print()
        print("  python serve.py --mode online --model_name gemini --model_version gemini-1.5-flash")
        print()
        print("Sau đó mở file index.html trong trình duyệt để test!")
    else:
        print("⚠️  CÒN MỘT SỐ VẤN ĐỀ CẦN KHẮC PHỤC")
        print()
        print("Vui lòng làm theo hướng dẫn phía trên và chạy lại script này.")
    
    print("=" * 60)
    
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
