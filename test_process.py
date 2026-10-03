"""
Script test nhanh để xử lý giáo trình triết học
"""

import sys
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).parent))

from insert_data.process_philosophy_documents import PhilosophyDocumentProcessor

def main():
    print("🔧 Bắt đầu xử lý giáo trình...")
    
    file_path = "philosophy_books/1. Giao trinh Triet hoc Mac - Lenin 2021.docx"
    
    # Tạo processor
    processor = PhilosophyDocumentProcessor(chunk_size=500, overlap=50)
    
    try:
        # Xử lý file
        print(f"📖 Đang đọc file: {file_path}")
        data = processor.process_file(file_path)
        
        print(f"\n✅ Thành công! Đã tạo {len(data)} chunks")
        print(f"📝 Tổng ký tự: {sum(len(d['content']) for d in data):,}")
        
        # Lưu vào CSV
        import pandas as pd
        df = pd.DataFrame(data)
        
        output_file = "data/philosophy_data.csv"
        Path("data").mkdir(exist_ok=True)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"💾 Đã lưu vào: {output_file}")
        
        # Hiển thị vài chunks đầu
        print("\n📚 Xem thử 3 chunks đầu tiên:")
        for i in range(min(3, len(data))):
            print(f"\n--- Chunk {i+1}: {data[i]['title']} ---")
            print(data[i]['content'][:200] + "...")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
