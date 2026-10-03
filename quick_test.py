"""
Script test siêu nhanh - chỉ xử lý 5000 ký tự đầu
"""

from docx import Document
import pandas as pd
import re
from pathlib import Path

def quick_test():
    print("🚀 Test nhanh xử lý file docx...")
    
    # Đọc file
    file_path = "philosophy_books/1. Giao trinh Triet hoc Mac - Lenin 2021.docx"
    print(f"📖 Đang đọc: {file_path}")
    
    doc = Document(file_path)
    
    # Lấy văn bản
    text = "\n".join([p.text for p in doc.paragraphs])
    print(f"✅ Đã đọc {len(text):,} ký tự, {len(doc.paragraphs)} đoạn")
    
    # Chỉ lấy 10000 ký tự đầu để test
    text = text[:10000]
    
    # Làm sạch
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Chia thành chunks 500 ký tự
    chunk_size = 500
    chunks = []
    
    for i in range(0, len(text), chunk_size - 50):  # overlap 50
        chunk = text[i:i+chunk_size]
        if len(chunk) >= 50:
            chunks.append(chunk)
    
    print(f"✅ Đã tạo {len(chunks)} chunks")
    
    # Tạo DataFrame
    data = []
    for idx, chunk in enumerate(chunks):
        data.append({
            '_id': f'triet_hoc_{idx}',
            'title': f'Giáo trình triết học (Phần {idx+1})',
            'content': chunk,
            'source_file': '1. Giao trinh Triet hoc Mac - Lenin 2021',
            'chunk_index': idx,
            'category': 'Triết học',
            'combined_information': f'Tiêu đề: Giáo trình triết học (Phần {idx+1}), Danh mục: Triết học, Nội dung: {chunk}'
        })
    
    df = pd.DataFrame(data)
    
    # Lưu CSV
    Path("data").mkdir(exist_ok=True)
    output_file = "data/philosophy_data.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"💾 Đã lưu {len(df)} chunks vào: {output_file}")
    print(f"\n📚 Chunk đầu tiên:")
    print(df.iloc[0]['content'][:300] + "...")
    
    return output_file

if __name__ == "__main__":
    quick_test()
