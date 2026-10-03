"""
Script để xử lý giáo trình triết học từ nhiều định dạng file (PDF, DOCX, TXT)
và chuyển đổi thành CSV để đưa vào ChromaDB
"""

import os
import pandas as pd
import argparse
from pathlib import Path
import re

# Import thư viện để đọc các loại file khác nhau
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PyPDF2 không được cài đặt. Không thể đọc file PDF.")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️ python-docx không được cài đặt. Không thể đọc file DOCX.")

try:
    import win32com.client
    import pythoncom
    DOC_AVAILABLE = True
except ImportError:
    DOC_AVAILABLE = False
    print("⚠️ pywin32 không được cài đặt. Không thể đọc file .doc cũ.")


class PhilosophyDocumentProcessor:
    """Class xử lý giáo trình triết học từ nhiều định dạng file"""
    
    def __init__(self, chunk_size=500, overlap=50):
        """
        Args:
            chunk_size: Số ký tự tối đa cho mỗi chunk (đoạn văn bản)
            overlap: Số ký tự chồng lấp giữa các chunk để giữ ngữ cảnh
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def read_txt_file(self, file_path):
        """Đọc file TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Thử với encoding khác nếu utf-8 không được
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def read_pdf_file(self, file_path):
        """Đọc file PDF"""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 chưa được cài đặt. Chạy: pip install PyPDF2")
        
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    
    def read_docx_file(self, file_path):
        """Đọc file DOCX"""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx chưa được cài đặt. Chạy: pip install python-docx")
        
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    
    def read_doc_file(self, file_path):
        """Đọc file DOC (định dạng Word cũ) - chỉ hoạt động trên Windows"""
        if not DOC_AVAILABLE:
            raise ImportError("pywin32 chưa được cài đặt. Chạy: pip install pywin32")
        
        print("⏳ Đang mở file .doc (có thể mất vài phút)...")
        # Khởi tạo COM
        pythoncom.CoInitialize()
        try:
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            
            # Mở file
            print("   → Đang đọc nội dung...")
            doc = word.Documents.Open(str(Path(file_path).absolute()))
            text = doc.Content.Text
            
            # Đóng file
            print("   → Đang đóng file...")
            doc.Close()
            word.Quit()
            
            print(f"✅ Đã đọc xong file .doc ({len(text):,} ký tự)")
            return text
        finally:
            pythoncom.CoUninitialize()
    
    def read_document(self, file_path):
        """Đọc tài liệu dựa trên phần mở rộng file"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.txt':
            return self.read_txt_file(file_path)
        elif file_ext == '.pdf':
            return self.read_pdf_file(file_path)
        elif file_ext == '.docx':
            return self.read_docx_file(file_path)
        elif file_ext == '.doc':
            # Thử đọc file .doc
            try:
                return self.read_doc_file(file_path)
            except Exception as e:
                print(f"⚠️  Không thể đọc file .doc: {e}")
                print(f"💡 Vui lòng chuyển đổi '{Path(file_path).name}' sang .docx bằng Microsoft Word")
                raise ValueError(f"Không thể đọc file .doc: {file_path}. Vui lòng chuyển sang .docx")
        else:
            raise ValueError(f"Định dạng file không được hỗ trợ: {file_ext}")
    
    def clean_text(self, text):
        """Làm sạch văn bản"""
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text)
        # Loại bỏ ký tự đặc biệt không cần thiết
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()
    
    def split_into_chunks(self, text):
        """Chia văn bản thành các chunk nhỏ hơn với overlap"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Lấy chunk với kích thước chunk_size
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Cố gắng cắt tại dấu câu để giữ ngữ nghĩa
            if end < text_length:
                # Tìm dấu câu gần nhất để cắt
                for delimiter in ['. ', '.\n', '! ', '? ', '\n\n']:
                    last_delim = chunk.rfind(delimiter)
                    if last_delim != -1:
                        chunk = chunk[:last_delim + len(delimiter)]
                        end = start + len(chunk)
                        break
            
            chunks.append(chunk.strip())
            
            # Di chuyển start với overlap
            start = end - self.overlap
            
            # Tránh vòng lặp vô hạn
            if start >= text_length - self.overlap:
                break
        
        return chunks
    
    def extract_chapter_title(self, text):
        """Trích xuất tiêu đề chương từ văn bản"""
        # Tìm các pattern phổ biến cho tiêu đề
        patterns = [
            r'CHƯƠNG\s+[IVX\d]+[:\s]*(.+?)(?:\n|$)',
            r'Chương\s+[IVX\d]+[:\s]*(.+?)(?:\n|$)',
            r'^(.{1,100})$',  # Dòng đầu tiên nếu ngắn
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:200], re.MULTILINE)
            if match:
                title = match.group(1).strip() if match.lastindex else match.group(0).strip()
                return title[:100]  # Giới hạn độ dài tiêu đề
        
        return "Nội dung triết học"
    
    def process_file(self, file_path):
        """Xử lý một file và trả về danh sách các chunks"""
        print(f"📖 Đang xử lý: {file_path}")
        
        try:
            # Đọc nội dung
            print("   → Đang đọc nội dung...")
            raw_text = self.read_document(file_path)
            
            if not raw_text or len(raw_text.strip()) == 0:
                print(f"⚠️  File rỗng hoặc không có nội dung văn bản")
                return []
            
            print(f"   ✓ Đã đọc {len(raw_text):,} ký tự")
            
            # Làm sạch
            print("   → Đang làm sạch văn bản...")
            cleaned_text = self.clean_text(raw_text)
            
            if len(cleaned_text) < 100:
                print(f"⚠️  Văn bản quá ngắn sau khi làm sạch ({len(cleaned_text)} ký tự)")
                return []
            
            # Chia thành chunks
            print("   → Đang chia thành các chunks...")
            chunks = self.split_into_chunks(cleaned_text)
            
            if not chunks:
                print(f"⚠️  Không tạo được chunk nào")
                return []
            
            print(f"   ✓ Đã tạo {len(chunks)} chunks")
            
            # Tạo metadata cho mỗi chunk
            file_name = Path(file_path).stem
            results = []
            
            print("   → Đang tạo metadata...")
            for idx, chunk in enumerate(chunks):
                if len(chunk) < 50:  # Bỏ qua chunk quá ngắn
                    continue
                
                # Trích xuất tiêu đề từ chunk
                title = self.extract_chapter_title(chunk)
                
                results.append({
                    '_id': f"{file_name}_chunk_{idx}",
                    'title': f"{title} (Phần {idx+1})",
                    'content': chunk,
                    'source_file': file_name,
                    'chunk_index': idx,
                    'category': 'Triết học',
                    'combined_information': f"Tiêu đề: {title} (Phần {idx+1}), Danh mục: Triết học, Nội dung: {chunk}"
                })
            
            print(f"✅ Đã tạo {len(results)} chunks từ file {file_name}")
            return results
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý file: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_folder(self, folder_path, output_csv):
        """Xử lý tất cả các file trong thư mục"""
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Thư mục không tồn tại: {folder_path}")
        
        # Tìm tất cả các file được hỗ trợ
        supported_extensions = ['.txt', '.pdf', '.docx', '.doc']
        files = []
        for ext in supported_extensions:
            files.extend(folder.glob(f'*{ext}'))
        
        if not files:
            raise ValueError(f"Không tìm thấy file nào trong thư mục: {folder_path}")
        
        print(f"\n🔍 Tìm thấy {len(files)} file:")
        for f in files:
            print(f"  - {f.name}")
        
        # Xử lý tất cả các file
        all_data = []
        for file_path in files:
            try:
                data = self.process_file(str(file_path))
                all_data.extend(data)
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {file_path.name}: {e}")
                continue
        
        if not all_data:
            raise ValueError("Không có dữ liệu nào được xử lý thành công")
        
        # Lưu vào CSV
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ Đã lưu {len(df)} chunks vào file: {output_csv}")
        print(f"📊 Tổng số ký tự đã xử lý: {sum(len(d['content']) for d in all_data):,}")
        
        return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Xử lý giáo trình triết học từ PDF, DOCX, TXT thành CSV để đưa vào ChromaDB"
    )
    parser.add_argument(
        '--input_folder', 
        type=str, 
        required=True,
        help='Thư mục chứa các file giáo trình triết học'
    )
    parser.add_argument(
        '--output_csv', 
        type=str, 
        default='./data/philosophy_data.csv',
        help='Đường dẫn file CSV đầu ra (mặc định: ./data/philosophy_data.csv)'
    )
    parser.add_argument(
        '--chunk_size', 
        type=int, 
        default=500,
        help='Kích thước mỗi chunk văn bản (mặc định: 500 ký tự)'
    )
    parser.add_argument(
        '--overlap', 
        type=int, 
        default=50,
        help='Số ký tự chồng lấp giữa các chunk (mặc định: 50 ký tự)'
    )
    
    args = parser.parse_args()
    
    # Tạo thư mục output nếu chưa có
    output_dir = Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Xử lý tài liệu
    processor = PhilosophyDocumentProcessor(
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    try:
        processor.process_folder(args.input_folder, args.output_csv)
        print("\n🎉 Hoàn thành! Bạn có thể chạy serve.py để khởi động chatbot.")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
