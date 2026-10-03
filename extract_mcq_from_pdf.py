"""
Script để extract câu hỏi trắc nghiệm từ PDF và tạo CSV chuẩn
"""
import fitz
import re
import csv
import os

def extract_all_text_from_pdf(pdf_path):
    """Đọc toàn bộ text từ PDF"""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()
    return full_text

def parse_questions(text):
    """Parse câu hỏi từ text"""
    questions = []
    
    # Pattern để tìm câu hỏi: Câu N: ... 
    # Và các đáp án A. B. C. D.
    # Và đáp án đúng
    
    # Split theo "Câu N:"
    pattern = r'Câu\s*(\d+)\s*:\s*'
    parts = re.split(pattern, text)
    
    # parts[0] = text trước câu 1
    # parts[1] = số câu 1, parts[2] = nội dung câu 1
    # parts[3] = số câu 2, parts[4] = nội dung câu 2
    # ...
    
    i = 1
    while i < len(parts) - 1:
        try:
            qid = int(parts[i])
            content = parts[i + 1]
            
            # Tìm câu hỏi (text trước đáp án A)
            q_match = re.match(r'(.+?)(?=\nA\.\s*|\nA\s+)', content, re.DOTALL)
            if not q_match:
                i += 2
                continue
            question_text = q_match.group(1).strip()
            question_text = re.sub(r'\s+', ' ', question_text)  # Normalize whitespace
            
            # Tìm các đáp án
            def find_option(letter, content):
                # Pattern: A. hoặc A\n
                pattern = rf'\n{letter}\.\s*(.+?)(?=\n[ABCD]\.\s*|\nĐáp án|\Z)'
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    opt = match.group(1).strip()
                    opt = re.sub(r'\s+', ' ', opt)
                    return opt
                # Thử pattern khác: A\n text
                pattern2 = rf'\n{letter}\s+(.+?)(?=\n[ABCD]\s+|\nĐáp án|\Z)'
                match2 = re.search(pattern2, content, re.DOTALL)
                if match2:
                    opt = match2.group(1).strip()
                    opt = re.sub(r'\s+', ' ', opt)
                    return opt
                return ""
            
            opt_A = find_option('A', content)
            opt_B = find_option('B', content)
            opt_C = find_option('C', content)
            opt_D = find_option('D', content)
            
            # Tìm đáp án đúng
            answer_match = re.search(r'Đáp án\s*:\s*([ABCD])', content, re.IGNORECASE)
            if not answer_match:
                answer_match = re.search(r'Đáp án\s*([ABCD])\s*là', content, re.IGNORECASE)
            
            answer = answer_match.group(1).upper() if answer_match else ""
            
            # Chỉ thêm nếu có đủ thông tin
            if question_text and answer:
                questions.append({
                    'qid': qid,
                    'question': question_text,
                    'A': opt_A,
                    'B': opt_B,
                    'C': opt_C,
                    'D': opt_D,
                    'answer': answer,
                })
            
        except Exception as e:
            print(f"Lỗi parse câu {i}: {e}")
        
        i += 2
    
    return questions

def classify_topic(question_text, options):
    """Phân loại chủ đề dựa trên nội dung câu hỏi"""
    text = (question_text + " " + " ".join(options)).lower()
    
    # Keywords cho từng topic
    topics = {
        "Vật chất & Ý thức": [
            "vật chất", "ý thức", "tồn tại", "thực tại khách quan", "não bộ",
            "phản ánh", "tinh thần", "vật thể", "lênin định nghĩa", "v.i.lênin"
        ],
        "Nguồn gốc & bản chất nhận thức": [
            "nhận thức", "nguồn gốc", "cảm giác", "tri giác", "tư duy",
            "lý tính", "cảm tính", "biểu tượng", "khái niệm", "phán đoán",
            "suy lý", "chân lý", "sai lầm"
        ],
        "Các quy luật biện chứng": [
            "quy luật", "biện chứng", "mâu thuẫn", "lượng chất", "phủ định",
            "thống nhất đấu tranh", "bước nhảy", "độ", "điểm nút",
            "phủ định của phủ định", "siêu hình", "hêghen"
        ],
        "Phạm trù phép biện chứng": [
            "phạm trù", "cái chung", "cái riêng", "nguyên nhân", "kết quả",
            "tất nhiên", "ngẫu nhiên", "nội dung", "hình thức", "bản chất",
            "hiện tượng", "khả năng", "hiện thực"
        ],
        "Chân lý": [
            "chân lý", "sự thật", "đúng đắn", "khách quan", "tuyệt đối",
            "tương đối", "cụ thể", "tiêu chuẩn"
        ],
        "Xã hội, giai cấp, nhà nước": [
            "xã hội", "giai cấp", "nhà nước", "kinh tế", "chính trị",
            "quan hệ sản xuất", "lực lượng sản xuất", "kiến trúc thượng tầng",
            "cơ sở hạ tầng", "hình thái", "cách mạng", "đảng", "quần chúng",
            "lịch sử", "duy vật lịch sử"
        ],
    }
    
    max_score = 0
    best_topic = "Khác"
    
    for topic, keywords in topics.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > max_score:
            max_score = score
            best_topic = topic
    
    return best_topic, max_score

def main():
    pdf_path = "data/PDF - 1000 Câu trắc nghiệm Triết.pdf"
    output_path = "data/philosophy_mcq_clean.csv"
    
    print("📖 Đang đọc PDF...")
    text = extract_all_text_from_pdf(pdf_path)
    print(f"✅ Đã đọc {len(text)} ký tự")
    
    print("🔍 Đang parse câu hỏi...")
    questions = parse_questions(text)
    print(f"✅ Tìm thấy {len(questions)} câu hỏi")
    
    # Phân loại topic
    print("📂 Đang phân loại chủ đề...")
    for q in questions:
        options = [q['A'], q['B'], q['C'], q['D']]
        topic, score = classify_topic(q['question'], options)
        q['topic'] = topic
        q['topic_score'] = score
    
    # Thống kê
    topic_counts = {}
    for q in questions:
        t = q['topic']
        topic_counts[t] = topic_counts.get(t, 0) + 1
    
    print("\n📊 Thống kê chủ đề:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"   {topic}: {count} câu")
    
    # Ghi CSV
    print(f"\n💾 Đang ghi file {output_path}...")
    
    fieldnames = ['_id', 'qid', 'question', 'A', 'B', 'C', 'D', 'answer', 'topic', 'topic_score']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for q in questions:
            row = {
                '_id': f"mcq_{q['qid']}",
                'qid': q['qid'],
                'question': q['question'],
                'A': q['A'],
                'B': q['B'],
                'C': q['C'],
                'D': q['D'],
                'answer': q['answer'],
                'topic': q['topic'],
                'topic_score': q['topic_score'],
            }
            writer.writerow(row)
    
    print(f"✅ Đã lưu {len(questions)} câu hỏi vào {output_path}")
    
    # In vài câu mẫu
    print("\n📝 Mẫu 5 câu đầu tiên:")
    for q in questions[:5]:
        print(f"\nCâu {q['qid']}: {q['question'][:80]}...")
        print(f"   A. {q['A'][:50]}..." if len(q['A']) > 50 else f"   A. {q['A']}")
        print(f"   B. {q['B'][:50]}..." if len(q['B']) > 50 else f"   B. {q['B']}")
        print(f"   C. {q['C'][:50]}..." if len(q['C']) > 50 else f"   C. {q['C']}")
        print(f"   D. {q['D'][:50]}..." if len(q['D']) > 50 else f"   D. {q['D']}")
        print(f"   Đáp án: {q['answer']} | Chủ đề: {q['topic']}")

if __name__ == "__main__":
    main()
