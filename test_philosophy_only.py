"""test_philosophy_only.py

Script test để kiểm tra chatbot CHỈ trả lời về triết học.

Chạy script này sau khi backend đã chạy tại:
    POST http://localhost:5002/api/search
"""

import requests


def test_chatbot(question):
    """Test một câu hỏi và xem phản hồi"""
    url = "http://localhost:5002/api/search"
    
    data = [{
        "role": "user",
        "content": question
    }]
    
    try:
        response = requests.post(url, json=data, timeout=60)

        # Backend đúng sẽ trả JSON có key 'content'.
        # Nếu bạn mở nhầm URL (ví dụ /) hoặc server lỗi, Flask có thể trả HTML/404.
        if response.status_code != 200:
            preview = response.text[:300].replace("\n", " ")
            return (
                f"Lỗi HTTP {response.status_code}. "
                f"Hãy chắc chắn backend đang chạy và bạn gọi đúng endpoint {url}. "
                f"Response: {preview}"
            )

        try:
            result = response.json()
        except Exception:
            preview = response.text[:300].replace("\n", " ")
            return f"Lỗi: Response không phải JSON. Preview: {preview}"

        content = result.get("content")
        if not content:
            return f"Lỗi: JSON thiếu key 'content'. JSON nhận được: {result}"
        return content

    except requests.exceptions.ConnectionError:
        return (
            "Lỗi: Không kết nối được tới server. "
            "Hãy chạy backend trước (python serve.py ...) và kiểm tra port 5002."
        )
    except requests.exceptions.Timeout:
        return "Lỗi: Request timeout (quá 60s). Backend có thể đang bận load model."
    except Exception as e:
        return f"Lỗi không xác định: {type(e).__name__}: {e}"


def run_tests():
    print("="*60)
    print("🧪 KIỂM TRA CHATBOT CHỈ TRẢ LỜI VỀ TRIẾT HỌC")
    print("="*60)
    print("\n⚠️  Đảm bảo server đã chạy: python serve.py ...\n")
    
    # Câu hỏi VỀ triết học - NÊN trả lời
    philosophy_questions = [
        "Triết học là gì?",
        "Giải thích về chủ nghĩa duy vật biện chứng",
        "Ba quy luật cơ bản của phép biện chứng là gì?",
        "Mối quan hệ giữa vật chất và ý thức"
    ]
    
    # Câu hỏi NGOÀI triết học - KHÔNG NÊN trả lời
    non_philosophy_questions = [
        "Cách nấu phở ngon",
        "Cách lập trình Python",
        "iPhone 15 giá bao nhiêu?",
        "Công thức tính diện tích hình tròn",
        "Cách chữa bệnh cảm cúm"
    ]
    
    print("✅ TEST 1: Câu hỏi VỀ triết học (NÊN trả lời)\n")
    for q in philosophy_questions:
        print(f"❓ Câu hỏi: {q}")
        answer = test_chatbot(q)
        print(f"💬 Trả lời: {answer[:200]}...")
        print("-"*60)
        print()
    
    print("\n❌ TEST 2: Câu hỏi NGOÀI triết học (KHÔNG NÊN trả lời)\n")
    for q in non_philosophy_questions:
        print(f"❓ Câu hỏi: {q}")
        answer = test_chatbot(q)
        
        # Kiểm tra xem có từ chối không
        refused = any(word in answer.lower() for word in [
            "xin lỗi", "không thể", "ngoài phạm vi", 
            "chỉ có thể trả lời", "triết học"
        ])
        
        if refused:
            print(f"✅ ĐÃ TỪ CHỐI: {answer[:150]}...")
        else:
            print(f"⚠️  ĐÃ TRẢ LỜI (không nên): {answer[:150]}...")
        
        print("-"*60)
        print()
    
    print("\n" + "="*60)
    print("✅ Hoàn thành kiểm tra!")
    print("="*60)


if __name__ == "__main__":
    print("\n💡 Cách sử dụng:")
    print("1. Chạy server: python serve.py --mode offline --model_engine ollama --model_version llama3.2:3b")
    print("2. Mở terminal mới và chạy: python test_philosophy_only.py")
    print()

    run_tests()
