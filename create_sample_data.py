"""
Script tạo dữ liệu mẫu về triết học để test chatbot
Chạy script này nếu bạn chưa có giáo trình triết học
"""

import pandas as pd
import os

def create_sample_philosophy_data():
    """Tạo dữ liệu mẫu về triết học Mác-Lênin"""
    
    sample_data = [
        {
            "_id": "triet_hoc_001",
            "title": "Khái niệm triết học",
            "content": """Triết học là học thuyết về những quy luật vận động, phát triển chung nhất của tự nhiên, xã hội và tư duy. Triết học ra đời từ nhu cầu nhận thức thế giới một cách tổng thể, toàn diện. Câu hỏi cơ bản của triết học là mối quan hệ giữa tư duy và tồn tại, giữa vật chất và ý thức. Triết học giúp con người có cái nhìn đúng đắn về thế giới và về bản thân mình.""",
            "source_file": "giao_trinh_co_ban",
            "chunk_index": 1,
            "category": "Triết học",
            "combined_information": "Tiêu đề: Khái niệm triết học, Danh mục: Triết học, Nội dung: Triết học là học thuyết về những quy luật vận động, phát triển chung nhất của tự nhiên, xã hội và tư duy."
        },
        {
            "_id": "triet_hoc_002",
            "title": "Chủ nghĩa duy vật biện chứng",
            "content": """Chủ nghĩa duy vật biện chứng là hạt nhân lý luận của triết học Mác-Lênin. Nó khẳng định vật chất là cơ sở, là cái có trước, ý thức là cái có sau, do vật chất sinh ra. Phương pháp biện chứng xem xét sự vật, hiện tượng trong mối liên hệ, trong sự vận động, phát triển. Ba quy luật cơ bản của chủ nghĩa duy vật biện chứng là: quy luật thống nhất và đấu tranh của các mặt đối lập, quy luật chuyển hóa từ những sự thay đổi về lượng thành những sự thay đổi về chất và ngược lại, quy luật phủ định của phủ định.""",
            "source_file": "giao_trinh_co_ban",
            "chunk_index": 2,
            "category": "Triết học",
            "combined_information": "Tiêu đề: Chủ nghĩa duy vật biện chứng, Danh mục: Triết học, Nội dung: Chủ nghĩa duy vật biện chứng là hạt nhân lý luận của triết học Mác-Lênin."
        },
        {
            "_id": "triet_hoc_003",
            "title": "Vật chất và ý thức",
            "content": """Vật chất là phạm trù triết học dùng để chỉ thực tại khách quan được đem lại cho con người trong cảm giác, được cảm giác của chúng ta chép lại, chụp lại, phản ánh và tồn tại không lệ thuộc vào cảm giác. Ý thức là sản phẩm của vật chất có tổ chức cao, là chức năng của bộ óc, là sự phản ánh thế giới khách quan vào đầu óc con người. Mối quan hệ giữa vật chất và ý thức là: vật chất quyết định ý thức, ý thức phản ánh vật chất; ý thức có tính độc lập tương đối và có tác động trở lại vật chất.""",
            "source_file": "giao_trinh_co_ban",
            "chunk_index": 3,
            "category": "Triết học",
            "combined_information": "Tiêu đề: Vật chất và ý thức, Danh mục: Triết học, Nội dung: Vật chất là phạm trù triết học dùng để chỉ thực tại khách quan."
        },
        {
            "_id": "triet_hoc_004",
            "title": "Quy luật thống nhất và đấu tranh của các mặt đối lập",
            "content": """Quy luật thống nhất và đấu tranh của các mặt đối lập là quy luật cơ bản nhất của phép biện chứng duy vật. Nó chỉ ra nguồn gốc, động lực bên trong của sự vận động và phát triển. Mâu thuẫn là sự thống nhất của các mặt đối lập. Các mặt đối lập vừa thống nhất, vừa đấu tranh với nhau. Thống nhất là tạm thời, tương đối; đấu tranh là tuyệt đối. Đấu tranh của các mặt đối lập là động lực thúc đẩy sự vận động và phát triển của sự vật.""",
            "source_file": "giao_trinh_co_ban",
            "chunk_index": 4,
            "category": "Triết học",
            "combined_information": "Tiêu đề: Quy luật thống nhất và đấu tranh của các mặt đối lập, Danh mục: Triết học, Nội dung: Quy luật này chỉ ra nguồn gốc, động lực bên trong của sự vận động và phát triển."
        },
        {
            "_id": "triet_hoc_005",
            "title": "Quy luật chuyển hóa từ lượng sang chất",
            "content": """Quy luật chuyển hóa từ những sự thay đổi về lượng thành những sự thay đổi về chất và ngược lại chỉ ra trình tự, con đường của quá trình phát triển. Lượng là sự quy định về mặt số lượng, quy mô, trình độ phát triển của sự vật. Chất là sự quy định khách quan vốn có của sự vật, làm cho sự vật là nó chứ không phải là cái khác. Sự tích lũy những thay đổi về lượng đến một chừng mực nhất định sẽ làm thay đổi về chất. Khi chất mới ra đời, lại đặt ra yêu cầu có những thay đổi về lượng mới tương ứng.""",
            "source_file": "giao_trinh_co_ban",
            "chunk_index": 5,
            "category": "Triết học",
            "combined_information": "Tiêu đề: Quy luật chuyển hóa từ lượng sang chất, Danh mục: Triết học, Nội dung: Quy luật này chỉ ra trình tự, con đường của quá trình phát triển."
        },
        {
            "_id": "triet_hoc_006",
            "title": "Quy luật phủ định của phủ định",
            "content": """Quy luật phủ định của phủ định chỉ ra hướng và xu thế của quá trình phát triển. Phủ định biện chứng là sự tự phủ định, là khâu của sự phát triển và của sự liên hệ, là khâu bảo đảm sự kế tục trong sự phát triển. Sự phát triển diễn ra theo vòng xoáy: từ cái cũ đến cái mới, từ cái thấp đến cái cao, từ cái đơn giản đến cái phức tạp. Mỗi vòng của sự phát triển trải qua hai lần phủ định, lần thứ nhất là phủ định cái cũ, lần thứ hai là phủ định cái phủ định. Quá trình này vừa là tiến lên vừa là quay trở lại, nhưng ở trình độ cao hơn.""",
            "source_file": "giao_trinh_co_ban",
            "chunk_index": 6,
            "category": "Triết học",
            "combined_information": "Tiêu đề: Quy luật phủ định của phủ định, Danh mục: Triết học, Nội dung: Quy luật này chỉ ra hướng và xu thế của quá trình phát triển."
        },
        {
            "_id": "triet_hoc_007",
            "title": "Nhận thức và thực tiễn",
            "content": """Thực tiễn là toàn bộ hoạt động vật chất có mục đích, mang tính lịch sử - xã hội của con người nhằm cải tạo tự nhiên và xã hội. Thực tiễn có ba hình thức cơ bản: hoạt động sản xuất vật chất, hoạt động chính trị - xã hội và hoạt động thực nghiệm khoa học. Nhận thức là quá trình phản ánh thế giới khách quan vào đầu óc con người. Thực tiễn là cơ sở, động lực và mục đích của nhận thức, là tiêu chuẩn của chân lý. Nhận thức có vai trò to lớn đối với thực tiễn, nó định hướng cho thực tiễn và giúp thực tiễn đạt mục đích đề ra.""",
            "source_file": "giao_trinh_co_ban",
            "chunk_index": 7,
            "category": "Triết học",
            "combined_information": "Tiêu đề: Nhận thức và thực tiễn, Danh mục: Triết học, Nội dung: Thực tiễn là toàn bộ hoạt động vật chất có mục đích của con người."
        },
        {
            "_id": "triet_hoc_008",
            "title": "Chân lý và tiêu chuẩn của chân lý",
            "content": """Chân lý là sự phù hợp của tri thức với đối tượng. Chân lý có tính khách quan, không phụ thuộc vào ý muốn chủ quan của con người. Chân lý vừa tuyệt đối vừa tương đối. Chân lý tuyệt đối là tri thức đúng đắn, khách quan mà không bao giờ bị lật đổ. Chân lý tương đối là tri thức đúng đắn nhưng chưa đầy đủ, chưa cạn kiệt sự vật. Chân lý tuyệt đối và chân lý tương đối có mối quan hệ biện chứng: chân lý tuyệt đối cư trú trong chân lý tương đối, chân lý tương đối chứa đựng hạt nhân chân lý tuyệt đối. Tiêu chuẩn của chân lý là thực tiễn.""",
            "source_file": "giao_trinh_co_ban",
            "chunk_index": 8,
            "category": "Triết học",
            "combined_information": "Tiêu đề: Chân lý và tiêu chuẩn của chân lý, Danh mục: Triết học, Nội dung: Chân lý là sự phù hợp của tri thức với đối tượng."
        },
        {
            "_id": "triet_hoc_009",
            "title": "Tồn tại xã hội và ý thức xã hội",
            "content": """Tồn tại xã hội là toàn bộ đời sống vật chất của xã hội, trong đó phương thức sản xuất vật chất là nội dung cơ bản nhất. Ý thức xã hội là sự phản ánh tồn tại xã hội vào đầu óc con người. Tồn tại xã hội quyết định ý thức xã hội, nhưng ý thức xã hội có tính độc lập tương đối và có tác động trở lại tồn tại xã hội. Các hình thái ý thức xã hội gồm: ý thức chính trị, ý thức pháp quyền, đạo đức, nghệ thuật, tôn giáo, khoa học và triết học.""",
            "source_file": "giao_trinh_co_ban",
            "chunk_index": 9,
            "category": "Triết học",
            "combined_information": "Tiêu đề: Tồn tại xã hội và ý thức xã hội, Danh mục: Triết học, Nội dung: Tồn tại xã hội là toàn bộ đời sống vật chất của xã hội."
        },
        {
            "_id": "triet_hoc_010",
            "title": "Phương thức sản xuất",
            "content": """Phương thức sản xuất là sự thống nhất giữa lực lượng sản xuất và quan hệ sản xuất. Lực lượng sản xuất là tổng hợp các yếu tố vật chất và con người trong quá trình sản xuất, bao gồm tư liệu sản xuất và sức lao động. Quan hệ sản xuất là tổng hợp các mối quan hệ kinh tế giữa người với người trong quá trình sản xuất, phân phối, trao đổi và tiêu dùng sản phẩm. Lực lượng sản xuất là yếu tố động, quyết định sự phát triển của phương thức sản xuất. Khi lực lượng sản xuất phát triển đến một trình độ nhất định sẽ mâu thuẫn với quan hệ sản xuất cũ, đòi hỏi phải thay đổi quan hệ sản xuất cho phù hợp.""",
            "source_file": "giao_trinh_co_ban",
            "chunk_index": 10,
            "category": "Triết học",
            "combined_information": "Tiêu đề: Phương thức sản xuất, Danh mục: Triết học, Nội dung: Phương thức sản xuất là sự thống nhất giữa lực lượng sản xuất và quan hệ sản xuất."
        }
    ]
    
    return sample_data


def main():
    print("🔧 Đang tạo dữ liệu mẫu về triết học...")
    
    # Tạo dữ liệu
    data = create_sample_philosophy_data()
    
    # Tạo DataFrame
    df = pd.DataFrame(data)
    
    # Tạo thư mục data nếu chưa có
    os.makedirs('data', exist_ok=True)
    
    # Lưu vào CSV
    output_file = 'data/philosophy_sample_data.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"✅ Đã tạo file: {output_file}")
    print(f"📊 Số lượng chunks: {len(df)}")
    print(f"📝 Tổng số ký tự: {sum(len(item['content']) for item in data):,}")
    print("\n💡 Bây giờ bạn có thể chạy:")
    print("   python serve.py --mode online --model_name gemini --model_version gemini-1.5-flash")
    print("\n⚠️  Lưu ý: Đây chỉ là dữ liệu mẫu. Để có chatbot chất lượng cao,")
    print("   hãy sử dụng script process_philosophy_documents.py với giáo trình đầy đủ.")


if __name__ == "__main__":
    main()
