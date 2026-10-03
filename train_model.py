"""
Script để fine-tune mô hình LLM với dữ liệu triết học của bạn
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import torch

def create_training_data():
    """Tạo dữ liệu training từ giáo trình triết học"""
    
    # Đọc dữ liệu triết học
    df = pd.read_csv('data/philosophy_data.csv')
    
    # Tạo cặp câu hỏi - trả lời từ chunks
    training_data = []
    
    for _, row in df.iterrows():
        # Tạo instruction format
        instruction = f"""Bạn là chuyên gia triết học. Hãy giải thích nội dung sau:

{row['content'][:500]}

Giải thích chi tiết:"""
        
        response = row['content']
        
        training_data.append({
            'instruction': instruction,
            'response': response,
            'text': f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        })
    
    return Dataset.from_list(training_data)

def fine_tune_model(model_name="vinai/PhoGPT-7B5-Instruct"):
    """Fine-tune mô hình với dữ liệu triết học"""
    
    print("🔧 Đang tải mô hình...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,  # Sử dụng 8-bit để tiết kiệm RAM
        device_map="auto"
    )
    
    print("📚 Đang tạo dữ liệu training...")
    dataset = create_training_data()
    
    # Tokenize dữ liệu
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    print("🎓 Bắt đầu training...")
    training_args = TrainingArguments(
        output_dir="./models/triet_hoc_model",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        save_steps=100,
        logging_steps=10,
        fp16=True,  # Sử dụng mixed precision
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    
    print("💾 Đang lưu mô hình...")
    model.save_pretrained("./models/triet_hoc_model_final")
    tokenizer.save_pretrained("./models/triet_hoc_model_final")
    
    print("✅ Hoàn thành! Mô hình được lưu tại: ./models/triet_hoc_model_final")

if __name__ == "__main__":
    print("="*60)
    print("🎓 FINE-TUNE MÔ HÌNH TRIẾT HỌC")
    print("="*60)
    print("\n⚠️  Yêu cầu:")
    print("  - GPU với ít nhất 8GB VRAM (hoặc 16GB RAM)")
    print("  - Khoảng 2-4 giờ training")
    print("\n")
    
    try:
        fine_tune_model()
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        print("\n💡 Nếu thiếu RAM, thử:")
        print("  1. Giảm batch_size")
        print("  2. Dùng mô hình nhỏ hơn")
        print("  3. Hoặc dùng Google Colab (miễn phí GPU)")
