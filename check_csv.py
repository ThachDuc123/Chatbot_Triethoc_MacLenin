import csv

with open('data/philosophy_mcq_clean.csv', 'r', encoding='utf-8') as f:
    reader = list(csv.DictReader(f))
    print(f'Tổng số câu: {len(reader)}')
    print()
    
    # Kiểm tra câu 306
    for r in reader:
        if r['qid'] == '306':
            print('=== Câu 306 ===')
            print(f"Câu hỏi: {r['question']}")
            print(f"A. {r['A']}")
            print(f"B. {r['B']}")
            print(f"C. {r['C']}")
            print(f"D. {r['D']}")
            print(f"Đáp án: {r['answer']}")
            print(f"Chủ đề: {r['topic']}")
            break
    
    print()
    print('=== 3 câu ngẫu nhiên ===')
    import random
    samples = random.sample(reader, 3)
    for s in samples:
        print(f"\nCâu {s['qid']}: {s['question'][:100]}...")
        print(f"  A. {s['A']}")
        print(f"  B. {s['B']}")
        print(f"  C. {s['C']}")
        print(f"  D. {s['D']}")
        print(f"  Đáp án đúng: {s['answer']}")
