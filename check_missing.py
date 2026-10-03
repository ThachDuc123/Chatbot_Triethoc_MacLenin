import csv

with open('data/philosophy_mcq_clean.csv', 'r', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r['qid'] == '1062':
            print(f"Câu {r['qid']}: {r['question']}")
            print(f"A: [{r['A']}]")
            print(f"B: [{r['B']}]")
            print(f"C: [{r['C']}]")
            print(f"D: [{r['D']}]")
            print(f"Answer: {r['answer']}")
            break

# Đếm câu hỏi thiếu đáp án
missing = 0
total = 0
for r in csv.DictReader(open('data/philosophy_mcq_clean.csv', 'r', encoding='utf-8')):
    total += 1
    if not r['A'] or not r['B'] or not r['C'] or not r['D']:
        missing += 1
        if missing <= 5:
            print(f"\nCâu {r['qid']} thiếu đáp án:")
            print(f"  Q: {r['question'][:60]}...")
            print(f"  A: [{r['A'][:30] if r['A'] else 'RỖNG'}]")
            print(f"  B: [{r['B'][:30] if r['B'] else 'RỖNG'}]")

print(f"\n=== Tổng kết ===")
print(f"Tổng: {total} câu")
print(f"Thiếu đáp án: {missing} câu")
print(f"Đầy đủ: {total - missing} câu")
