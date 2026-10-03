"""API smoke test for MCQ exact match + topic retrieval.

This avoids Windows PowerShell encoding pitfalls.
"""

from __future__ import annotations

import json
import sys
import urllib.request


def post(port: int, text: str) -> str:
    url = f"http://127.0.0.1:{port}/api/search"
    payload = [{"role": "user", "content": text}]
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(raw)
    return obj.get("content", raw)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5002

    tests = [
        "Triết học có chức năng cơ bản nào ?",
        "Câu 3: Trong xã hội có giai cấp, triết học:",
        "Cho mình câu hỏi về quy luật biện chứng",
        "Chủ đề Vật chất & Ý thức",
        # Regression: users paste chat-template artifacts sometimes.
        "Xin lỗi nếu có sai sót. Câu trả lời đúng là: B. Chủ nghĩa duy vật siêu hình trong lịch sử <|im_start|>user",
    ]

    for t in tests:
        print("=" * 60)
        print("USER:", t)
        try:
            ans = post(port, t)
            print("BOT :", ans)
        except Exception as e:
            print("ERR :", e)


if __name__ == "__main__":
    main()
