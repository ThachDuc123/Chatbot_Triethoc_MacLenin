"""Quickly verify MCQ items exist in ChromaDB and can be retrieved by query.

This does not call any LLM. It only tests vector search + simple heuristics.
"""

from __future__ import annotations

import re

from rag.core import RAG
from llms.llms import LLMs


def main() -> None:
    # Dummy LLM (we won't generate). But RAG requires llm instance.
    # We'll use offline onnx placeholder if available; instead, construct a minimal stub.
    class _StubLLM:
        def generate_content(self, prompt):
            raise RuntimeError("LLM not used in this verifier")

    llm = _StubLLM()
    rag = RAG(
        type="chromadb",
        embeddingName="Alibaba-NLP/gte-multilingual-base",
        embedding_backend="sentence_transformers",
        llm=llm,
    )

    queries = [
        "Câu 3: Trong xã hội có giai cấp, triết học",
        "Triết học có chức năng cơ bản nào",
        "Đáp án đúng là gì",
    ]

    for q in queries:
        hits = rag.vector_search(q, limit=3)
        print("=" * 60)
        print("Query:", q)
        for i, h in enumerate(hits, start=1):
            doc = (h.get("combined_information") or "")
            # show a short snippet
            doc1 = re.sub(r"\s+", " ", doc)[:220]
            print(f"{i}. score={h.get('score'):.3f} :: {doc1}...")


if __name__ == "__main__":
    main()
