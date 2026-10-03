"""Create structured MCQ dataset from the PDF "1000 Câu trắc nghiệm Triết".

Outputs a CSV that can be ingested into ChromaDB via insert_data.load_csv_to_chromadb.

The PDF is expected to contain blocks like:

Câu 3: ...
A. ...
B. ...
C. ...
D. ...
Đáp án: A ...

We try to be tolerant to formatting differences (A) / A: / A. etc.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    from PyPDF2 import PdfReader
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "PyPDF2 is required. Please install dependencies from requirements.txt"
    ) from e


@dataclass
class MCQ:
    qid: int
    question: str
    A: str
    B: str
    C: str
    D: str
    answer: str
    explanation: str = ""
    section: str = ""

    @property
    def _id(self) -> str:
        return f"mcq_triet_{self.qid}"

    def to_row(self) -> dict:
        row = asdict(self)
        row["_id"] = self._id
        # Combined info optimized for retrieval.
        row[
            "combined_information"
        ] = (
            f"Loại: Trắc nghiệm triết học; "
            f"Phần: {self.section}; "
            f"Câu {self.qid}: {self.question} "
            f"A. {self.A} B. {self.B} C. {self.C} D. {self.D} "
            f"Đáp án đúng: {self.answer}. "
            f"Giải thích: {self.explanation}"
        ).strip()
        return row


_QUESTION_RE = re.compile(
    r"(?:^|\n)\s*C\s*[âa]u\s*(\d{1,4})\s*[:\-\.]\s*(.*?)\s*(?=\n\s*A\s*[\.:\)])",
    re.IGNORECASE | re.DOTALL,
)

# Match options that start at beginning of line.
_OPTION_RE = re.compile(
    r"\n\s*([ABCD])\s*[\.:\)]\s*(.+?)(?=(?:\n\s*[ABCD]\s*[\.:\)])|(?:\n\s*Đ\s*[áa]p\s*[áa]n\s*[:\-])|$)",
    re.IGNORECASE | re.DOTALL,
)

_ANSWER_RE = re.compile(
    r"\n\s*Đ\s*[áa]p\s*[áa]n\s*[:\-]?\s*([ABCD])\b(.*?)(?=(\n\s*C\s*[âa]u\s*\d+\s*[:\-\.]\s*)|$)",
    re.IGNORECASE | re.DOTALL,
)

_SECTION_HINT_RE = re.compile(
    r"^(PHẦN|CHƯƠNG|CHƯƠNG\s+\w+|BÀI)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_text(text: str) -> str:
    # Normalize whitespace; keep Vietnamese accents.
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    # Fix broken hyphen line breaks.
    text = re.sub(r"-\n", "", text)
    # Collapse too many newlines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        pages.append(t)
    return _normalize_text("\n".join(pages))


def guess_section_near(text: str, start_idx: int) -> str:
    # Look back a bit for a section header.
    window_start = max(0, start_idx - 2000)
    window = text[window_start:start_idx]
    matches = list(_SECTION_HINT_RE.finditer(window))
    if not matches:
        return ""
    # Take the last seen header.
    return matches[-1].group(0).strip()


def parse_mcqs(text: str, max_q: Optional[int] = None) -> List[MCQ]:
    mcqs: List[MCQ] = []

    for qm in _QUESTION_RE.finditer(text):
        qid = int(qm.group(1))
        if max_q is not None and qid > max_q:
            continue

        q_start = qm.start()
        q_text = qm.group(2).strip()

        # Slice from question start to next question start to keep parsing local.
        next_q = _QUESTION_RE.search(text, qm.end())
        block = text[qm.start() : (next_q.start() if next_q else len(text))]

        options = {"A": "", "B": "", "C": "", "D": ""}
        for om in _OPTION_RE.finditer(block):
            key = om.group(1).upper()
            val = re.sub(r"\s+", " ", om.group(2)).strip()
            options[key] = val

        am = _ANSWER_RE.search(block)
        answer = am.group(1).upper() if am else ""
        explanation = ""
        if am:
            explanation = re.sub(r"\s+", " ", (am.group(2) or "")).strip()
            # Remove common trailing phrases
            explanation = re.sub(
                r"^(là\s+đáp\s+án\s+đúng\s*)", "", explanation, flags=re.IGNORECASE
            ).strip()

        # Only accept if we have at least question + 2 options.
        filled_opts = sum(1 for v in options.values() if v)
        if not q_text or filled_opts < 2:
            continue

        section = guess_section_near(text, q_start)

        mcqs.append(
            MCQ(
                qid=qid,
                question=re.sub(r"\s+", " ", q_text),
                A=options["A"],
                B=options["B"],
                C=options["C"],
                D=options["D"],
                answer=answer,
                explanation=explanation,
                section=section,
            )
        )

    # Deduplicate by qid (keep first)
    dedup = {}
    for m in mcqs:
        dedup.setdefault(m.qid, m)

    return [dedup[k] for k in sorted(dedup.keys())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        type=str,
        default=str(
            Path("philosophy_books") / "PDF - 1000 Câu trắc nghiệm Triết.pdf"
        ),
        help="Path to the PDF file.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=str(Path("data") / "philosophy_mcq_1000.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--max_q",
        type=int,
        default=None,
        help="Optional: only keep questions with id <= max_q",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    text = extract_text_from_pdf(pdf_path)
    mcqs = parse_mcqs(text, max_q=args.max_q)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([m.to_row() for m in mcqs])

    # Put _id first if present.
    cols = ["_id"] + [c for c in df.columns if c != "_id"]
    df = df[cols]

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Extracted {len(df)} questions -> {out_path}")

    # Small preview
    if len(df) > 0:
        preview = df.iloc[0].to_dict()
        print("Preview first question:")
        for k in ["_id", "qid", "section", "question", "A", "B", "C", "D", "answer"]:
            if k in preview:
                print(f"- {k}: {preview[k]}")


if __name__ == "__main__":
    main()
