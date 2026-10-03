"""Tag philosophy MCQ CSV with topic labels using keyword rules.

Why rules-based?
- Works fully offline
- Fast and deterministic
- Easy to tweak keywords without re-training

Outputs a new CSV with an added `topic` column (Vietnamese topic name).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def normalize(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t)
    return t


def score_topic(text: str, keywords: List[str]) -> int:
    t = normalize(text)
    score = 0
    for kw in keywords:
        kw_n = normalize(kw)
        # Give higher weight for exact phrase occurrences.
        if kw_n in t:
            score += 3 if " " in kw_n else 1
    return score


def pick_topic(text: str, rules: Dict) -> Tuple[str, int]:
    best_topic = ""
    best_score = 0
    for _, cfg in rules.items():
        s = score_topic(text, cfg.get("keywords", []))
        if s > best_score:
            best_score = s
            best_topic = cfg.get("topic_vi", "")
    return best_topic, best_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default="data/philosophy_mcq_1000.csv")
    parser.add_argument("--out_csv", default="data/philosophy_mcq_1000_tagged.csv")
    parser.add_argument("--rules", default="topic_rules_philosophy.json")
    parser.add_argument(
        "--min_score",
        type=int,
        default=2,
        help="Minimum keyword score to accept a topic; otherwise topic will be 'Khác'.",
    )
    args = parser.parse_args()

    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))

    df = pd.read_csv(args.in_csv)

    topics = []
    scores = []
    for _, row in df.iterrows():
        blob = " ".join(
            str(row.get(c, ""))
            for c in ["question", "A", "B", "C", "D", "combined_information"]
            if c in df.columns
        )
        topic, s = pick_topic(blob, rules)
        if s < args.min_score:
            topic = "Khác"
        topics.append(topic)
        scores.append(s)

    df["topic"] = topics
    df["topic_score"] = scores

    # Improve retrieval: include topic in combined_information
    if "combined_information" in df.columns:
        def add_topic(ci: str, topic: str) -> str:
            ci = ci or ""
            if "Chủ đề:" in ci:
                return ci
            return f"Chủ đề: {topic}; " + ci

        df["combined_information"] = [add_topic(ci, t) for ci, t in zip(df["combined_information"], df["topic"]) ]

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"Wrote tagged CSV -> {out} ({len(df)} rows)")
    print(df["topic"].value_counts().head(15))


if __name__ == "__main__":
    main()
