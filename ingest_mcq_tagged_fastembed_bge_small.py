"""Ingest tagged MCQ dataset into the Chroma collection used by run_chatbot.ps1 default.

Target collection:
- embedding_backend: fastembed
- embedding_model: BAAI/bge-small-en-v1.5
- collection name: bge-small-en-v1.5__fastembed

This script is idempotent: it skips IDs already present.
"""

from __future__ import annotations

import argparse

import chromadb
import pandas as pd

from insert_data.build_chromadb import load_csv_to_chromadb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/philosophy_mcq_1000_tagged.csv")
    parser.add_argument("--persist_dir", default="./chroma_db")
    parser.add_argument("--model_name", default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.persist_dir)
    collection_name = "bge-small-en-v1.5__fastembed"
    col = client.get_or_create_collection(name=collection_name)

    df = pd.read_csv(args.csv)
    ids = df["_id"].astype(str).tolist()

    existing = set()
    chunk = 200
    for i in range(0, len(ids), chunk):
        got = col.get(ids=ids[i : i + chunk])
        for x in got.get("ids", []) or []:
            existing.add(str(x))

    to_add = df[~df["_id"].astype(str).isin(existing)]

    if len(to_add) == 0:
        print(f"All {len(df)} MCQ items already present in {collection_name}. Nothing to do.")
        return

    tmp_path = "data/__tmp_mcq_tagged_to_add.csv"
    to_add.to_csv(tmp_path, index=False, encoding="utf-8-sig")

    print(f"Adding {len(to_add)} new MCQ items into {collection_name}...")
    load_csv_to_chromadb(
        csv_path=tmp_path,
        persist_dir=args.persist_dir,
        model_name=args.model_name,
        embedding_backend="fastembed",
    )

    print("Done.")
    print("New count:", col.count())


if __name__ == "__main__":
    main()
