"""Ingest the extracted philosophy MCQ CSV into ChromaDB.

Run this as a file (instead of `python -c ...`) to avoid PowerShell stdout quirks.
"""

from __future__ import annotations

import argparse

import chromadb

import pandas as pd

from insert_data import load_csv_to_chromadb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/philosophy_mcq_1000.csv")
    parser.add_argument("--persist_dir", default="./chroma_db")
    parser.add_argument("--model_name", default="Alibaba-NLP/gte-multilingual-base")
    parser.add_argument(
        "--embedding_backend",
        default="sentence_transformers",
        choices=["fastembed", "sentence_transformers"],
        help="Use sentence_transformers to match default server embedding backend and avoid ONNX fastembed stalls on some Windows machines.",
    )
    args = parser.parse_args()

    # Verify collection & make ingestion idempotent (avoid duplicate-id errors).
    base_name = args.model_name.split("/")[-1]
    collection_name = f"{base_name}__{args.embedding_backend}"
    client = chromadb.PersistentClient(path=args.persist_dir)
    col = client.get_or_create_collection(name=collection_name)

    df = pd.read_csv(args.csv)
    ids = df["_id"].astype(str).tolist()

    # Chroma get() can retrieve by ids; filter out those that already exist.
    existing = set()
    try:
        # Chunk to avoid too-long requests.
        chunk = 200
        for i in range(0, len(ids), chunk):
            got = col.get(ids=ids[i : i + chunk])
            for x in got.get("ids", []) or []:
                existing.add(str(x))
    except Exception:
        # If get(ids=...) isn't supported by the underlying version, fall back.
        existing = set()

    to_add = df[~df["_id"].astype(str).isin(existing)]
    if len(to_add) == 0:
        print(f"All {len(df)} ids already in collection. Skipping add.")
    else:
        # Use the existing loader to embed + add
        print(f"Adding {len(to_add)} new items into `{collection_name}`...")
        # Write a temp CSV subset for the loader.
        tmp_path = "data/__tmp_mcq_to_add.csv"
        to_add.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        load_csv_to_chromadb(
            csv_path=tmp_path,
            persist_dir=args.persist_dir,
            model_name=args.model_name,
            embedding_backend=args.embedding_backend,
        )

    print(f"Collection: {collection_name}")
    print(f"Count: {col.count()}")


if __name__ == "__main__":
    main()
