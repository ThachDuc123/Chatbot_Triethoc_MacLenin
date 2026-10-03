import chromadb

from embeddings.fastEmbed import FastEmbedding


def main():
    c = chromadb.PersistentClient(path="./chroma_db")
    col = c.get_collection("bge-small-en-v1.5__fastembed")

    embedder = FastEmbedding(name="BAAI/bge-small-en-v1.5")

    keywords = [
        "biện chứng",
        "duy vật",
        "phủ định",
        "hình thái kinh tế",
        "giá trị thặng dư",
        "lực lượng sản xuất",
        "quan hệ sản xuất",
    ]

    for kw in keywords:
        q = embedder.encode([kw])
        q = q[0] if isinstance(q, list) else q[0].tolist()
        r = col.query(query_embeddings=[q], n_results=3)
        print("\nKW:", kw)
        print("IDs:", r["ids"][0])
        docs = r["documents"][0]
        print("doc0:", (docs[0][:220] if docs and docs[0] else ""))


if __name__ == "__main__":
    main()
