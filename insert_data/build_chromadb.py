import pandas as pd
import ast  # To safely parse string to list
import chromadb
from chromadb.config import Settings
import os
try:
    from fastembed import TextEmbedding
except Exception:
    TextEmbedding = None
import argparse

# ---- IMPORTANT (Windows stability) ----
# sentence-transformers/transformers may try to auto-import TensorFlow/Keras.
# This project uses PyTorch; disable TF backend explicitly to avoid crashes
# and protobuf/TensorFlow version conflicts on Windows.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

class DataNotFoundError(Exception):
    def __init__(self):
        super().__init__(f"Please make sure you have valid CSV file")

def csv_exists(file_name: str) -> bool:
    """
    Check if a CSV file exists.

    Args:
        filename (str): Absolute path of the file (e.g., "C:/Users/products.csv")

    Returns:
        bool: True if file exists, False otherwise
    """
    if not file_name.endswith(".csv"):
        raise ValueError("Filename must end with .csv")
    
    return os.path.isfile(file_name)


def load_csv_to_chromadb(
    csv_path: str,
    persist_dir: str = "./chroma_db",
    model_name: str = "Alibaba-NLP/gte-multilingual-base",
    embedding_backend: str = "sentence_transformers",
):
    # Load CSV
    if csv_exists(file_name=csv_path):
        df = pd.read_csv(csv_path)
    else:
        raise DataNotFoundError

    if 'combined_infomation' in df.columns:
        df = df.drop(columns=['combined_information'])

    # Kiểm tra nếu không có cột combined_information, tạo mới
    if 'combined_information' not in df.columns:
        df['combined_information'] = df.apply(lambda row: ', '.join(f"{col}: {row[col]}" for col in df.columns if col != '_id'), axis=1)

    # Load embedding model
    if embedding_backend == "fastembed":
        if TextEmbedding is None:
            raise ImportError("fastembed is not installed. Please install fastembed or use embedding_backend='sentence_transformers'.")
        fe = TextEmbedding(name=model_name, max_length=512)

        # Generate embeddings from 'combined_information' column
        texts = df["combined_information"].tolist()
        vectors = [v.tolist() for v in fe.embed(texts)]
        df["embedding"] = vectors
    else:
        # SentenceTransformers (may download large HF models)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, trust_remote_code=True)

        # Generate embeddings from 'combined_information' column
        df['embedding'] = df['combined_information'].apply(lambda x: model.encode(x).tolist())

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=persist_dir)

    if '/' in model_name:
        base_name = model_name.split('/')[1]
    else:
        base_name = model_name
    collection_name = f"{base_name}__{embedding_backend}"
    collection = client.get_or_create_collection(name=collection_name)

    # Tạo metadata động dựa trên các cột có sẵn trong CSV
    metadatas = []
    for _, row in df.iterrows():
        metadata = {}
        # Lấy tất cả các cột trừ _id, embedding, combined_information
        for col in df.columns:
            if col not in ['_id', 'embedding', 'combined_information']:
                metadata[col] = str(row[col]) if pd.notna(row[col]) else ""
        metadatas.append(metadata)

    # Add to Chroma
    collection.add(
        ids=df['_id'].astype(str).tolist(),
        documents=df['combined_information'].tolist(),
        embeddings=df['embedding'].tolist(),
        metadatas=metadatas
    )

    print(f"{len(df)} items added to collection `{collection_name}`.")

# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments to embedding csv data to chromadb vector store")
    parser.add_argument("--csv_path", type=str, required=True, help="Declare CSV data file to embedding.")
    parser.add_argument("--persist_dir", type=str, default="./chroma_db", help="Default directory to store chromadb vector store.")
    parser.add_argument("--model_name", type=str, default="Alibaba-NLP/gte-multilingual-base", help="Choose model to embedding.")

    args = parser.parse_args()
    load_csv_to_chromadb(csv_path=args.csv_path, persist_dir=args.persist_dir, model_name=args.model_name)
