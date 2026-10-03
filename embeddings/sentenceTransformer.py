import os

# SentenceTransformer embeddings run on PyTorch. Disable Transformers' TF backend
# to avoid importing TensorFlow/Keras on Windows (common source of crashes).
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from pydantic.v1 import BaseModel, Field, validator
from embeddings import BaseEmbedding, EmbeddingConfig
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedding(BaseEmbedding):
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config.name)
        self.config = config
        self.embedding_model = SentenceTransformer(self.config.name, trust_remote_code=True)

    def encode(self, text: str):
        return self.embedding_model.encode(text)
