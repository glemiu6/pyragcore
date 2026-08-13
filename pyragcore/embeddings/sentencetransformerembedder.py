#pyragcore/embeddings/sentencetransformerembedder.py
import torch
from pyragcore.interfaces.base_embedder import BaseEmbedder
from sentence_transformers import SentenceTransformer
from pyragcore.exceptions import EmbeddingException
from pyragcore.utils_io.logger import get_logger

logger=get_logger(__name__)
class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self,model_name:str="all-mpnet-base-v2",device:str= "cuda" if torch.cuda.is_available() else "cpu"):
        """
        SentenceTransformerEmbedder: Wraps a SentenceTransformer model and provides utilities for embedding text into
        vectors representation and detects language for a single input or batches.
        Usage example:
        embedder = SentenceTransformerEmbedder()
        embeddings=embedder.embed([text])
        """
        self.model=SentenceTransformer(model_name,device=device)


    def embed(self,texts:list[str],batch_size:int=16)-> list[list[float]]:
        while batch_size>=1:
            try:
                embeddings=self.model.encode(texts,batch_size=batch_size)
                return embeddings.tolist()
            except RuntimeError as e:
                if batch_size==1:
                    raise EmbeddingException(f"Embedding failed: {e}") from e 
                logger.warning(
                    "Embedding batch_size=%d failed (%s), retrying with %d",
                    batch_size,e,batch_size//2
                )
            batch_size =batch_size//2

    def embed_one(self,text:str)->list[float]:
        try:
            embeddings=self.model.encode(text)
            return embeddings.tolist()
        except RuntimeError as e:
            raise EmbeddingException(f"Embedding failed: {e}") from e

    def get_dimension(self) ->int:
        return self.model.get_sentence_embedding_dimension()

    def get_tag(self) ->str:
        return f"sentence-transformer:{self.model.name}"


if __name__=="__main__":
    embedder=SentenceTransformerEmbedder()
    print("Starting embedding...")
    print(embedder.embed_one("hello world"))
    print(embedder.get_dimension())