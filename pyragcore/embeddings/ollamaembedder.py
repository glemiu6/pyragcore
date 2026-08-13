#pyragcore/embeddings/ollama_embedder.py
from langchain_ollama import OllamaEmbeddings

from pyragcore.exceptions import EmbeddingException
from pyragcore.interfaces.base_embedder import BaseEmbedder
from pyragcore.utils_io.logger import get_logger

logger = get_logger(__name__)



class OllamaEmbedder(BaseEmbedder):
    def __init__(self,model_name:str="mxbai-embed-large:latest",base_url:str|None=None) -> None:
        kwargs = {"model":model_name}
        if base_url:
            kwargs["base_url"] = base_url
        self.model = OllamaEmbeddings(**kwargs)
        self._dimensions:int|None = None


    def embed(self, text: list[str]) -> list[list[float]]:
        try:
            return self.model.embed_documents(texts)
        except Exception as e:
            raise EmbeddingException(f"Ollama embeddings failed: {e}") from e


    def embed_one(self, text: str) -> list[float]:
        try:
            return self.model.embed_query(text)
        except Exception as e:
            raise EmbeddingException(f"Ollama embeddings failed: {e}") from e


    def get_dimension(self) -> int:
        if self._dimensions is None:
            probe = self.embed_one("dimensions probe")
            self._dimensions = len(probe)
        return self._dimensions


    def get_tag(self) ->str:
        return f"ollama:{self.model.model}"