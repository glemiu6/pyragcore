#pyragcore/__init__.py
from pyragcore.pipeline.base_pipeline import BasePipeline
from pyragcore.embeddings.sentencetransformerembedder import SentenceTransformerEmbedder
from pyragcore.retrieval.vector_store import FaissVectorStore
from pyragcore.retrieval.retriver import FaissRetriever
from pyragcore.llm.ollama_llm import OllamaResponder

# interfaces
from pyragcore.interfaces.base_loader import BaseLoader
from pyragcore.interfaces.base_chunker import BaseChunker
from pyragcore.interfaces.base_embedder import BaseEmbedder
from pyragcore.interfaces.base_vector_store import BaseVectorStore
from pyragcore.interfaces.base_llm import BaseLLM
from pyragcore.interfaces.base_retriever import BaseRetriever


#configuration
from pyragcore.config import RagConfig

# exceptions
from pyragcore.exceptions import (
    BotRagException,
    EmbeddingException,
    RetrievalException,
    VectorStoreException,
    ModelNotFoundException,
)

__all__ = [
    # concrete classes
    "BasePipeline",
    "SentenceTransformerEmbedder",
    "FaissVectorStore",
    "FaissRetriever",
    "OllamaResponder",
    # interfaces
    "BaseLoader",
    "BaseChunker",
    "BaseEmbedder",
    "BaseVectorStore",
    "BaseLLM",
    "BaseRetriever",
    #configuration
    "RagConfig",
    # exceptions
    "BotRagException",
    "EmbeddingException",
    "RetrievalException",
    "VectorStoreException",
    "ModelNotFoundException",
]

__version__ = "0.2.0"
__author__ = "Vlad Digori"