#pyragcore/__init__.py
"""
Lazy re-exports so `import pyragcore` (or importing any single submodule)
doesn't force torch/sentence-transformers/langchain-ollama to load unless
the specific symbol that needs them is actually accessed.
"""

_LAZY_EXPORTS = {
    "BasePipeline": "pyragcore.pipeline.base_pipeline",
    "BaseEmbedder": "pyragcore.interfaces.base_embedder",
    "BaseVectorStore": "pyragcore.interfaces.base_vector_store",
    "BaseLLM": "pyragcore.interfaces.base_llm",
    "FaissVectorStore": "pyragcore.retrieval.vector_store",
    "FaissRetriever": "pyragcore.retrieval.retriver",
    "SentenceTransformerEmbedder": "pyragcore.embeddings.sentencetransformerembedder",
    "OllamaEmbedder": "pyragcore.embeddings.ollama_embedder",
    "OllamaResponder": "pyragcore.llm.ollama_llm",
    "Chunker": "pyragcore.ingestion.chunker",
    "RagConfig": "pyragcore.config",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib
        module = importlib.import_module(_LAZY_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'pyragcore' has no attribute '{name}'")

__version__ = "0.3.0"
__author__ = "Vlad Digori"