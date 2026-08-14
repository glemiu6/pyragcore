#pyragcore/config.py
from dataclasses import dataclass


@dataclass
class RagConfig:
    # LLM settings
    model_name: str|None = None
    stream: bool = True

    # Embedding settings
    embedding_model: str = "mxbai-embed-large:latest"
    embedding_backend:str = "ollama"
    device: str|None = None
    batch_size: int = 64

    # Chunking settings
    chunk_size: int = 600
    chunk_overlap: int = 150
    max_tokens: int | None = None

    # Retrieval settings
    top_k: int = 5
    metric: str = "l2"

    # Vector store settings
    autosave: bool = True
    load_if_exist: bool = True

    # Voice settings
    voice_enabled: bool = False
