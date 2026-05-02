# 🧠 pyragcore

A reusable, modular RAG (Retrieval-Augmented Generation) core library built on FAISS and Ollama. Use it as the foundation for any AI project that needs document ingestion, semantic search, and LLM-powered responses.

---
![PyPI](https://img.shields.io/pypi/v/pyragcore)
![Downloads](https://img.shields.io/pypi/dm/pyragcore)
![Python](https://img.shields.io/badge/python-3.13+-blue)
![License](https://img.shields.io/github/license/glemiu6/pyragcore)

---
## Features

- 🗂️ **FAISS vector store** with persistence, deduplication, and metadata filtering
- 🔢 **SentenceTransformer embeddings** with GPU support
- 🔍 **Semantic retrieval** with MMR search and metadata filtering
- 🤖 **Ollama LLM integration** for local, private inference
- 🎙️ **Voice input/output** support
- 🧱 **Abstract base classes** for building custom pipelines
- 📦 **Modular optional dependencies** — install only what you need

---
## Requirements

- Python 3.13+
- [Ollama](https://ollama.com) installed and running (for LLM features)
- NVIDIA GPU with CUDA 12.8+ (optional, falls back to CPU)

---
## Installation

```bash
pip install pyragcore          # core only (FAISS + tqdm + langchain-text-splitters)
pip install pyragcore[embeddings]  # + SentenceTransformers
pip install pyragcore[ollama]      # + Ollama LLM
pip install pyragcore[voice]       # + speech input/output
pip install pyragcore[all]         # everything
```

---

## Quick Start

```python
from pyragcore.pipeline.base_pipeline import BasePipeline
from pyragcore.embeddings.sentencetransformerembedder import SentenceTransformerEmbedder
from pyragcore.retrieval.vector_store import FaissVectorStore
from pyragcore.retrieval.retriver import FaissRetriever
from pyragcore.llm.ollama_llm import Responder


# Extend BasePipeline for your use case
class MyPipeline(BasePipeline):
    def ingest(self, source: str) -> str:
        # implement your ingestion logic
        ...


pipeline = MyPipeline(persist_dir="./memory", output_folder="./output")
source_id = pipeline.ingest("./my_document.pdf")
answer = pipeline.ask("What is this document about?", source_id=source_id)
print(answer)
```

---

## Architecture

```
pyragcore/
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
├── py.typed
├── README.md
└── pyragcore
    ├── embeddings
    │   └── sentencetransformerembedder.py
    ├── exceptions.py
    ├── ingestion
    │   └── chunker.py
    ├── interfaces
    │   ├── base_chunker.py
    │   ├── base_embedder.py
    │   ├── base_llm.py
    │   ├── base_loader.py
    │   ├── base_retriever.py
    │   └── base_vector_store.py
    ├── llm
    │   ├── prompt.py
    │   └── responder.py
    ├── pipeline
    │   └── base_pipeline.py
    ├── retrieval
    │   ├── retriver.py
    │   └── vector_store.py
    └── utils_io
        ├── choose_model.py
        ├── logger.py
        └── voice.py

```

---

## Building a Custom Pipeline

Extend `BasePipeline` and implement `ingest()`:

```python
from pyragcore.pipeline.base_pipeline import BasePipeline
from interfaces.base_loader import BaseLoader
from pyragcore.ingestion.chunker import Chunker
from tqdm import tqdm


class MyLoader(BaseLoader):
    def read(self, path) -> dict:
        # read your source and return
        return {
            "text": "...",
            "metadatas": {
                "file_id": "unique_id",
                "file_name": "my_file.txt",
                "source": path,
            }
        }


class MyPipeline(BasePipeline):
    def __init__(self, persist_dir: str, output_folder: str, model_name: str = "llama3.2"):
        super().__init__(persist_dir, output_folder, model_name)
        self.chunker = Chunker()

    def ingest(self, source: str) -> str:
        loader = MyLoader()
        content = loader.read(source)
        text = content.get("text", "")
        metadata = content.get("metadatas", {})
        source_id = metadata.get("file_id", "")

        if self._is_ingested(source_id):
            print("Already ingested, skipping...")
            return source_id

        chunks = self.chunker.chunk(text, metadata)
        documents, metadatas, ids = [], [], []

        for i, item in enumerate(chunks):
            documents.append(item["chunk"])
            metadatas.append(item["metadatas"])
            ids.append(f"{source_id}_chunk_{i}")

        BATCH_SIZE = 64
        all_embeddings = []
        for start in tqdm(range(0, len(documents), BATCH_SIZE), desc="Embedding"):
            batch = documents[start:start + BATCH_SIZE]
            all_embeddings.extend(self.embedder.embed(batch))

        self.vector_store.add(
            embeddings=all_embeddings,
            documents=documents,
            metadata=metadatas,
            ids=ids
        )
        return source_id
```

---

## FaissVectorStore

```python
from pyragcore.retrieval.vector_store import FaissVectorStore

store = FaissVectorStore(dim=768, persist_path="./memory", autosave=True)

# add documents
store.add(embeddings=[[...]], documents=["text"], metadata=[{"file_id": "abc"}], ids=["id_0"])

# search
results = store.search(query_embedding=[...], k=5)

# search with filter
results = store.search_with_filter(query_embedding=[...], k=5, where={"file_id": "abc"})

# MMR search for diversity
results = store.mmr_search(query_embedding=[...], k=5, lamda_param=0.5)

# list ingested files
files = store.list_files()
```

---

## SentenceTransformerEmbedder

```python
from pyragcore.embeddings.sentencetransformerembedder import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(model_name="all-mpnet-base-v2")

# embed multiple texts
embeddings = embedder.embed(["text one", "text two"])

# embed a single query
embedding = embedder.embed_one("what is a database?")
```

---



## PyTorch with CUDA

`pyragcore` does not pin a specific PyTorch version to stay flexible. Install the version that matches your system:

```bash
# CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CPU only
pip install torch torchvision
```

---

## Exceptions

```python
from pyragcore.exceptions import (
    BotRagException,        # base exception
    EmbeddingException,     # embedding failed
    RetrievalException,     # retrieval failed
    VectorStoreException,   # vector store error
    ModelNotFoundException, # ollama model not found
)
```

---
## Custom Backends (v0.2.0+)

You can now swap any component with your own implementation:

### Custom SentenceTransformerEmbedder
```python
from pyragcore import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        # your implementation
        ...
    
    def embed_one(self, text: str) -> list[float]:
        ...
    
    def get_dimension(self) -> int:
        return 768
if __name__=="__main__":
    rag = RagPipeline("memory", "output", embedder=MyEmbedder())
```


### Custom Vector Store
```python
from pyragcore import BaseVectorStore

class MyVectorStore(BaseVectorStore):
    def add(self, embeddings, documents, metadata, ids):
        ...
    
    def search(self, query_embedding, k=5):
        ...
if __name__ =="__main__":
    rag = RagPipeline("memory", "output", vector_store=MyVectorStore())
```
---

## Projects Built with pyragcore

- [StudyBot](https://github.com/glemiu6/StudyBot) — Chat with your documents and YouTube videos
- [Coder-Assistant](https://github.com/glemiu6/Code-Assistant) — AI assistant for your codebase *(WIP)* (Soon)

---
## Contributing
1. Fork the repo
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request
---
## License

[MIT](LICENSE)