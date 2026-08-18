from abc import ABC, abstractmethod

from pyragcore.retrieval.vector_store import FaissVectorStore
from pyragcore.retrieval.retriver import FaissRetriever
from pyragcore.llm.ollama_llm import OllamaResponder
from pyragcore.ingestion.chunker import Chunker
from pyragcore.interfaces.base_embedder import BaseEmbedder
from pyragcore.interfaces.base_vector_store import BaseVectorStore
from pyragcore.interfaces.base_llm import BaseLLM
from pyragcore.config import RagConfig
from pyragcore.utils_io.choose_model import choose_model


class BasePipeline(ABC):
    def __init__(self, persist_dir: str,
                 output_folder: str,
                 config: RagConfig = None,
                 model_name: str | None = None,
                 embedder: BaseEmbedder = None,
                 vector_store: BaseVectorStore = None,
                 llm: BaseLLM = None,
                 chunker: Chunker = None):
        self.persist_dir = persist_dir
        self.output_folder = output_folder
        self.config = config or RagConfig()
        self.model_name = model_name or self.config.model_name or choose_model()

        self.embedder = embedder or self._build_embedder()

        self.vector_store = vector_store or FaissVectorStore(
            dim=self.embedder.get_dimension(),
            persist_path=self.persist_dir,
            autosave=self.config.autosave,
            load_if_exist=self.config.load_if_exist,
            metric=self.config.metric
        )
        self.retriever = FaissRetriever(self.vector_store, self.embedder)
        if llm is not None:
            self.llm = llm
        else:
            self.llm = OllamaResponder(self.model_name,base_url=self.config.ollama_base_url)
        self.chunker = chunker or Chunker()
        self._voice = None

    def _build_embedder(self) -> BaseEmbedder:
        """
        Build the embedder configured by self.config.embedding_backend.
        Only called when no `embedder=` is explicitly injected into the pipeline.

        Supported backends:
            "ollama"                -> OllamaEmbedder
            "sentence_transformers" -> SentenceTransformerEmbedder
        """
        backend = self.config.embedding_backend

        if backend == "ollama":
            from pyragcore.embeddings.ollamaembedder import OllamaEmbedder
            return OllamaEmbedder(model_name=self.config.embedding_model)

        elif backend == "sentence_transformers":
            from pyragcore.embeddings.sentencetransformerembedder import SentenceTransformerEmbedder
            embedder_kwargs = {"model_name": self.config.embedding_model}
            if self.config.device is not None:
                embedder_kwargs["device"] = self.config.device
            return SentenceTransformerEmbedder(**embedder_kwargs)

        else:
            raise ValueError(
                f"Unknown embedding_backend '{backend}'. "
                f"Expected 'ollama' or 'sentence_transformers'."
            )

    @abstractmethod
    def ingest(self, source: str) -> str:
        pass

    def chunk_text(self, text: str, metadata: dict) -> list[dict]:
        """
        Preferred way for ingest() implementations to chunk text — applies this
        pipeline's configured chunk_size/chunk_overlap/max_tokens automatically,
        instead of each ingest() needing to read RagConfig itself.
        """
        return self.chunker.chunk(
            text,
            metadata,
            size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            max_tokens=self.config.max_tokens,
        )

    def add_to_store(self, embeddings: list[list[float]], documents: list[str],
                      metadata: list[dict], ids: list[str]) -> None:
        """
        Preferred way for ingest() implementations to add to the vector store —
        automatically tags vectors with this pipeline's embedder so mixing
        embedders across ingests on the same store is caught early.
        """
        self.vector_store.add(
            embeddings=embeddings,
            documents=documents,
            metadata=metadata,
            ids=ids,
            embedder_tag=self.embedder.get_tag(),
        )

    def ask(self, question: str, source_id: str | None = None,
            chat_history: list[dict] | None = None,
            stream: bool | None = None) -> str:
        stream = self.config.stream if stream is None else stream
        retriever_results = self.retriever.retrieve(
            question, source_id, k=self.config.top_k
        )
        context = "\n\n".join([r["document"] for r in retriever_results])
        response = self.llm.answer(question, context, chat_history, stream=stream)
        return response

    def _is_ingested(self, file_id: str) -> bool:
        return file_id in self.vector_store.list_files()

    @property
    def voice(self):
        if self._voice is None:
            from pyragcore.utils_io.voice import Voice
            self._voice = Voice()
        return self._voice

    def say(self, text: str) -> None:
        self.voice.speak(text)

    def hear(self) -> str | None:
        return self.voice.listen()

    def get_ingested_sources(self) -> list[dict]:
        seen = {}
        for meta in self.vector_store.metadatas:
            if meta and "file_id" in meta:
                file_id = meta["file_id"]
                if file_id not in seen:
                    seen[file_id] = meta.get("file_name", file_id)
        return [{"file_id": k, "file_name": v} for k, v in seen.items()]