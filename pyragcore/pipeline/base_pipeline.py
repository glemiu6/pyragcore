from abc import ABC, abstractmethod

from pyragcore.embeddings.sentencetransformerembedder import SentenceTransformerEmbedder
from pyragcore.retrieval.vector_store import FaissVectorStore
from pyragcore.retrieval.retriver import FaissRetriever
from pyragcore.llm.ollama_llm import OllamaResponder
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
                 llm: BaseLLM = None):
        self.persist_dir = persist_dir
        self.output_folder = output_folder
        self.config = config or RagConfig()
        self.model_name = model_name or self.config.model_name or choose_model()

        embedder_kwargs = {"model_name": self.config.embedding_model}
        if self.config.device is not None:
            embedder_kwargs["device"] = self.config.device
        self.embedder = embedder or SentenceTransformerEmbedder(**embedder_kwargs)

        self.vector_store = vector_store or FaissVectorStore(
            dim=self.embedder.get_dimension(),
            persist_path=self.persist_dir,
            autosave=self.config.autosave,
            load_if_exist=self.config.load_if_exist,
            metric=self.config.metric
        )
        self.retriever = FaissRetriever(self.vector_store, self.embedder)
        self.llm = llm or OllamaResponder(self.model_name)
        self._voice = None

    @abstractmethod
    def ingest(self, source: str) -> str:
        pass

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