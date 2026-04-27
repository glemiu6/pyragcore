from abc import ABC,abstractmethod
from pyragcore.retrieval.vector_store import VectorStore
from pyragcore.llm.responder import Responder
from pyragcore.embeddings.embedder import Embedder
from pyragcore.retrieval.retriver import Retriver
from pyragcore.utils_io.choose_model import choose_model

class BasePipeline(ABC):
    def __init__(self,persist_dir:str,output_folder:str):
        self.persist_dir=persist_dir
        self.output_folder=output_folder
        self.model_name=choose_model()
        self.embedder = Embedder()
        self.vector_store = VectorStore(dim=self.embedder.model.get_embedding_dimension(),
                                        persist_path=self.persist_dir,
                                        autosave=True,
                                        load_if_exist=True)
        self.retriever = Retriver(self.vector_store,self.embedder)
        self.responder = Responder(self.model_name)
        self._voice = None

    @abstractmethod
    def ingest(self,source:str)->str:
        pass

    def ask(self, question: str, source_id: str | None = None, chat_history: list[dict] | None = None) -> str:
        retriever_results = self.retriever.retrieve(question, source_id)
        print(f"Retrieved {len(retriever_results)} chunks")
        print(f"source_id: {source_id}")
        context = "\n\n".join([r["document"] for r in retriever_results])
        print(f"context length: {len(context)}")
        response = self.responder.answer(question, context, chat_history)
        return response

    def _is_ingested(self, file_id: str) -> bool:
        return file_id in self.vector_store.list_files()

    @property
    def voice(self):
        if self._voice is None:
            from ragcore.utils_io.voice import Voice
            self._voice = Voice()
        return self._voice

    def say(self,text:str)->None:
        self.voice.speak(text)

    def hear(self)->str|None:
        return self.voice.listen()

    def get_ingested_sources(self)->list[dict]:
        seen={}
        for meta in self.vector_store.metadatas:
            if meta and "file_id" in meta:
                file_id=meta["file_id"]
                if file_id not in seen:
                    seen[file_id]=meta.get("file_name",file_id)
        return [{"file_id":k,"file_name":v}for k,v in seen.items()]