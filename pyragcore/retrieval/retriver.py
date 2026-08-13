#pyragcore/retrieval/retriver.py
from pyragcore.exceptions import RetrievalException
from pyragcore.interfaces.base_embedder import BaseEmbedder
from pyragcore.interfaces.base_retriever import BaseRetriever
from pyragcore.interfaces.base_vector_store import BaseVectorStore


class FaissRetriever(BaseRetriever):
    def __init__(self,vector_store:BaseVectorStore,embedder:BaseEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder



    def retrieve(self,question:str,source_id:str|None=None,k:int=5)-> list[dict]:
        try:
            query_embedding=self.embedder.embed_one(question)
            where = {"file_id":source_id} if source_id else None
            return self.vector_store.search(query_embedding,k=k,where=where)
        except Exception as e:
            raise RetrievalException(f"Retrieval failed: {e}") from e
