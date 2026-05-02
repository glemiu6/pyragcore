#pyragcore/retrieval/retriver.py
from pyragcore.interfaces.base_embedder import BaseEmbedder
from pyragcore.interfaces.base_vector_store import BaseVectorStore
from pyragcore.exceptions import RetrievalException
from pyragcore.interfaces.base_retriever import BaseRetriever

class Retriever(BaseRetriever):
    def __init__(self,vector_store:BaseVectorStore,embedder:BaseEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder



    def retrieve(self,question:str,source_id:str|None=None,k:int=5)-> list[dict]:
        try:
            query_embedding=self.embedder.embed_one(question)

            if source_id:
                return self.vector_store.search_with_filter(query_embedding,k=k,where={"file_id":source_id})
            else:
                return self.vector_store.search(query_embedding,k=k,return_score=False)
        except Exception as e:
            raise RetrievalException(f"Retrieval failed: {e}")