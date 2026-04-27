from ..embeddings.embedder import Embedder
from ..retrieval.vector_store import VectorStore
from ..exceptions import RetrievalException

class Retriver:
    def __init__(self,vector_store:VectorStore,embedder:Embedder):
        self.vector_store = vector_store
        self.embedder = embedder



    def retrieve(self,question,source_id,k=5):
        try:
            query_embedding=self.embedder.embed_one(question)

            if source_id:
                return self.vector_store.search_with_filter(query_embedding,k=k,where={"file_id":source_id})
            else:
                return self.vector_store.search(query_embedding,k=k,return_score=False)
        except Exception as e:
            raise RetrievalException(f"Retrieval failed: {e}")