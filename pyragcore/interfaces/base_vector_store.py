#pyragcore/interfaces/base_vector_store.py
from abc import ABC, abstractmethod

class BaseVectorStore(ABC):

    @abstractmethod
    def add(self,embeddings:list[list[float]],documents:list[str],metadata:list[dict],ids:list[str]) -> None:
        pass

    @abstractmethod
    def search(self,query_embedding:list[float],k:int=5,return_score:bool=True)->list[dict]:
        pass

    @abstractmethod
    def search_with_filter(self,query_embedding:list[float],k:int=5,where:dict|None=None)->list[dict]:
        pass

    @abstractmethod
    def persist(self)->None:
        pass

    @abstractmethod
    def list_files(self)->list[str]:
        pass

    @abstractmethod
    def delete(self,ids:list[str])->None:
        pass

    @abstractmethod
    def clear(self)->None:
        pass