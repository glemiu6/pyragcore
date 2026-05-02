#pyragcore/interfaces/base_embedder.py
from abc import ABC,abstractmethod

class BaseEmbedder(ABC):

    @abstractmethod
    def embed(self,text:list[str])->list[list[float]]:
        """
        Embed a list of texts into vectors.
        """
        pass

    @abstractmethod
    def embed_one(self,text:str)->list[float]:
        """
        Embed a single text into vector.
        """
        pass

    @abstractmethod
    def get_dimension(self)->int:
        """
        Return the embedding dimension.
        """
        pass