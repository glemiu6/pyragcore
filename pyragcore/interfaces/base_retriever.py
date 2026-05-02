#pyragcore/interfaces/base_retriever.py
from abc import ABC, abstractmethod
class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self,question:str,source_id:str|None=None,k:int=5)->list[dict]:
        """
        Retrieve relevant documents for a question
        """
        pass