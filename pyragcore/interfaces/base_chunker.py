#pyragcore/interfaces/base_chunker.py
from abc import ABC, abstractmethod
from typing import Any

class BaseChunker(ABC):

    @abstractmethod
    def chunk(self,text:str,metadata:dict[str,Any])->list[dict]:
        """
        Returns:
            list[dict]: [{"chunk":str,"metadatas":dict}]

        """
        pass


    @abstractmethod
    def token_counter(self,text:str)->int:
        pass