#pyragcore/interfaces/base_llm.py
from abc import ABC,abstractmethod
class BaseLLM(ABC):

    @abstractmethod
    def generate(self,prompt:str)->str:
        """
        Generate a response from a prompt
        """
        pass

    @abstractmethod
    def stream(self,prompt:str):
        """
        Stream a response token by token
        """
        pass
