#pyragcore/interfaces/base_loader.py
from abc import ABC,abstractmethod

class BaseLoader(ABC):

    @abstractmethod
    def read(self,path)->dict:
        """
        Returns:
            dict:{
                "text":str,
                "metadatas":{
                    "file_id":str,
                    "file_name":str,
                    "source":str,"}
                    }
        """
        pass

