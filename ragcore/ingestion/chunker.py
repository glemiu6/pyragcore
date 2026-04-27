from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional,Dict,Any

try:
    import tiktoken
except ImportError:
    tiktoken=None

class Chunker:
    def __init__(self,encoding_name:str="cl100k_base"):
        self.encoding_name=encoding_name
        if tiktoken:
            self.encoder=tiktoken.get_encoding(encoding_name)

        else:
            self.encoder=None


    def chunk(self,text:str,metadata:Dict[str,Any],size:int=600,overlap:int=150,max_tokens:Optional[int]=None,return_token_count:bool=False)-> List[str] | tuple[list[str], list[int]]:
        """
        Splits text into chunks with optional token limitation and counting
        
        :param text: The text to be chunked
        :type text: str
        :param metadata: The metadata to be used for chunking
        :type metadata: dict[str,Any]
        :param size: The size of each chunk
        :type size: int
        :param overlap: The overlap between chunks
        :type overlap: int
        :param max_tokens: The maximum number of tokens to consider from the text
        :type max_tokens: Optional[int]
        :param return_token_count: Whether to return token counts alongside chunks
        :type return_token_count: bool
        :return: A list of text chunks or a tuple of chunks and their token counts
        :rtype: List[str] | tuple[list[str], list[int]]
        """
        text=self.max_token_limiter(text,max_tokens)

        splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self.encoding_name,
            chunk_size=size,
            chunk_overlap=overlap
        )    
        chunks=splitter.split_text(text)
        result=[]
        for i,chunk in enumerate(chunks):
            meta=metadata.copy()
            meta.update({
                "chunk_id":i,
                "chunk_size":size,
                "chunk_overlap":overlap,
                "tokens":self.token_counter(chunk)
            })
            result.append({
                "chunk":chunk,
                "metadatas":meta,
            })
        if return_token_count:
            token_count = [self.token_counter(c) for c in chunks]
            return result, token_count
        
        return result

    def token_counter(self,text:str)->int:
        """
        Counts tokens using tiktoken if available , otherwise fallback to len(.split)
        
        Args:
            text (str): The text for tokenization
            

        Return: 
            The size of the encoding
            rtype: int
        """
        if self.encoder:
            return len(self.encoder.encode(text))
        return len(text.split())

    def max_token_limiter(self,text:str,max_tokens:Optional[int])->str:
        """
        Trims text to max_tokens provided
        
        :param text: The text to be trimmed
        :type text: str
        :param max_tokens: The max amount of tokens 
        :type max_tokens: Optional[int]
        :return: a sting that fits the max tokens 
        :rtype: str
        """
        if not max_tokens:
            return text

        if self.encoder:
            tokens=self.encoder.encode(text)
            tokens=tokens[:max_tokens]
            return self.encoder.decode(tokens)
        
        return " ".join(text.split()[:max_tokens])
    