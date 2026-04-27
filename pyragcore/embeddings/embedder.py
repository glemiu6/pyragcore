import torch
from sentence_transformers import SentenceTransformer
from typing import List, Any
from langid.langid import LanguageIdentifier,model
from pyragcore.exceptions import EmbeddingException
class Embedder:
    def __init__(self,model_name:str="all-mpnet-base-v2"):
        """
        Embedder: Wraps a SentenceTransformer model and provides utilities for embedding text into
        vectors representation and detects language for a single input or batches.
        Usage example:
        embedder = Embedder()
        embeddings=embedder.embed([text])
        """
        self.model=SentenceTransformer(model_name,device="cuda" if torch.cuda.is_available() else "cpu")


    def embed(self,texts:List[str],batch_size=None)-> Any | None:
        """
        Embeds a list of texts into vector representations using the SentenceTransformer model.

        This method encodes multiple texts at once and automatically reduces the batch size
        if a RuntimeError occurs.

        :param texts-> a list of input text to embed
        :param batch_size-> the batch size used for encoding.Defaults to 16.

        Returns:
            List[List[float]]: A list of embedding vectors, one per input text.

        Raises:
            RuntimeError: If embedding fails even with the smallest batch size.
        """
        if batch_size is None:
            batch_size=16
        while batch_size>=1:
            try:
                embeddings=self.model.encode(texts,batch_size=batch_size)
                return embeddings.tolist()
            except RuntimeError as e:
                if batch_size==1:
                    raise EmbeddingException(f"Embedding failed: {e}")
            batch_size =batch_size//2

    def embed_one(self,text:str)->List[float]:
        """
        Embed a single text into a vector representation
        This method embeds one piece of text into a vector representation.
        Used for query embedding.

        :param text -> the imput text to embed

        Return:
            List[float]: The embedding vector representing the imput text.
        """
        try:
            embbedings=self.model.encode(text)
            return embbedings.tolist()
        except RuntimeError as e:
            raise EmbeddingException(f"Embedding failed: {e}")

    def detect_language(self,texts:str)->(str,float):
        """
        Detect the language of a single input or batch using the langid model.

        The method returns the detected language code along with a confidence score.
        If the confidence score is below the threshold, English ('en') is returned as a fallback language.

        :param texts -> The input text whose language code should be detected.

        Return:
            Tuple[str,float]: A tuple containing the detected language code and the confidence score.
        """
        identifier=LanguageIdentifier.from_modelstring(model, norm_probs=True)
        lang, score = identifier.classify(texts)
        response_language = lang if score > 0.7 else 'en'
        return response_language,score


if __name__=="__main__":
    embedder=Embedder()
    print("Starting embedding...")
    print(embedder.embed_one("hello world"))