#pyragcore/embeddings/embedder.py
import torch
from pyragcore.interfaces.base_embedder import BaseEmbedder
from sentence_transformers import SentenceTransformer
from langid.langid import LanguageIdentifier,model
from pyragcore.exceptions import EmbeddingException
class Embedder(BaseEmbedder):
    def __init__(self,model_name:str="all-mpnet-base-v2",device:str= "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Embedder: Wraps a SentenceTransformer model and provides utilities for embedding text into
        vectors representation and detects language for a single input or batches.
        Usage example:
        embedder = Embedder()
        embeddings=embedder.embed([text])
        """
        self.model=SentenceTransformer(model_name,device=device)


    def embed(self,texts:list[str],batch_size:int=16)-> list[list[float]]:
        while batch_size>=1:
            try:
                embeddings=self.model.encode(texts,batch_size=batch_size)
                return embeddings.tolist()
            except RuntimeError as e:
                if batch_size==1:
                    raise EmbeddingException(f"Embedding failed: {e}")
            batch_size =batch_size//2

    def embed_one(self,text:str)->list[float]:
        try:
            embeddings=self.model.encode(text)
            return embeddings.tolist()
        except RuntimeError as e:
            raise EmbeddingException(f"Embedding failed: {e}")

    def get_dimension(self) ->int:
        return self.model.get_sentence_embedding_dimension()

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
    print(embedder.get_dimension())