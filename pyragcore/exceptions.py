# pyragcore/exceptions.py
class BotRagException(Exception):
    """Base exception for BotRag"""
    pass

class FileNotSupportedException(BotRagException):
    """Raised when file format is not supported"""
    pass

class EmbeddingException(BotRagException):
    """Raised when embedding fails"""
    pass

class RetrievalException(BotRagException):
    """Raised when retrieval fails"""
    pass

class ModelNotFoundException(BotRagException):
    """Raised when ollama model is not found"""
    pass

class VectorStoreException(BotRagException):
    """Raised when vector store operations fail"""
    pass