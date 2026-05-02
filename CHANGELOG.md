## 0.2.1
- rename `Retriever`->`FaissRetriever`
- rename `Embedder`->`SentenceTransformerEmbedder`
- rename `VectorStore`->`FaissVectorStore`
- rename `Responder`->`OllamaLLM`

## 0.2.0
- Added backend agnostic interfaces (`BaseEmbedder`, `BaseVectorStore`, `BaseLLM`, `BaseRetriever`)
- Moved `BaseLoader` and `BaseChunker` to `interfaces/` folder
- `BasePipeline` now accepts custom `embedder` and `vector_store` implementations
- `SentenceTransformerEmbedder` now implements `BaseEmbedder` interface
- `FaissVectorStore` now implements `BaseVectorStore` interface
- `FaissRetriever` now implements `BaseRetriever` interface
- Added `get_dimension()` method to `SentenceTransformerEmbedder`
- `model_name` parameter is now optional in `BasePipeline` — prompts user if not provided

## 0.1.11
- fix dependency issues for darwin and linux systems on cuda

## 0.1.10
- resolve error for `__init__.py`

## 0.1.9
- cleaner re-export 
- corrected the name of the `FaissRetriever` class
- made the function `mmr_search` cleaner

## 0.1.8
- Added streaming support
- Type hints support
- init imports for cleaner exports


## 0.1.0
- Initial release