import pytest

from pyragcore.retrieval.vector_store import FaissVectorStore
from pyragcore.retrieval.retriver import FaissRetriever
from pyragcore.exceptions import RetrievalException


class StubEmbedder:
    """Deterministic embedder mapping specific query strings to specific vectors."""

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    def embed_one(self, text: str) -> list[float]:
        return self.mapping[text]


@pytest.fixture
def populated_store(persist_dir):
    store = FaissVectorStore(dim=4, persist_path=persist_dir)
    store.add(
        embeddings=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        documents=["about cats", "about dogs", "about birds"],
        metadata=[{"file_id": "f1"}, {"file_id": "f1"}, {"file_id": "f2"}],
        ids=["id-cats", "id-dogs", "id-birds"],
    )
    return store


class TestFaissRetriever:
    def test_retrieve_without_source_id(self, populated_store):
        embedder = StubEmbedder({"tell me about cats": [1, 0, 0, 0]})
        retriever = FaissRetriever(populated_store, embedder)
        results = retriever.retrieve("tell me about cats", k=1)
        assert results[0]["document"] == "about cats"
        assert "id" in results[0]

    def test_retrieve_with_source_id_filters(self, populated_store):
        embedder = StubEmbedder({"query": [1, 0, 0, 0]})
        retriever = FaissRetriever(populated_store, embedder)
        results = retriever.retrieve("query", source_id="f2", k=5)
        assert all(r["metadata"]["file_id"] == "f2" for r in results)

    def test_retrieve_wraps_embedder_failure(self, populated_store):
        class BrokenEmbedder:
            def embed_one(self, text):
                raise RuntimeError("model exploded")

        retriever = FaissRetriever(populated_store, BrokenEmbedder())
        with pytest.raises(RetrievalException):
            retriever.retrieve("anything")

    def test_retrieve_result_shape_consistent_with_and_without_source_id(self, populated_store):
        embedder = StubEmbedder({"q": [1, 0, 0, 0]})
        retriever = FaissRetriever(populated_store, embedder)
        no_filter = retriever.retrieve("q", k=1)
        with_filter = retriever.retrieve("q", source_id="f1", k=1)
        assert set(no_filter[0].keys()) == set(with_filter[0].keys())