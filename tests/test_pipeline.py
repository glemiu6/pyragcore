import pytest

from pyragcore.pipeline.base_pipeline import BasePipeline
from pyragcore.retrieval.vector_store import FaissVectorStore
from pyragcore.config import RagConfig
from pyragcore.exceptions import VectorStoreException


class ConcretePipeline(BasePipeline):
    """Minimal concrete subclass so we can instantiate BasePipeline for testing."""

    def ingest(self, source: str) -> str:
        text = source
        metadata = {"file_id": "f1", "file_name": "f1.txt"}
        chunks = self.chunk_text(text, metadata)
        docs = [c["chunk"] for c in chunks]
        metas = [c["metadatas"] for c in chunks]
        ids = [f"f1_chunk_{i}" for i in range(len(chunks))]
        embeddings = self.embedder.embed(docs)
        self.add_to_store(embeddings, docs, metas, ids)
        return "f1"


def build_pipeline(persist_dir, fake_embedder, fake_llm, config=None):
    vector_store = FaissVectorStore(dim=fake_embedder.get_dimension(), persist_path=persist_dir)
    return ConcretePipeline(
        persist_dir=persist_dir,
        output_folder=persist_dir,
        config=config,
        embedder=fake_embedder,
        vector_store=vector_store,
        llm=fake_llm,
    )


class TestBasePipelineWiring:
    def test_construction_with_injected_dependencies_skips_choose_model(self, persist_dir, fake_embedder, fake_llm):
        # should not block on input() since llm is injected and model_name is irrelevant
        pipeline = build_pipeline(persist_dir, fake_embedder, fake_llm)
        assert pipeline.embedder is fake_embedder
        assert pipeline.llm is fake_llm
        assert pipeline.chunker is not None

    def test_default_config_used_when_none_provided(self, persist_dir, fake_embedder, fake_llm):
        pipeline = build_pipeline(persist_dir, fake_embedder, fake_llm)
        assert isinstance(pipeline.config, RagConfig)

    def test_ingest_uses_configured_chunk_size(self, persist_dir, fake_embedder, fake_llm):
        config = RagConfig(chunk_size=50, chunk_overlap=10, top_k=2)
        pipeline = build_pipeline(persist_dir, fake_embedder, fake_llm, config=config)
        long_text = " ".join(f"word{i}" for i in range(500))

        pipeline.ingest(long_text)

        assert pipeline.vector_store.count() > 0
        for meta in pipeline.vector_store.metadatas:
            assert meta["chunk_size"] == 50
            assert meta["chunk_overlap"] == 10

    def test_add_to_store_tags_with_embedder(self, persist_dir, fake_embedder, fake_llm):
        pipeline = build_pipeline(persist_dir, fake_embedder, fake_llm)
        pipeline.ingest("short document text about testing pipelines")
        assert pipeline.vector_store.embedder_tag == fake_embedder.get_tag()

    def test_add_to_store_raises_on_embedder_mismatch(self, persist_dir, fake_embedder, fake_llm):
        from tests.conftest import FakeEmbedder
        pipeline = build_pipeline(persist_dir, fake_embedder, fake_llm)
        pipeline.ingest("first document")

        other_embedder = FakeEmbedder(dim=fake_embedder.dim, tag="different:tag")
        pipeline.embedder = other_embedder
        with pytest.raises(VectorStoreException):
            pipeline.ingest("second document")

    def test_ask_uses_configured_top_k(self, persist_dir, fake_embedder, fake_llm):
        config = RagConfig(top_k=2, chunk_size=50, chunk_overlap=10)
        pipeline = build_pipeline(persist_dir, fake_embedder, fake_llm, config=config)
        long_text = " ".join(f"word{i}" for i in range(500))
        pipeline.ingest(long_text)

        answer = pipeline.ask("what is this about?", source_id="f1")
        assert answer.startswith("answered:")

    def test_ask_with_no_ingested_docs_still_answers(self, persist_dir, fake_embedder, fake_llm):
        pipeline = build_pipeline(persist_dir, fake_embedder, fake_llm)
        answer = pipeline.ask("anything?")
        assert answer.startswith("answered:")

    def test_is_ingested_reflects_store_state(self, persist_dir, fake_embedder, fake_llm):
        pipeline = build_pipeline(persist_dir, fake_embedder, fake_llm)
        assert pipeline._is_ingested("f1") is False
        pipeline.ingest("some content")
        assert pipeline._is_ingested("f1") is True

    def test_get_ingested_sources(self, persist_dir, fake_embedder, fake_llm):
        pipeline = build_pipeline(persist_dir, fake_embedder, fake_llm)
        pipeline.ingest("some content")
        sources = pipeline.get_ingested_sources()
        assert len(sources) == 1
        assert sources[0]["file_id"] == "f1"
        assert sources[0]["file_name"] == "f1.txt"

    def test_max_tokens_config_reaches_chunker(self, persist_dir, fake_embedder, fake_llm):
        config = RagConfig(max_tokens=20, chunk_size=600, chunk_overlap=150)
        pipeline = build_pipeline(persist_dir, fake_embedder, fake_llm, config=config)
        long_text = " ".join(f"word{i}" for i in range(500))
        chunks = pipeline.chunk_text(long_text, {"file_id": "f1"})
        # with max_tokens=20 trimming applied before chunking, total content should be small
        total_chars = sum(len(c["chunk"]) for c in chunks)
        assert total_chars < len(long_text)