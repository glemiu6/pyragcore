import numpy as np
import pytest

from pyragcore.retrieval.vector_store import FaissVectorStore
from pyragcore.exceptions import VectorStoreException


def make_store(persist_dir, dim=4, metric="l2", autosave=True, load_if_exist=True):
    return FaissVectorStore(
        dim=dim, persist_path=persist_dir, metric=metric,
        autosave=autosave, load_if_exist=load_if_exist,
    )


class TestAddAndSearch:
    def test_add_then_search_returns_expected_document(self, persist_dir):
        store = make_store(persist_dir)
        store.add(
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            documents=["doc-a", "doc-b", "doc-c"],
            metadata=[{"file_id": "f1"}, {"file_id": "f1"}, {"file_id": "f2"}],
            ids=["id-a", "id-b", "id-c"],
        )
        results = store.search([1, 0, 0, 0], k=1)
        assert len(results) == 1
        assert results[0]["document"] == "doc-a"
        assert results[0]["id"] == "id-a"
        assert "score" in results[0]

    def test_search_result_schema_consistent_with_and_without_filter(self, persist_dir):
        store = make_store(persist_dir)
        store.add(
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]],
            documents=["doc-a", "doc-b"],
            metadata=[{"file_id": "f1"}, {"file_id": "f2"}],
            ids=["id-a", "id-b"],
        )
        unfiltered = store.search([1, 0, 0, 0], k=1)
        filtered = store.search([1, 0, 0, 0], k=1, where={"file_id": "f1"})
        assert set(unfiltered[0].keys()) == set(filtered[0].keys())
        assert set(unfiltered[0].keys()) == {"id", "document", "metadata", "score"}

    def test_search_with_where_filters_correctly(self, persist_dir):
        store = make_store(persist_dir)
        store.add(
            embeddings=[[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 0, 1, 0]],
            documents=["doc-a", "doc-b", "doc-c"],
            metadata=[{"file_id": "f1"}, {"file_id": "f2"}, {"file_id": "f2"}],
            ids=["id-a", "id-b", "id-c"],
        )
        results = store.search([1, 0, 0, 0], k=5, where={"file_id": "f2"})
        assert all(r["metadata"]["file_id"] == "f2" for r in results)
        assert len(results) == 2

    def test_return_score_false_omits_score(self, persist_dir):
        store = make_store(persist_dir)
        store.add(
            embeddings=[[1, 0, 0, 0]], documents=["doc-a"],
            metadata=[{}], ids=["id-a"],
        )
        results = store.search([1, 0, 0, 0], k=1, return_score=False)
        assert "score" not in results[0]

    def test_search_on_empty_store_returns_empty_list(self, persist_dir):
        store = make_store(persist_dir)
        results = store.search([1, 0, 0, 0], k=5)
        assert results == []

    def test_search_caps_results_at_k_even_with_filter(self, persist_dir):
        store = make_store(persist_dir)
        embeddings = [[1, 0, 0, 0]] * 10
        docs = [f"doc-{i}" for i in range(10)]
        meta = [{"file_id": "same"}] * 10
        ids = [f"id-{i}" for i in range(10)]
        store.add(embeddings=embeddings, documents=docs, metadata=meta, ids=ids)

        results = store.search([1, 0, 0, 0], k=3, where={"file_id": "same"})
        assert len(results) == 3

    def test_deduplication_by_content_hash(self, persist_dir):
        store = make_store(persist_dir)
        store.add(
            embeddings=[[1, 0, 0, 0]], documents=["same text"],
            metadata=[{"v": 1}], ids=["id-1"],
        )
        store.add(
            embeddings=[[0, 1, 0, 0]], documents=["same text"],
            metadata=[{"v": 2}], ids=["id-2"],
        )
        # same document text -> same hash -> should update, not duplicate
        assert store.count() == 1
        assert store.metadatas[0]["v"] == 2

    def test_caller_metadata_dict_not_mutated(self, persist_dir):
        store = make_store(persist_dir)
        original_meta = {"file_id": "f1"}
        store.add(
            embeddings=[[1, 0, 0, 0]], documents=["doc-a"],
            metadata=[original_meta], ids=["id-a"],
        )
        assert "hash" not in original_meta


class TestCosineVsL2:
    def test_cosine_normalizes_vectors_before_search(self, persist_dir):
        store = make_store(persist_dir, metric="cosine")
        # two vectors pointing the same direction but very different magnitude
        store.add(
            embeddings=[[1, 0, 0, 0], [100, 0, 0, 0], [0, 1, 0, 0]],
            documents=["small", "large", "orthogonal"],
            metadata=[{}, {}, {}],
            ids=["a", "b", "c"],
        )
        results = store.search([2, 0, 0, 0], k=2)
        top_docs = {r["document"] for r in results}
        # after normalization, "small" and "large" are the same direction as the
        # query and should both outrank "orthogonal"
        assert "orthogonal" not in top_docs

    def test_l2_metric_does_not_normalize(self, persist_dir):
        store = make_store(persist_dir, metric="l2")
        assert store.normalize is False

    def test_cosine_metric_sets_normalize_true(self, persist_dir):
        store = make_store(persist_dir, metric="cosine")
        assert store.normalize is True

    def test_invalid_metric_raises_on_index_creation(self, persist_dir):
        store = FaissVectorStore(dim=4, persist_path=persist_dir, metric="bogus")
        with pytest.raises(ValueError):
            store.create_index()


class TestPersistence:
    def test_persist_and_reload_round_trip(self, persist_dir):
        store = make_store(persist_dir)
        store.add(
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]],
            documents=["doc-a", "doc-b"],
            metadata=[{"file_id": "f1"}, {"file_id": "f2"}],
            ids=["id-a", "id-b"],
        )

        reloaded = FaissVectorStore(dim=4, persist_path=persist_dir, load_if_exist=True)
        assert reloaded.count() == 2
        assert set(reloaded.documents) == {"doc-a", "doc-b"}
        results = reloaded.search([1, 0, 0, 0], k=1)
        assert results[0]["document"] == "doc-a"

    def test_persistence_uses_json_not_pickle(self, persist_dir):
        import os
        store = make_store(persist_dir)
        store.add(embeddings=[[1, 0, 0, 0]], documents=["doc-a"], metadata=[{}], ids=["id-a"])
        assert os.path.exists(os.path.join(persist_dir, "meta.json"))
        assert not os.path.exists(os.path.join(persist_dir, "meta.pkl"))

    def test_meta_json_is_valid_json(self, persist_dir):
        import json, os
        store = make_store(persist_dir)
        store.add(embeddings=[[1, 0, 0, 0]], documents=["doc-a"], metadata=[{}], ids=["id-a"])
        with open(os.path.join(persist_dir, "meta.json")) as f:
            data = json.load(f)
        assert "documents" in data and "metadatas" in data and "ids" in data

    def test_persist_with_no_data_is_noop(self, persist_dir):
        store = make_store(persist_dir)
        store.persist()  # index is None, should not raise

    def test_load_missing_files_gives_empty_store(self, persist_dir):
        store = FaissVectorStore(dim=4, persist_path=persist_dir, load_if_exist=True)
        assert store.count() == 0
        assert store.documents == []

    def test_clear_removes_files_and_resets_state(self, persist_dir):
        import os
        store = make_store(persist_dir)
        store.add(embeddings=[[1, 0, 0, 0]], documents=["doc-a"], metadata=[{}], ids=["id-a"])
        store.clear()
        assert store.count() == 0
        assert not os.path.exists(os.path.join(persist_dir, "meta.json"))
        assert not os.path.exists(os.path.join(persist_dir, "index.faiss"))
        assert store.embedder_tag is None


class TestDelete:
    def test_delete_removes_document_and_rebuilds_index(self, persist_dir):
        store = make_store(persist_dir)
        store.add(
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            documents=["doc-a", "doc-b", "doc-c"],
            metadata=[{}, {}, {}],
            ids=["id-a", "id-b", "id-c"],
        )
        store.delete(["id-b"])
        assert store.count() == 2
        assert "doc-b" not in store.documents
        results = store.search([0, 1, 0, 0], k=3)
        assert all(r["document"] != "doc-b" for r in results)

    def test_delete_persists_when_autosave_true(self, persist_dir):
        store = make_store(persist_dir, autosave=True)
        store.add(
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]],
            documents=["doc-a", "doc-b"],
            metadata=[{}, {}],
            ids=["id-a", "id-b"],
        )
        store.delete(["id-a"])

        reloaded = FaissVectorStore(dim=4, persist_path=persist_dir, load_if_exist=True)
        assert reloaded.count() == 1
        assert reloaded.documents == ["doc-b"]

    def test_delete_empty_list_is_noop(self, persist_dir):
        store = make_store(persist_dir)
        store.add(embeddings=[[1, 0, 0, 0]], documents=["doc-a"], metadata=[{}], ids=["id-a"])
        store.delete([])
        assert store.count() == 1


class TestEmbedderTagGuard:
    def test_first_add_records_tag(self, persist_dir):
        store = make_store(persist_dir)
        store.add(
            embeddings=[[1, 0, 0, 0]], documents=["doc-a"], metadata=[{}], ids=["id-a"],
            embedder_tag="fake:v1",
        )
        assert store.embedder_tag == "fake:v1"

    def test_matching_tag_on_second_add_is_fine(self, persist_dir):
        store = make_store(persist_dir)
        store.add(
            embeddings=[[1, 0, 0, 0]], documents=["doc-a"], metadata=[{}], ids=["id-a"],
            embedder_tag="fake:v1",
        )
        store.add(
            embeddings=[[0, 1, 0, 0]], documents=["doc-b"], metadata=[{}], ids=["id-b"],
            embedder_tag="fake:v1",
        )
        assert store.count() == 2

    def test_mismatched_tag_raises(self, persist_dir):
        store = make_store(persist_dir)
        store.add(
            embeddings=[[1, 0, 0, 0]], documents=["doc-a"], metadata=[{}], ids=["id-a"],
            embedder_tag="fake:v1",
        )
        with pytest.raises(VectorStoreException, match="Embedder mismatch"):
            store.add(
                embeddings=[[0, 1, 0, 0]], documents=["doc-b"], metadata=[{}], ids=["id-b"],
                embedder_tag="ollama:different-model",
            )
        # the failed add must not have partially mutated the store
        assert store.count() == 1

    def test_no_tag_skips_check(self, persist_dir):
        store = make_store(persist_dir)
        store.add(embeddings=[[1, 0, 0, 0]], documents=["doc-a"], metadata=[{}], ids=["id-a"])
        store.add(embeddings=[[0, 1, 0, 0]], documents=["doc-b"], metadata=[{}], ids=["id-b"])
        assert store.count() == 2

    def test_tag_persists_and_reloads(self, persist_dir):
        store = make_store(persist_dir)
        store.add(
            embeddings=[[1, 0, 0, 0]], documents=["doc-a"], metadata=[{}], ids=["id-a"],
            embedder_tag="fake:v1",
        )
        reloaded = FaissVectorStore(dim=4, persist_path=persist_dir, load_if_exist=True)
        assert reloaded.embedder_tag == "fake:v1"

        with pytest.raises(VectorStoreException, match="Embedder mismatch"):
            reloaded.add(
                embeddings=[[0, 1, 0, 0]], documents=["doc-b"], metadata=[{}], ids=["id-b"],
                embedder_tag="different:tag",
            )


class TestMMRSearch:
    def test_mmr_search_returns_expected_schema(self, persist_dir):
        store = make_store(persist_dir)
        store.add(
            embeddings=[[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 0, 1, 0], [0, 0, 0.9, 0.1]],
            documents=["a", "b", "c", "d"],
            metadata=[{}, {}, {}, {}],
            ids=["id-a", "id-b", "id-c", "id-d"],
        )
        results = store.mmr_search([1, 0, 0, 0], k=2)
        assert len(results) == 2
        for r in results:
            assert set(r.keys()) == {"id", "document", "metadata", "score"}

    def test_mmr_search_on_empty_store_returns_empty(self, persist_dir):
        store = make_store(persist_dir)
        results = store.mmr_search([1, 0, 0, 0], k=3)
        assert results == []

    def test_mmr_search_diversity_prefers_different_directions(self, persist_dir):
        store = make_store(persist_dir)
        # two near-duplicates of the query, one orthogonal
        store.add(
            embeddings=[[1, 0, 0, 0], [0.99, 0.01, 0, 0], [0, 1, 0, 0]],
            documents=["dup1", "dup2", "diverse"],
            metadata=[{}, {}, {}],
            ids=["id-1", "id-2", "id-3"],
        )
        # lambda=0 -> pure diversity after the first (most relevant) pick
        results = store.mmr_search([1, 0, 0, 0], k=2, lamda_param=0.0)
        docs = [r["document"] for r in results]
        assert "diverse" in docs


class TestGetByIdAndFile:
    def test_get_by_id_found(self, persist_dir):
        store = make_store(persist_dir)
        store.add(embeddings=[[1, 0, 0, 0]], documents=["doc-a"], metadata=[{"file_id": "f1"}], ids=["id-a"])
        result = store.get_by_id("id-a")
        assert result["document"] == "doc-a"
        assert result["metadata"]["file_id"] == "f1"

    def test_get_by_id_not_found(self, persist_dir):
        store = make_store(persist_dir)
        assert store.get_by_id("nope") is None

    def test_get_by_file(self, persist_dir):
        store = make_store(persist_dir)
        store.add(
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            documents=["a", "b", "c"],
            metadata=[{"file_id": "f1"}, {"file_id": "f1"}, {"file_id": "f2"}],
            ids=["id-a", "id-b", "id-c"],
        )
        results = store.get_by_file("f1")
        assert len(results) == 2
        assert {r["document"] for r in results} == {"a", "b"}


class TestListFiles:
    def test_list_files_returns_unique_ids(self, persist_dir):
        store = make_store(persist_dir)
        store.add(
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            documents=["a", "b", "c"],
            metadata=[{"file_id": "f1"}, {"file_id": "f1"}, {"file_id": "f2"}],
            ids=["id-a", "id-b", "id-c"],
        )
        files = store.list_files()
        assert set(files) == {"f1", "f2"}