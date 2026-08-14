from pyragcore.ingestion.chunker import Chunker


def make_long_text(paragraphs: int = 30) -> str:
    return " ".join(
        f"This is sentence number {i} in a fairly long paragraph about testing." for i in range(paragraphs)
    )


class TestChunk:
    def test_returns_list_of_dicts_with_chunk_and_metadatas(self):
        chunker = Chunker()
        text = make_long_text()
        result = chunker.chunk(text, metadata={"file_id": "f1"})

        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert set(item.keys()) == {"chunk", "metadatas"}
            assert isinstance(item["chunk"], str)
            assert item["chunk"] != ""

    def test_metadata_is_augmented_not_replaced(self):
        chunker = Chunker()
        text = make_long_text()
        result = chunker.chunk(text, metadata={"file_id": "f1", "source": "unit-test"})

        for item in result:
            meta = item["metadatas"]
            assert meta["file_id"] == "f1"
            assert meta["source"] == "unit-test"
            assert "chunk_id" in meta
            assert "chunk_size" in meta
            assert "chunk_overlap" in meta
            assert "tokens" in meta

    def test_chunk_ids_are_sequential(self):
        chunker = Chunker()
        text = make_long_text()
        result = chunker.chunk(text, metadata={})
        ids = [item["metadatas"]["chunk_id"] for item in result]
        assert ids == list(range(len(result)))

    def test_original_metadata_dict_not_mutated(self):
        chunker = Chunker()
        text = make_long_text()
        original_meta = {"file_id": "f1"}
        chunker.chunk(text, metadata=original_meta)
        # chunk() should copy the dict per-chunk, not mutate the caller's dict
        assert original_meta == {"file_id": "f1"}

    def test_custom_size_and_overlap_are_respected(self):
        chunker = Chunker()
        text = make_long_text(50)
        small = chunker.chunk(text, metadata={}, size=100, overlap=20)
        large = chunker.chunk(text, metadata={}, size=1000, overlap=100)
        # smaller chunk size should produce more, smaller chunks
        assert len(small) > len(large)
        for item in small:
            assert item["metadatas"]["chunk_size"] == 100
        for item in large:
            assert item["metadatas"]["chunk_size"] == 1000

    def test_return_token_count_true_returns_tuple(self):
        chunker = Chunker()
        text = make_long_text()
        result, token_counts = chunker.chunk(text, metadata={}, return_token_count=True)
        assert isinstance(result, list)
        assert isinstance(token_counts, list)
        assert len(result) == len(token_counts)
        assert all(isinstance(c, int) and c > 0 for c in token_counts)

    def test_max_tokens_trims_text_before_chunking(self):
        chunker = Chunker()
        text = make_long_text(200)
        full_tokens = chunker.token_counter(text)
        trimmed = chunker.max_token_limiter(text, max_tokens=10)
        trimmed_tokens = chunker.token_counter(trimmed)

        assert trimmed_tokens <= 10
        assert trimmed_tokens < full_tokens

    def test_max_token_limiter_noop_when_none(self):
        chunker = Chunker()
        text = "some text here"
        assert chunker.max_token_limiter(text, max_tokens=None) == text

    def test_chunk_with_max_tokens_produces_smaller_output(self):
        chunker = Chunker()
        text = make_long_text(200)
        unrestricted = chunker.chunk(text, metadata={}, size=600, overlap=150)
        restricted = chunker.chunk(text, metadata={}, size=600, overlap=150, max_tokens=50)
        assert len(restricted) <= len(unrestricted)

    def test_token_counter_matches_tiktoken_when_available(self):
        chunker = Chunker()
        text = "hello world, this is a test."
        count = chunker.token_counter(text)
        assert count > 0
        if chunker.encoder is not None:
            assert count == len(chunker.encoder.encode(text))
        else:
            assert count == len(text.split())

    def test_empty_text_returns_empty_or_no_error(self):
        chunker = Chunker()
        result = chunker.chunk("", metadata={})
        assert isinstance(result, list)