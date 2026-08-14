"""
Shared fixtures for the pyragcore test suite.

FakeEmbedder / FakeLLM let us test FaissRetriever and BasePipeline wiring
without needing sentence-transformers, torch, or a running Ollama server.

FakeEncoding + patch_tiktoken let Chunker tests run without network access.
tiktoken normally lazy-downloads its encoding table from a remote blob store
on first use (openaipublic.blob.core.windows.net) — this fails in offline or
network-restricted environments. That's a real gap in Chunker (no offline
fallback), tracked separately; here we just mock it out so the test suite
itself doesn't require network access.
"""
import random
import pytest

from pyragcore.interfaces.base_embedder import BaseEmbedder
from pyragcore.interfaces.base_llm import BaseLLM


class FakeEncoding:
    """
    Minimal stand-in for a tiktoken Encoding: whitespace-based word tokenizer
    with a stable vocab so encode/decode round-trip exactly. Not a faithful
    BPE simulation — only used so Chunker/RecursiveCharacterTextSplitter logic
    can be tested without hitting the network for a real encoding table.
    """

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._rev: dict[int, str] = {}

    def _get_id(self, word: str) -> int:
        if word not in self._vocab:
            idx = len(self._vocab)
            self._vocab[word] = idx
            self._rev[idx] = word
        return self._vocab[word]

    def encode(self, text: str, **kwargs) -> list[int]:
        if text == "":
            return []
        return [self._get_id(w) for w in text.split(" ")]

    def encode_ordinary(self, text: str) -> list[int]:
        return self.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return " ".join(self._rev[t] for t in tokens if t in self._rev)


@pytest.fixture(autouse=True)
def patch_tiktoken(monkeypatch):
    """Applies to every test in the suite; harmless no-op for tests that never touch tiktoken."""
    import tiktoken
    fake = FakeEncoding()
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: fake)


class FakeEmbedder(BaseEmbedder):
    """
    Deterministic fake embedder: hashes text into a fixed-size vector so the
    same text always produces the same embedding (needed for dedup-by-hash
    tests) without pulling in a real model.
    """

    def __init__(self, dim: int = 16, tag: str = "fake:v1"):
        self.dim = dim
        self.tag = tag

    def _vec(self, text: str) -> list[float]:
        rng = random.Random(hash(text) % (2**32))
        return [rng.uniform(-1, 1) for _ in range(self.dim)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._vec(text)

    def get_dimension(self) -> int:
        return self.dim

    def get_tag(self) -> str:
        return self.tag


class FakeLLM(BaseLLM):
    """Echoes back the prompt length so ask() can be tested without Ollama."""

    def generate(self, prompt: str) -> str:
        return f"fake-answer(len={len(prompt)})"

    def stream(self, prompt: str):
        return self.generate(prompt)

    def answer(self, question, context, chat_history=None, stream=False):
        return f"answered:{question}|context_len={len(context)}"


@pytest.fixture
def fake_embedder():
    return FakeEmbedder(dim=16)


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def persist_dir(tmp_path):
    d = tmp_path / "store"
    d.mkdir()
    return str(d)