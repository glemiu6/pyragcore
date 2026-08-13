#pyragcore/retrieval/vector_store.py
import faiss
import json
import os
from typing import Optional
from tqdm import tqdm
from pyragcore.exceptions import VectorStoreException
from pyragcore.interfaces.base_vector_store import BaseVectorStore


class FaissVectorStore(BaseVectorStore):
    def __init__(self, dim: int, persist_path: str, index_type: str = "flat", metric: str = "l2",
                 normalizr: bool = True, autosave: bool = True, load_if_exist: bool = True):
        self.dim = dim
        self.persist_path = persist_path
        self.index_type = index_type
        self.metric = metric
        self.normalize = normalizr and metric == "cosine"
        self.autosave = autosave
        self.load_if_exist = load_if_exist

        self.index_path = os.path.join(persist_path, "index.faiss")
        self.meta_path = os.path.join(persist_path, "meta.json")

        self.index: Optional[faiss.Index] = None

        self.documents = []
        self.metadatas = []
        self.ids = []
        self.hash_map = {}
        self.embeddings = []

        self.size = 0

        # Identifies which embedder produced the vectors in this store, so mixing
        # embedders (e.g. sentence-transformers then Ollama) is caught explicitly
        # instead of silently degrading search quality. Set on first add(), verified
        # on every subsequent add() and on load().
        self.embedder_tag: str | None = None

        if self.load_if_exist and self.exists():
            self.load()

    def _check_embedder_tag(self, embedder_tag: str | None) -> None:
        """
        Raise if embedder_tag conflicts with the tag already recorded for this store.
        A None tag on either side is treated as "unknown" and skips the check, so
        callers that don't pass a tag keep working exactly as before.
        """
        if embedder_tag is None or self.embedder_tag is None:
            if embedder_tag is not None and self.embedder_tag is None:
                self.embedder_tag = embedder_tag
            return
        if embedder_tag != self.embedder_tag:
            raise VectorStoreException(
                f"Embedder mismatch: this store was built with '{self.embedder_tag}', "
                f"but got vectors from '{embedder_tag}'. Mixing embedders in one store "
                f"produces meaningless similarity scores — use a separate persist_dir "
                f"or a matching embedder."
            )

    def _normalize(self, arr):
        """L2-normalize rows of a numpy array. No-op unless cosine metric + normalize enabled."""
        import numpy as np
        if not self.normalize:
            return arr
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return arr / norms

    def load(self):
        """
        Load the FAISS index and metadata from disk.

        - Loads the FAISS index from self.index_path, or sets self.index=None if not found.
        - Loads metadata (documents, metadatas, ids) from self.meta_path, or empty if not found.
        - Rebuilds self.hash_map from metadata for deduplication.
        - Updates self.size to the number of documents.
        """
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                if self.index.ntotal > 0:
                    all_vecs = self.index.reconstruct_n(0, self.index.ntotal)
                    self.embeddings = all_vecs.tolist()
                else:
                    self.embeddings = []
            else:
                self.index = None
                self.embeddings = []

            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    self.metadatas = data.get("metadatas", [])
                    self.ids = data.get("ids", [])
                    self.embedder_tag = data.get("embedder_tag")
                self.hash_map = {
                    m.get("hash"): i
                    for i, m in enumerate(self.metadatas)
                    if m and "hash" in m
                }
            else:
                self.documents = []
                self.metadatas = []
                self.ids = []
                self.hash_map = {}

            self.size = len(self.documents)
        except Exception as e:
            raise VectorStoreException(f"Failed to load vector store: {e}") from e

    def create_index(self):
        """
        Create faiss index based on the dimension and metrics

        - Uses IndexFlatL2 for 'l2' metric
        - Uses IndexFlatIP for 'cosine' metric
        - Assigns the new index to self.index
        """
        if self.metric == "l2":
            self.index = faiss.IndexFlatL2(self.dim)
        elif self.metric == "cosine":
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            raise ValueError("metric must be 'l2' or 'cosine'")

    def exists(self):
        return os.path.exists(self.index_path) and os.path.exists(self.meta_path)

    def add(self, embeddings: list[list[float]],
            documents: list[str],
            metadata: list[dict],
            ids: list[str],
            batch_size: int = 1000,
            persist: bool = True,
            embedder_tag: str | None = None) -> None:
        import numpy as np
        import hashlib
        import uuid
        """
        Add documents and embeddings in batches with progress bar.
        Handles very large datasets without freezing.

        Args:
            embedder_tag (str | None, optional): Identifier for the embedder that produced
                these embeddings (e.g. "sentence-transformer:all-mpnet-base-v2" or
                "ollama:mxbai-embed-large"). If the store already has vectors from a
                different embedder, raises VectorStoreException instead of silently
                mixing incompatible embedding spaces. Pass None to skip the check.
        """
        self._check_embedder_tag(embedder_tag)

        if self.index is None:
            self.create_index()

        new_embeddings = []
        new_documents = []
        new_metadata = []
        new_ids = []

        try:
            for doc, emb, meta, id_ in tqdm(zip(documents, embeddings, metadata, ids),
                                            total=len(documents),
                                            desc="Preparing embeddings"):
                chunk_hash = hashlib.sha256(doc.encode('utf-8')).hexdigest()
                meta = dict(meta)  # avoid mutating caller's dict in place
                meta["hash"] = chunk_hash

                if chunk_hash in self.hash_map:
                    idx = self.hash_map[chunk_hash]
                    self.documents[idx] = doc
                    self.metadatas[idx] = meta
                    self.ids[idx] = id_
                    self.embeddings[idx] = emb
                else:
                    if not id_:
                        id_ = str(uuid.uuid4())
                    new_documents.append(doc)
                    new_metadata.append(meta)
                    new_ids.append(id_)
                    new_embeddings.append(emb)
                    self.hash_map[chunk_hash] = self.size + len(new_documents) - 1

            if not new_embeddings:
                return

            self.documents.extend(new_documents)
            self.metadatas.extend(new_metadata)
            self.ids.extend(new_ids)
            self.embeddings.extend(new_embeddings)

            for start in tqdm(range(0, len(new_embeddings), batch_size), desc="FAISS adding"):
                end = start + batch_size
                arr = np.array(new_embeddings[start:end], dtype=np.float32).reshape(-1, self.dim)
                arr = self._normalize(arr)
                self.index.add(arr)

            self.size = len(self.documents)

            if persist and self.autosave:
                os.makedirs(self.persist_path, exist_ok=True)
                faiss.write_index(self.index, self.index_path)
                with open(self.meta_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"documents": self.documents,
                         "metadatas": self.metadatas,
                         "ids": self.ids,
                         "embedder_tag": self.embedder_tag},
                        f
                    )
        except VectorStoreException:
            raise
        except Exception as e:
            raise VectorStoreException(f"Failed to add embeddings: {e}") from e

    def add_batch(self,
                  embeddings: list[list[float]],
                  documents: list[str],
                  metadatas: list[dict],
                  ids: list[str],
                  batch_size: int = 64,
                  embedder_tag: str | None = None) -> None:
        self._check_embedder_tag(embedder_tag)

        if self.index is None:
            self.create_index()

        total = len(embeddings)

        for start in range(0, total, batch_size):
            end = start + batch_size

            batch_embeddings = embeddings[start:end]
            batch_ids = ids[start:end]
            batch_metadatas = metadatas[start:end]
            batch_documents = documents[start:end]

            self.add(batch_embeddings, batch_documents, batch_metadatas, batch_ids, persist=False)
        if self.autosave:
            self.persist()

    def upsert(self,
               embeddings: list[list[float]],
               documents: list[str],
               metadata: list[dict],
               ids: list[str]) -> None:
        """
        Update existing documents if their hash exists, or insert new ones.
        Ensures both self.embeddings and FAISS index stay in sync.
        """
        import numpy as np
        import hashlib
        import uuid

        if self.index is None:
            self.create_index()

        new_embeddings = []
        new_documents = []
        new_metadata = []
        new_ids = []

        try:
            for doc, emb, meta, id_ in zip(documents, embeddings, metadata, ids):
                chunk_hash = hashlib.sha256(doc.encode('utf-8')).hexdigest()
                meta = dict(meta)
                meta["hash"] = chunk_hash

                if chunk_hash in self.hash_map:
                    idx = self.hash_map[chunk_hash]
                    self.documents[idx] = doc
                    self.metadatas[idx] = meta
                    self.ids[idx] = id_
                    self.embeddings[idx] = emb
                else:
                    if not id_:
                        id_ = str(uuid.uuid4())
                    new_documents.append(doc)
                    new_metadata.append(meta)
                    new_ids.append(id_)
                    new_embeddings.append(emb)
                    self.hash_map[chunk_hash] = self.size + len(new_documents) - 1

            if new_embeddings:
                self.documents.extend(new_documents)
                self.metadatas.extend(new_metadata)
                self.ids.extend(new_ids)
                self.embeddings.extend(new_embeddings)

                arr = np.array(new_embeddings, dtype=np.float32).reshape(-1, self.dim)
                arr = self._normalize(arr)
                self.index.add(arr)

            self.size = len(self.documents)
        except Exception as e:
            raise VectorStoreException(f"Failed to upsert embeddings: {e}") from e

    def contains(self, hash_value: str) -> bool:
        """Checks if the given hash value matches the hash stored at hash_map."""
        return hash_value in self.hash_map

    def count(self) -> int:
        """Returns the number of documents in the vector store."""
        return self.size

    def search(self,
               query_embedding: list[float],
               k: int = 5,
               where: dict | None = None,
               return_score: bool = True) -> list[dict]:
        """
        Perform a similarity search, with optional metadata filtering.

        Args:
            query_embedding (list[float]): The embedding vector of the query.
            k (int, optional): Number of nearest neighbors to retrieve. Defaults to 5.
            where (dict | None, optional): Filter conditions on document metadata (key-value pairs).
                Only documents matching all conditions are returned. Defaults to None (no filtering).
            return_score (bool, optional): Whether to include "score" in each result. Defaults to True.

        Returns:
            list[dict]: Each dict contains:
                - "id": Unique ID of the document.
                - "document": The retrieved document text.
                - "metadata": Metadata associated with the document.
                - "score" (optional): Distance/similarity from the query embedding.
        """
        import numpy as np
        try:
            if self.index is None or self.size == 0:
                return []

            new_query_embedding = np.array([query_embedding], dtype=np.float32)
            new_query_embedding = self._normalize(new_query_embedding)

            fetch_k = min(self.size, k * 20) if where else min(self.size, k)
            if fetch_k <= 0:
                return []

            distance, index = self.index.search(new_query_embedding, fetch_k)
            indices = index[0]
            distances = distance[0]

            results = []
            for idx, dist in zip(indices, distances):
                if idx < 0 or idx >= self.size:
                    continue
                if where:
                    meta = self.metadatas[idx]
                    if not all(meta.get(key) == val for key, val in where.items()):
                        continue

                result = {
                    "id": self.ids[idx],
                    "document": self.documents[idx],
                    "metadata": self.metadatas[idx],
                }
                if return_score:
                    result["score"] = float(dist)
                results.append(result)

                if len(results) >= k:
                    break

            return results
        except Exception as e:
            raise VectorStoreException(f"Failed to search embeddings: {e}") from e

    def mmr_search(self,
                   query_embedding: list[float],
                   k: int = 5,
                   lamda_param: float = 0.5) -> list[dict]:
        """
        Perform Maximal Marginal Relevance (MMR) search to balance relevance and diversity.

        Args:
            query_embedding (list[float]): The embedding vector of the query.
            k (int, optional): Number of documents to return. Defaults to 5.
            lamda_param (float, optional): Trade-off parameter between relevance and diversity
                (0 <= lambda <= 1). Higher values prioritize relevance, lower values prioritize
                diversity. Defaults to 0.5.

        Returns:
            list[dict]: Each dict contains "id", "document", "metadata", and "score"
                (cosine similarity to the query), ordered by MMR selection.
        """
        import numpy as np
        try:
            if self.index is None or self.size == 0:
                return []

            qu = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            qu = self._normalize(qu)
            distance, indices = self.index.search(qu, min(self.size, k * 4))
            candidates = [i for i in indices[0] if 0 <= i < self.size]

            if not candidates:
                return []

            candidate_emb = [self.embeddings[i] for i in candidates]
            query = np.array(query_embedding)

            def cos(a, b):
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                return float(np.dot(a, b) / denom) if denom else 0.0

            sim_query = [cos(query, e) for e in candidate_emb]

            selected_idx = []
            for _ in range(min(k, len(candidate_emb))):
                if not selected_idx:
                    idx = int(np.argmax(sim_query))
                else:
                    scores = []
                    for i, emb in enumerate(candidate_emb):
                        if i in selected_idx:
                            scores.append(-1e9)
                            continue
                        sim_q = sim_query[i]
                        sim_s = max(cos(emb, candidate_emb[j]) for j in selected_idx)
                        scores.append(lamda_param * sim_q - (1 - lamda_param) * sim_s)
                    idx = int(np.argmax(scores))
                selected_idx.append(idx)

            results = []
            for i in selected_idx:
                doc_idx = candidates[i]
                results.append({
                    "id": self.ids[doc_idx],
                    "document": self.documents[doc_idx],
                    "metadata": self.metadatas[doc_idx],
                    "score": sim_query[i],
                })
            return results
        except Exception as e:
            raise VectorStoreException(f"Failed MMR search: {e}") from e

    def get_by_id(self, id: str) -> dict | None:
        """
        Retrieve a single document by its unique ID.

        Args:
            id (str): The unique identifier of the document.

        Returns:
            dict | None: Dictionary containing "id", "document", "metadata".
                Returns None if the ID does not exist.
        """
        if id not in self.ids:
            return None
        idx = self.ids.index(id)
        return {
            "id": self.ids[idx],
            "document": self.documents[idx],
            "metadata": self.metadatas[idx]
        }

    def get_by_file(self, file_id: str) -> list[dict]:
        """
        Retrieve all documents associated with a specific file ID.

        Args:
            file_id (str): The unique identifier of the file.

        Returns:
            list[dict]: Each dict contains "id", "document", "metadata".
                Returns an empty list if no documents match the file ID.
        """
        results = []
        for i, meta in enumerate(self.metadatas):
            if meta.get("file_id") == file_id:
                results.append({
                    "id": self.ids[i],
                    "document": self.documents[i],
                    "metadata": meta
                })
        return results

    def delete(self, ids: list[str]) -> None:
        """
        Delete documents from the vector store by their IDs.

        Rebuilds the FAISS index from the remaining embeddings to keep
        search results consistent.

        Args:
            ids (list[str]): List of document IDs to remove from the store.

        Returns:
            None
        """
        import numpy as np

        if not ids:
            return

        try:
            keep_doc = []
            keep_meta = []
            keep_id = []
            keep_emb = []

            id_set = set(ids)
            for doc, meta, id_, emb in zip(self.documents, self.metadatas, self.ids, self.embeddings):
                if id_ not in id_set:
                    keep_doc.append(doc)
                    keep_meta.append(meta)
                    keep_id.append(id_)
                    keep_emb.append(emb)
            self.documents = keep_doc
            self.metadatas = keep_meta
            self.ids = keep_id
            self.embeddings = keep_emb

            self.hash_map = {
                m.get("hash"): i
                for i, m in enumerate(self.metadatas)
                if m and "hash" in m
            }

            self.create_index()
            if self.embeddings:
                all_emb = np.array(self.embeddings, dtype=np.float32)
                all_emb = self._normalize(all_emb)
                self.index.add(all_emb)
            self.size = len(self.documents)

            if self.autosave:
                self.persist()
        except Exception as e:
            raise VectorStoreException(f"Failed to delete embeddings: {e}") from e

    def persist(self) -> None:
        """
        Persist the current state of the vector store to disk.

        Saves the FAISS index to `self.index_path` and metadata
        (documents, metadatas, ids) to `self.meta_path` as JSON.

        If no index exists, the method does nothing.

        Returns:
            None
        """
        try:
            if self.index is None:
                return
            os.makedirs(self.persist_path, exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "documents": self.documents,
                        "metadatas": self.metadatas,
                        "ids": self.ids,
                        "embedder_tag": self.embedder_tag,
                    }, f
                )
        except Exception as e:
            raise VectorStoreException(f"Failed to persist vector store: {e}") from e

    def clear(self) -> None:
        """
        Clear the entire vector store: removes all stored documents, metadata,
        IDs, embeddings, resets the FAISS index and hash map, and deletes
        persisted index/metadata files from disk if they exist.

        Returns:
            None
        """
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        self.documents = []

        self.index = None
        self.hash_map = {}
        self.size = 0
        self.embedder_tag = None

        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)

    def list_files(self) -> list[str]:
        """
        List unique file identifiers stored in the vector database.

        Returns:
            list[str]: A list of unique file IDs.
        """
        files = set()
        for meta in self.metadatas:
            if meta and "file_id" in meta:
                files.add(meta["file_id"])
        return list(files)