from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import chromadb

from reusable_code.ai_client import embed_texts
from reusable_code.documents import chunk_document


class EvidenceIndex:
    def __init__(self, runtime_root: str | Path, collection_name: str = "lab2_evidence"):
        self.runtime_root = Path(runtime_root).resolve()
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.chroma_path = self.runtime_root / "chroma"
        self.registry_path = self.runtime_root / "index_registry.json"
        self.chunks_path = self.runtime_root / "chunks.jsonl"
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _load_registry(self) -> dict[str, Any]:
        if not self.registry_path.is_file():
            return {"documents": {}, "embedding_model": "", "chunk_count": 0}
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"documents": {}, "embedding_model": "", "chunk_count": 0}

    def _load_chunks(self) -> list[dict[str, Any]]:
        if not self.chunks_path.is_file():
            return []
        chunks = []
        for line in self.chunks_path.read_text(encoding="utf-8").splitlines():
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return chunks

    def status(self) -> dict[str, Any]:
        registry = self._load_registry()
        return {
            "indexed_documents": len(registry.get("documents", {})),
            "indexed_chunks": int(registry.get("chunk_count", 0)),
            "embedding_model": registry.get("embedding_model", ""),
            "collection_count": self.collection.count(),
        }

    def build(
        self,
        data_root: str | Path,
        manifest: list[dict[str, Any]],
        embedding_model: str = "text-embedding-3-small",
        batch_size: int = 64,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        previous = self._load_registry()
        previous_docs = previous.get("documents", {})
        current_docs = {item["doc_id"]: item for item in manifest if not item.get("read_error")}

        changed_ids = {
            doc_id
            for doc_id, item in current_docs.items()
            if doc_id not in previous_docs
            or previous_docs[doc_id].get("content_hash") != item.get("content_hash")
            or previous.get("embedding_model") != embedding_model
        }
        removed_ids = set(previous_docs) - set(current_docs)
        stale_chunk_ids: list[str] = []
        for doc_id in changed_ids | removed_ids:
            stale_chunk_ids.extend(previous_docs.get(doc_id, {}).get("chunk_ids", []))
        if stale_chunk_ids:
            self.collection.delete(ids=stale_chunk_ids)

        old_chunks = [
            chunk
            for chunk in self._load_chunks()
            if chunk.get("doc_id") not in changed_ids | removed_ids
        ]
        new_chunks: list[dict[str, Any]] = []
        total_changed = len(changed_ids)
        for number, doc_id in enumerate(sorted(changed_ids), start=1):
            new_chunks.extend(chunk_document(data_root, manifest, doc_id))
            if progress:
                progress(number, max(total_changed, 1), f"Chunked {number}/{total_changed} changed documents")

        total_batches = (len(new_chunks) + batch_size - 1) // batch_size
        for batch_number, offset in enumerate(range(0, len(new_chunks), batch_size), start=1):
            batch = new_chunks[offset : offset + batch_size]
            vectors = embed_texts([chunk["text"] for chunk in batch], model=embedding_model)
            self.collection.upsert(
                ids=[chunk["chunk_id"] for chunk in batch],
                embeddings=vectors,
                documents=[chunk["text"] for chunk in batch],
                metadatas=[
                    {
                        "doc_id": chunk["doc_id"],
                        "doc_type": chunk["doc_type"],
                        "filename": chunk["filename"],
                        "source_path": chunk["source_path"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                    }
                    for chunk in batch
                ],
            )
            if progress:
                progress(batch_number, max(total_batches, 1), f"Embedded batch {batch_number}/{total_batches}")

        all_chunks = old_chunks + new_chunks
        self.chunks_path.write_text(
            "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in all_chunks),
            encoding="utf-8",
        )
        chunks_by_doc: dict[str, list[str]] = {}
        for chunk in all_chunks:
            chunks_by_doc.setdefault(chunk["doc_id"], []).append(chunk["chunk_id"])
        registry = {
            "embedding_model": embedding_model,
            "chunk_count": len(all_chunks),
            "documents": {
                doc_id: {
                    "content_hash": item["content_hash"],
                    "chunk_ids": chunks_by_doc.get(doc_id, []),
                }
                for doc_id, item in current_docs.items()
            },
        }
        self.registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "changed_documents": len(changed_ids),
            "removed_documents": len(removed_ids),
            "embedded_chunks": len(new_chunks),
            **self.status(),
        }

    def search(
        self,
        query: str,
        embedding_model: str,
        n_results: int = 8,
        doc_type: str | None = None,
        entity: str | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Search query is required")
        candidates: dict[str, dict[str, Any]] = {}

        if self.collection.count() > 0:
            query_vector = embed_texts([query], model=embedding_model)[0]
            request: dict[str, Any] = {
                "query_embeddings": [query_vector],
                "n_results": min(max(n_results * 3, 10), self.collection.count()),
                "include": ["documents", "metadatas", "distances"],
            }
            if doc_type:
                request["where"] = {"doc_type": doc_type}
            raw = self.collection.query(**request)
            for chunk_id, text, metadata, distance in zip(
                raw.get("ids", [[]])[0],
                raw.get("documents", [[]])[0],
                raw.get("metadatas", [[]])[0],
                raw.get("distances", [[]])[0],
            ):
                if entity and entity.casefold() not in f"{text} {metadata.get('filename', '')}".casefold():
                    continue
                candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": text,
                    **metadata,
                    "distance": float(distance),
                    "semantic_score": 1.0 / (1.0 + max(float(distance), 0.0)),
                    "keyword_score": 0.0,
                }

        terms = [term for term in re.findall(r"[A-Za-z0-9]+", query.casefold()) if len(term) > 2]
        if entity:
            terms.append(entity.casefold())
        for chunk in self._load_chunks():
            if doc_type and chunk["doc_type"] != doc_type:
                continue
            haystack = f"{chunk['filename']} {chunk['text']}".casefold()
            if entity and entity.casefold() not in haystack:
                continue
            hits = sum(haystack.count(term) for term in set(terms))
            if not hits:
                continue
            score = min(1.0, hits / max(3, len(set(terms))))
            existing = candidates.setdefault(
                chunk["chunk_id"],
                {**chunk, "distance": None, "semantic_score": 0.0, "keyword_score": 0.0},
            )
            existing["keyword_score"] = max(existing.get("keyword_score", 0.0), score)

        ranked = []
        for result in candidates.values():
            result["score"] = round(
                0.72 * result.get("semantic_score", 0.0)
                + 0.28 * result.get("keyword_score", 0.0),
                6,
            )
            ranked.append(result)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        for rank, item in enumerate(ranked[:n_results], start=1):
            item["rank"] = rank
        return ranked[:n_results]
