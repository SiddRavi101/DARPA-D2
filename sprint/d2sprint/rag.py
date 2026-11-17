"""
Clinical-protocol retrieval, ported from SecondLife.

SecondLife shipped two retrievers: FAISS + fastembed (BAAI/bge-small-en-v1.5)
in Python and a keyword BM25-ish retriever on Android. Here they are fused:

  * BM25 (Okapi, pure Python) — always available, deterministic, no model file
  * dense FAISS — used when faiss + fastembed + a cached embedding model exist
  * Reciprocal-rank fusion of the two

Corpus = SecondLife's data/protocols.json + data/tccc_protocols.json (MARCH-tagged).
Every hit carries a citation string, a SecondLife hard rule ("citations are
mandatory; '' if none, never None").
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

log = logging.getLogger("d2sprint.rag")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CORPORA = (DATA_DIR / "tccc_protocols.json", DATA_DIR / "protocols.json")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_STOP = set("""a an and are as at be by for from has have he her his i if in into is it its of on or
she that the their them then there these they this to was were will with you your not no do does
should can may must any all each per than when where which who""".split())


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


def citation(chunk: dict) -> str:
    src = chunk.get("source", "")
    page = chunk.get("page")
    return f"{src}, p.{page}" if src and page not in (None, "", 0) else (src or "")


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(map(len, docs)) / max(1, len(docs))
        self.tf = [Counter(d) for d in docs]
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, q: list[str]) -> list[float]:
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


class ProtocolRetriever:
    def __init__(self, corpora=DEFAULT_CORPORA, use_dense: Optional[bool] = None):
        self.chunks: list[dict] = []
        for p in corpora:
            p = Path(p)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                self.chunks.extend(data.get("chunks", data) if isinstance(data, dict) else data)
        if not self.chunks:
            raise RuntimeError("no protocol corpus found")
        self.bm25 = BM25([tokenize(c["text"]) for c in self.chunks])
        self._dense = None
        if use_dense is None:
            use_dense = os.environ.get("D2_DENSE_RAG", "1") == "1"
        if use_dense:
            self._dense = self._try_dense()

    def _try_dense(self):
        try:
            import faiss
            import numpy as np
            from fastembed import TextEmbedding

            emb = TextEmbedding(model_name=EMBED_MODEL, cache_dir=os.environ.get("D2_EMBED_CACHE"),
                                local_files_only=True)
            vecs = np.array(list(emb.embed([c["text"] for c in self.chunks])), dtype=np.float32)
            faiss.normalize_L2(vecs)
            index = faiss.IndexFlatIP(vecs.shape[1])
            index.add(vecs)
            log.info("dense RAG enabled (%d chunks)", len(self.chunks))
            return emb, index
        except Exception as e:
            log.info("dense RAG unavailable (%s); BM25 only", type(e).__name__)
            return None

    def retrieve(self, query: str, top_k: int = 3, march: Optional[str] = None) -> list[dict]:
        cand = list(range(len(self.chunks)))
        if march:
            tagged = [i for i in cand if self.chunks[i].get("march") == march]
            cand = tagged or cand
        bm = self.bm25.scores(tokenize(query))
        ranks: dict[int, float] = {}
        for r, i in enumerate(sorted(cand, key=lambda i: -bm[i])):
            if bm[i] > 0:
                ranks[i] = ranks.get(i, 0) + 1 / (60 + r)
        if self._dense is not None:
            import faiss
            import numpy as np

            emb, index = self._dense
            q = np.array(list(emb.embed([query])), dtype=np.float32)
            faiss.normalize_L2(q)
            _, idx = index.search(q, min(len(self.chunks), 20))
            cset = set(cand)
            r = 0
            for i in idx[0]:
                if int(i) in cset:
                    ranks[int(i)] = ranks.get(int(i), 0) + 1 / (60 + r)
                    r += 1
        best = sorted(ranks, key=lambda i: -ranks[i])[:top_k]
        return [{**self.chunks[i], "score": round(ranks[i], 5), "citation": citation(self.chunks[i])}
                for i in best]
