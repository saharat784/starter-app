"""✏️ ส่วนที่ 3: Retrieval — ค้นหา chunk ที่เกี่ยวข้องกับคำถาม

ค่าตั้งต้นตอนยังไม่ทำ: ไม่คืน chunk ใดเลย LLM จึงไม่เห็นเอกสาร
"""
import collections
import math
import re

from core.types import Chunk
from core.utils import cosine, cosine_many
from student import settings


def _tokenize(text: str) -> list[str]:
    """Tokenize Thai and alphanumeric code/terms into tokens and n-grams."""
    text_lower = text.lower()
    tokens = []

    # 1. Alphanumeric words, codes (like BB-SAVE-26, is_prime, v3.2), and numbers
    words = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|\d+(?:\.\d+)?%?", text_lower)
    tokens.extend(words)

    # 2. Thai script segments -> generate character 2-grams and 3-grams for robust substring matching
    thai_blocks = re.findall(r"[\u0e00-\u0e7f]+", text_lower)
    for block in thai_blocks:
        if len(block) <= 4:
            tokens.append(block)
        else:
            # 2-grams and 3-grams
            for i in range(len(block) - 1):
                tokens.append(block[i:i + 2])
            for i in range(len(block) - 2):
                tokens.append(block[i:i + 3])

    return tokens


def make_query(question: str, options: list[str]) -> str:
    """สร้างข้อความที่จะใช้ค้นหา

    รับ:  question  คำถาม (หรือ requirement ของโจทย์เขียนโค้ด ซึ่งกรณีนี้ options จะเป็น list ว่าง)
          options   ข้อความของตัวเลือก ก–จ (ไม่มีตัวอักษร key นำหน้า)
    คืน:  สตริงคำค้น (ห้ามว่าง)
    """
    q = question.strip()
    include_options = getattr(settings, "QUERY_WITH_OPTIONS", True)

    if options and include_options:
        clean_opts = [o.strip() for o in options if o.strip()]
        if clean_opts:
            return f"{q} {' '.join(clean_opts)}"
    return q or "query"


def _compute_bm25_scores(corpus_tokens: list[list[str]], query_tokens: list[str]) -> list[float]:
    """Compute BM25 scores for each document in corpus against query."""
    n_docs = len(corpus_tokens)
    if n_docs == 0 or not query_tokens:
        return [0.0] * n_docs

    doc_lengths = [len(doc) for doc in corpus_tokens]
    avg_dl = sum(doc_lengths) / max(n_docs, 1) or 1.0

    # Document frequency for query terms
    df: collections.Counter = collections.Counter()
    for doc in corpus_tokens:
        for t in set(doc):
            df[t] += 1

    k1 = 1.5
    b = 0.75
    scores = [0.0] * n_docs
    q_counts = collections.Counter(query_tokens)

    for term, q_weight in q_counts.items():
        if term not in df:
            continue
        n_qi = df[term]
        # Standard Lucene/BM25 IDF formula
        idf = math.log(1.0 + (n_docs - n_qi + 0.5) / (n_qi + 0.5))
        for idx, doc in enumerate(corpus_tokens):
            tf = doc.count(term)
            if tf > 0:
                len_norm = 1.0 - b + b * (doc_lengths[idx] / avg_dl)
                term_score = idf * (tf * (k1 + 1.0)) / (tf + k1 * len_norm)
                scores[idx] += term_score * q_weight

    return scores


def search(chunks: list[Chunk], query_text: str, query_vector: list[float] | None, k: int) -> list[tuple[Chunk, float]]:
    """เลือก chunk ที่เกี่ยวข้องที่สุด k ชิ้น

    รับ:  chunks        chunk ทั้งหมดของเอกสารชุดนี้ (chunk.text, chunk.vector, chunk.doc_id)
          query_text    คำค้น (จาก make_query)
          query_vector  vector ของคำค้น (จาก embedding.embed_query) หรือ None ถ้ายังไม่ได้ทำ embedding
          k             จำนวนที่ระบบแนะนำ (5) จะใช้ settings.TOP_K ของตัวเองแทนก็ได้
    คืน:  list ของ (chunk, score) เรียงจากเกี่ยวข้องมากไปน้อย
    """
    if not chunks:
        return []

    top_k = getattr(settings, "TOP_K", k)
    hybrid_weight = getattr(settings, "HYBRID_WEIGHT", 0.35)
    min_score = getattr(settings, "MIN_SCORE", None)

    # 1. Dense vector scores (cosine similarity)
    has_vectors = query_vector is not None and any(c.vector is not None for c in chunks)
    if has_vectors and query_vector is not None:
        valid_dim = len(query_vector)
        dummy_zero = [0.0] * valid_dim
        chunk_vectors = [c.vector if c.vector is not None and len(c.vector) == valid_dim else dummy_zero for c in chunks]
        cos_scores = cosine_many(query_vector, chunk_vectors)
        # Normalize cosine [-1, 1] to [0, 1]
        dense_scores = [max(0.0, float(s)) for s in cos_scores]
    else:
        dense_scores = [0.0] * len(chunks)

    # 2. Sparse lexical scores (BM25)
    query_tokens = _tokenize(query_text)
    corpus_tokens = [_tokenize(f"{c.doc_id} {c.text}") for c in chunks]
    raw_bm25 = _compute_bm25_scores(corpus_tokens, query_tokens)
    max_bm25 = max(raw_bm25) if raw_bm25 else 0.0
    bm25_scores = [(s / max_bm25) if max_bm25 > 0 else 0.0 for s in raw_bm25]

    # 3. Hybrid score fusion
    scored: list[tuple[Chunk, float]] = []
    for i, chunk in enumerate(chunks):
        if has_vectors and max_bm25 > 0:
            final_score = (1.0 - hybrid_weight) * dense_scores[i] + hybrid_weight * bm25_scores[i]
        elif has_vectors:
            final_score = dense_scores[i]
        else:
            final_score = bm25_scores[i]

        if min_score is not None and final_score < min_score:
            continue
        scored.append((chunk, float(final_score)))

    # Sort descending by score
    scored.sort(key=lambda item: item[1], reverse=True)
    if scored and scored[0][1] > 0.5:
        max_s = scored[0][1]
        scored = [item for item in scored if item[1] >= max_s * 0.70 or item == scored[0]]
    return scored[:top_k]
