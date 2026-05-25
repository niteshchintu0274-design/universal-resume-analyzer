from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_resume_rag_context(
    resume_text: str,
    user_message: str,
    analysis_result: dict[str, Any],
    chat_history: list[dict[str, str]] | None = None,
    job_description: str = "",
    analysis_id: int | None = None,
    persist_dir: str | Path | None = None,
    provider: str = "local",
    max_chars: int = 5000,
) -> str:
    chunks = split_resume_chunks(resume_text, analysis_result)
    if not chunks:
        return resume_text[:max_chars]

    profile = analysis_result.get("profile", {})
    query = build_retrieval_query(user_message, profile, chat_history, job_description)
    selected = []

    if provider == "chroma":
        selected = retrieve_with_chroma(chunks, query, resume_text, analysis_id, persist_dir)

    if not selected:
        selected = retrieve_with_tfidf(chunks, query, limit=6)

    context_parts = [
        "Detected profile:\n" + json.dumps(profile, ensure_ascii=False),
        "Relevant resume context:\n" + "\n\n".join(selected),
    ]
    if job_description.strip():
        context_parts.append("Target job description excerpt:\n" + job_description.strip()[:1800])
    return "\n\n".join(part for part in context_parts if part).strip()[:max_chars]


def split_resume_chunks(resume_text: str, analysis_result: dict[str, Any]) -> list[str]:
    chunks: list[str] = []
    for section_name, data in (analysis_result.get("sections") or {}).items():
        excerpt = str(data.get("excerpt") or "").strip()
        if excerpt:
            chunks.append(f"{section_name}: {excerpt}")

    text = clean_text(resume_text)
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    for paragraph in paragraphs:
        if len(paragraph) <= 900:
            chunks.append(paragraph)
            continue
        for index in range(0, len(paragraph), 700):
            piece = paragraph[index : index + 900].strip()
            if piece:
                chunks.append(piece)

    return dedupe_chunks(chunks)


def build_retrieval_query(
    user_message: str,
    profile: dict[str, Any],
    chat_history: list[dict[str, str]] | None,
    job_description: str,
) -> str:
    history_text = " ".join(str(item.get("content", "")) for item in (chat_history or [])[-4:])
    profile_terms = " ".join(
        str(profile.get(key, ""))
        for key in ("role", "industry", "sub_domain", "career_category")
    )
    skills = " ".join(str(skill) for skill in profile.get("target_skills", [])[:10])
    return " ".join([user_message, history_text, profile_terms, skills, job_description[:1200]]).strip()


def retrieve_with_tfidf(chunks: list[str], query: str, limit: int = 6) -> list[str]:
    if not query.strip():
        return chunks[:limit]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4000)
        matrix = vectorizer.fit_transform([query, *chunks])
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    except ValueError:
        return chunks[:limit]

    ranked = sorted(zip(chunks, similarities), key=lambda item: item[1], reverse=True)
    selected = [chunk for chunk, score in ranked if score > 0][:limit]
    return selected or chunks[:limit]


def retrieve_with_chroma(
    chunks: list[str],
    query: str,
    resume_text: str,
    analysis_id: int | None,
    persist_dir: str | Path | None,
) -> list[str]:
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_core.embeddings import Embeddings
    except Exception:
        return []

    class LocalHashEmbeddings(Embeddings):
        def __init__(self) -> None:
            self.vectorizer = HashingVectorizer(
                n_features=512,
                alternate_sign=False,
                norm="l2",
                ngram_range=(1, 2),
            )

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.vectorizer.transform(texts).toarray().tolist()

        def embed_query(self, text: str) -> list[float]:
            return self.vectorizer.transform([text]).toarray()[0].tolist()

    collection_name = f"resume_{analysis_id or hashlib.sha1(resume_text.encode('utf-8')).hexdigest()[:16]}"
    target_dir = str(Path(persist_dir or Path.cwd() / "database" / "chroma"))
    documents = [Document(page_content=chunk, metadata={"chunk": index}) for index, chunk in enumerate(chunks)]

    try:
        store = Chroma(
            collection_name=collection_name,
            embedding_function=LocalHashEmbeddings(),
            persist_directory=target_dir,
        )
        if not store.get().get("ids"):
            store.add_documents(documents)
        return [doc.page_content for doc in store.similarity_search(query, k=6)]
    except Exception:
        return []


def clean_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedupe_chunks(chunks: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for chunk in chunks:
        value = re.sub(r"\s+", " ", chunk).strip()
        key = value.lower()
        if len(value) < 20 or key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped
