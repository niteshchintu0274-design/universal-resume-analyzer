from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .skill_matcher import extract_skills
except ImportError:
    from skill_matcher import extract_skills


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent.parent / "database" / "resume_analyzer.db"


@dataclass(frozen=True)
class Job:
    id: int
    title: str
    company: str
    description: str
    skills: str
    location: str
    apply_url: str
    created_at: str


def get_connection(database_path: str | Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection and ensure the parent directory exists."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_jobs_table(database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
    """Create the jobs table used for resume-to-job recommendations."""
    with get_connection(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                company TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL,
                skills TEXT NOT NULL DEFAULT '',
                location TEXT NOT NULL DEFAULT '',
                apply_url TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def add_job(
    title: str,
    description: str,
    company: str = "",
    skills: str | list[str] | tuple[str, ...] = "",
    location: str = "",
    apply_url: str = "",
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Insert a job into SQLite and return its id."""
    if not title.strip():
        raise ValueError("Job title is required.")
    if not description.strip():
        raise ValueError("Job description is required.")

    init_jobs_table(database_path)
    skills_text = serialize_skills(skills)

    with get_connection(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO jobs (title, company, description, skills, location, apply_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title.strip(),
                company.strip(),
                description.strip(),
                skills_text,
                location.strip(),
                apply_url.strip(),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def bulk_add_jobs(
    jobs: Iterable[dict[str, Any]],
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[int]:
    """Insert many jobs and return their ids."""
    ids: list[int] = []
    for job in jobs:
        ids.append(
            add_job(
                title=job.get("title", ""),
                company=job.get("company", ""),
                description=job.get("description", ""),
                skills=job.get("skills", ""),
                location=job.get("location", ""),
                apply_url=job.get("apply_url", ""),
                database_path=database_path,
            )
        )
    return ids


def list_jobs(database_path: str | Path = DEFAULT_DATABASE_PATH) -> list[dict[str, Any]]:
    """Return all jobs from SQLite as dictionaries."""
    init_jobs_table(database_path)

    with get_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT id, title, company, description, skills, location, apply_url, created_at
            FROM jobs
            ORDER BY id DESC
            """
        ).fetchall()

    return [row_to_job_dict(row) for row in rows]


def get_job(job_id: int, database_path: str | Path = DEFAULT_DATABASE_PATH) -> dict[str, Any] | None:
    """Return a single job by id."""
    init_jobs_table(database_path)

    with get_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT id, title, company, description, skills, location, apply_url, created_at
            FROM jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()

    return row_to_job_dict(row) if row else None


def recommend_jobs(
    resume_text: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    limit: int = 5,
    min_similarity: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Recommend jobs using cosine similarity between the resume and jobs in SQLite.

    Similarity is computed with TF-IDF vectors built from:
    - resume text
    - extracted resume skills
    - job description
    - job skill text
    """
    if not resume_text.strip():
        return []

    jobs = list_jobs(database_path)
    if not jobs:
        return []

    resume_skills = extract_skills(resume_text)
    resume_document = build_resume_document(resume_text, resume_skills)
    job_documents = [build_job_document(job) for job in jobs]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    matrix = vectorizer.fit_transform([resume_document, *job_documents])
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    threshold = normalize_similarity_threshold(min_similarity)
    recommendations: list[dict[str, Any]] = []
    for job, similarity in zip(jobs, similarities):
        raw_similarity = float(similarity)
        score = round(float(similarity) * 100, 2)
        if score < threshold:
            continue

        job_skills = normalize_skills(job["skills"])
        matched_skills = [skill for skill in job_skills if skill in resume_skills]
        missing_skills = [skill for skill in job_skills if skill not in resume_skills]

        recommendations.append(
            {
                **job,
                "similarity": score,
                "cosine_similarity": round(raw_similarity, 4),
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "resume_skills": resume_skills,
            }
        )

    recommendations.sort(key=lambda item: item["similarity"], reverse=True)
    return recommendations[: max(0, limit)]


def recommend_jobs_for_analysis(
    analysis_id: int,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Recommend jobs using resume_text stored in the analyses table."""
    init_jobs_table(database_path)

    with get_connection(database_path) as connection:
        row = connection.execute("SELECT resume_text FROM analyses WHERE id = ?", (analysis_id,)).fetchone()

    if row is None:
        raise ValueError(f"Analysis not found: {analysis_id}")

    return recommend_jobs(row["resume_text"], database_path=database_path, limit=limit)


def delete_job(job_id: int, database_path: str | Path = DEFAULT_DATABASE_PATH) -> bool:
    """Delete a job by id. Returns True when a row was deleted."""
    init_jobs_table(database_path)

    with get_connection(database_path) as connection:
        cursor = connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        connection.commit()
        return cursor.rowcount > 0


def seed_sample_jobs(database_path: str | Path = DEFAULT_DATABASE_PATH) -> list[int]:
    """Add a few sample jobs for local testing."""
    sample_jobs = [
        {
            "title": "Python Flask Developer",
            "company": "Acme Labs",
            "description": "Build Flask APIs with Python, SQL, Docker, Git, AWS, HTML, CSS, and Tailwind.",
            "skills": ["python", "flask", "sql", "docker", "git", "aws", "html", "css", "tailwind"],
            "location": "Remote",
        },
        {
            "title": "React Frontend Developer",
            "company": "Bright UI",
            "description": "Create responsive interfaces using JavaScript, React, HTML, CSS, Tailwind, Git, and API integration.",
            "skills": ["javascript", "react", "html", "css", "tailwind", "git"],
            "location": "Hybrid",
        },
        {
            "title": "Machine Learning Engineer",
            "company": "DataWorks",
            "description": "Develop machine learning and NLP pipelines with Python, pandas, numpy, deep learning, Docker, and AWS.",
            "skills": ["python", "machine learning", "deep learning", "nlp", "pandas", "numpy", "docker", "aws"],
            "location": "Bengaluru",
        },
    ]
    return bulk_add_jobs(sample_jobs, database_path=database_path)


def row_to_job_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a SQLite row into a plain dictionary."""
    return {
        "id": row["id"],
        "title": row["title"],
        "company": row["company"],
        "description": row["description"],
        "skills": row["skills"],
        "location": row["location"],
        "apply_url": row["apply_url"],
        "created_at": row["created_at"],
    }


def build_resume_document(resume_text: str, resume_skills: list[str]) -> str:
    """Build weighted text for resume vectorization."""
    return " ".join(
        [
            resume_text,
            " ".join(resume_skills * 4),
        ]
    )


def build_job_document(job: dict[str, Any]) -> str:
    """Build weighted text for job vectorization."""
    skills = normalize_skills(job.get("skills", ""))
    return " ".join(
        [
            job.get("title", ""),
            job.get("company", ""),
            job.get("description", ""),
            job.get("skills", ""),
            " ".join(skills * 4),
        ]
    )


def serialize_skills(skills: str | list[str] | tuple[str, ...]) -> str:
    """Store skills as a comma-separated string."""
    if isinstance(skills, str):
        return skills.strip()
    return ", ".join(str(skill).strip().lower() for skill in skills if str(skill).strip())


def normalize_skills(skills: str | list[str] | tuple[str, ...]) -> list[str]:
    """Normalize job skills from a string or sequence."""
    if isinstance(skills, str):
        raw_skills = [skill.strip() for skill in skills.split(",")]
    else:
        raw_skills = [str(skill).strip() for skill in skills]

    normalized: list[str] = []
    for skill in raw_skills:
        value = " ".join(skill.lower().split())
        if value and value not in normalized:
            normalized.append(value)

    return normalized


def normalize_similarity_threshold(value: float) -> float:
    """Accept min_similarity as either 0-1 cosine value or 0-100 percentage."""
    if value <= 0:
        return 0.0
    if value <= 1:
        return value * 100
    return min(value, 100.0)
