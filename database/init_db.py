from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "database" / "resume_analyzer.db"


SAMPLE_JOBS: list[dict[str, str]] = [
    {
        "title": "Python Developer",
        "company": "TechNova Solutions",
        "required_skills": "python, flask, django, sql, mongodb, git, docker, aws",
        "description": (
            "Build and maintain Python web applications, REST APIs, and internal automation tools. "
            "Work with Flask or Django, relational databases, MongoDB, Git workflows, Docker containers, "
            "and AWS deployment environments."
        ),
        "salary_range": "INR 6-12 LPA",
        "location": "Bengaluru, India",
    },
    {
        "title": "Data Scientist",
        "company": "Insight Analytics",
        "required_skills": "python, machine learning, deep learning, nlp, pandas, numpy, sql",
        "description": (
            "Analyze business data, build predictive models, create experiments, and present insights to "
            "stakeholders. The role involves Python, pandas, numpy, SQL, machine learning, NLP, and model "
            "evaluation for production use cases."
        ),
        "salary_range": "INR 8-18 LPA",
        "location": "Hyderabad, India",
    },
    {
        "title": "Frontend Developer",
        "company": "PixelCraft Labs",
        "required_skills": "javascript, react, html, css, tailwind, git",
        "description": (
            "Create responsive, accessible, and polished user interfaces using React, JavaScript, HTML, CSS, "
            "and Tailwind CSS. Collaborate with backend developers and designers to ship high-quality web "
            "features."
        ),
        "salary_range": "INR 5-11 LPA",
        "location": "Pune, India",
    },
    {
        "title": "Backend Developer",
        "company": "ServerSide Systems",
        "required_skills": "python, java, flask, django, sql, mongodb, git, docker, aws",
        "description": (
            "Design backend services, database schemas, APIs, authentication flows, and integrations. "
            "Candidates should understand Python or Java, SQL, MongoDB, Docker, Git, and cloud deployment."
        ),
        "salary_range": "INR 7-15 LPA",
        "location": "Chennai, India",
    },
    {
        "title": "Full Stack Developer",
        "company": "CodeBridge Technologies",
        "required_skills": "python, javascript, react, flask, django, sql, mongodb, html, css, tailwind, git, docker",
        "description": (
            "Develop end-to-end web applications across frontend and backend layers. Work with React, "
            "Tailwind CSS, Python APIs, SQL or MongoDB databases, Git, testing, and containerized deployment."
        ),
        "salary_range": "INR 8-16 LPA",
        "location": "Remote, India",
    },
    {
        "title": "Machine Learning Engineer",
        "company": "ModelWorks AI",
        "required_skills": "python, machine learning, deep learning, nlp, pandas, numpy, docker, aws, git",
        "description": (
            "Build, train, evaluate, and deploy machine learning systems. Responsibilities include data "
            "pipelines, feature engineering, deep learning experiments, NLP models, Docker packaging, and "
            "AWS-based model services."
        ),
        "salary_range": "INR 10-22 LPA",
        "location": "Gurugram, India",
    },
    {
        "title": "DevOps Engineer",
        "company": "DeployMate Cloud",
        "required_skills": "docker, aws, git, python, sql",
        "description": (
            "Support CI/CD pipelines, infrastructure automation, container deployments, monitoring, and "
            "release reliability. Use Docker, AWS, Git, scripting with Python, and operational best practices."
        ),
        "salary_range": "INR 8-17 LPA",
        "location": "Noida, India",
    },
    {
        "title": "Data Analyst",
        "company": "MetricTree Consulting",
        "required_skills": "sql, python, pandas, numpy",
        "description": (
            "Prepare dashboards, analyze business metrics, clean datasets, and write SQL queries for reporting. "
            "Use Python, pandas, numpy, spreadsheets, and visualization tools to communicate insights clearly."
        ),
        "salary_range": "INR 4-9 LPA",
        "location": "Mumbai, India",
    },
    {
        "title": "Android Developer",
        "company": "AppForge Mobile",
        "required_skills": "java, sql, git",
        "description": (
            "Develop Android applications, integrate REST APIs, manage local storage, debug performance issues, "
            "and collaborate with designers and backend teams. Strong Java, SQL, and Git fundamentals are expected."
        ),
        "salary_range": "INR 5-12 LPA",
        "location": "Ahmedabad, India",
    },
    {
        "title": "iOS Developer",
        "company": "SwiftWave Apps",
        "required_skills": "git, sql",
        "description": (
            "Build and maintain iOS applications, integrate APIs, implement clean UI flows, manage app data, "
            "and follow code review practices. Experience with Git, local storage concepts, and API integration "
            "is important."
        ),
        "salary_range": "INR 6-14 LPA",
        "location": "Kochi, India",
    },
    {
        "title": "Cloud Engineer",
        "company": "NimbusStack",
        "required_skills": "aws, docker, python, sql, git",
        "description": (
            "Design and support cloud infrastructure, deployment pipelines, container workloads, backups, "
            "security policies, and monitoring. The role uses AWS, Docker, Python scripting, SQL systems, and Git."
        ),
        "salary_range": "INR 9-20 LPA",
        "location": "Bengaluru, India",
    },
    {
        "title": "Cybersecurity Analyst",
        "company": "SecureLayer Infosec",
        "required_skills": "python, sql, git, aws",
        "description": (
            "Monitor security events, investigate incidents, automate security checks, analyze logs, and support "
            "cloud security reviews. Python scripting, SQL querying, Git, and AWS knowledge are useful for this role."
        ),
        "salary_range": "INR 6-13 LPA",
        "location": "Delhi, India",
    },
    {
        "title": "UI/UX Designer",
        "company": "HumanFirst Design",
        "required_skills": "html, css, javascript, react, tailwind, git",
        "description": (
            "Design user flows, wireframes, prototypes, and responsive interfaces. Collaborate with frontend "
            "engineers and use HTML, CSS, JavaScript, React, Tailwind CSS, and Git awareness to create practical designs."
        ),
        "salary_range": "INR 5-10 LPA",
        "location": "Jaipur, India",
    },
    {
        "title": "Business Analyst",
        "company": "ProcessIQ Advisory",
        "required_skills": "sql, python, pandas",
        "description": (
            "Gather requirements, document workflows, analyze business data, support product decisions, and "
            "coordinate between stakeholders and engineering teams. SQL, Python, and pandas are helpful for analysis."
        ),
        "salary_range": "INR 5-11 LPA",
        "location": "Remote, India",
    },
    {
        "title": "Database Administrator",
        "company": "DataVault Systems",
        "required_skills": "sql, mongodb, python, aws, docker",
        "description": (
            "Administer relational and NoSQL databases, optimize queries, manage backups, monitor performance, "
            "and support cloud-hosted database systems. SQL, MongoDB, Python automation, AWS, and Docker are valuable."
        ),
        "salary_range": "INR 7-16 LPA",
        "location": "Hyderabad, India",
    },
]


def get_connection(database_path: str | Path = DATABASE_PATH) -> sqlite3.Connection:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def create_tables(connection: sqlite3.Connection) -> None:
    create_analyses_table(connection)
    create_jobs_table(connection)
    ensure_jobs_columns(connection)
    create_indexes(connection)
    connection.commit()


def create_analyses_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            candidate_name TEXT,
            email TEXT,
            phone TEXT,
            score INTEGER,
            extracted_skills TEXT NOT NULL DEFAULT '[]',
            resume_text TEXT NOT NULL,
            job_description TEXT,
            claude_result TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
        """
    )


def create_jobs_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            required_skills TEXT NOT NULL DEFAULT '',
            skills TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL,
            salary_range TEXT NOT NULL DEFAULT '',
            location TEXT NOT NULL DEFAULT '',
            apply_url TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def ensure_jobs_columns(connection: sqlite3.Connection) -> None:
    existing_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
    }

    column_definitions = {
        "required_skills": "TEXT NOT NULL DEFAULT ''",
        "skills": "TEXT NOT NULL DEFAULT ''",
        "salary_range": "TEXT NOT NULL DEFAULT ''",
        "location": "TEXT NOT NULL DEFAULT ''",
        "apply_url": "TEXT NOT NULL DEFAULT ''",
        "created_at": "TEXT NOT NULL DEFAULT ''",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
    }

    for column_name, column_definition in column_definitions.items():
        if column_name not in existing_columns:
            connection.execute(f"ALTER TABLE jobs ADD COLUMN {column_name} {column_definition}")

    connection.execute(
        """
        UPDATE jobs
        SET required_skills = skills
        WHERE required_skills = '' AND skills != ''
        """
    )
    connection.execute(
        """
        UPDATE jobs
        SET skills = required_skills
        WHERE skills = '' AND required_skills != ''
        """
    )


def create_indexes(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE INDEX IF NOT EXISTS idx_jobs_title ON jobs(title)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_jobs_location ON jobs(location)")


def seed_jobs(connection: sqlite3.Connection, jobs: list[dict[str, str]] = SAMPLE_JOBS) -> int:
    inserted_or_updated = 0

    for job in jobs:
        upsert_job(connection, job)
        inserted_or_updated += 1

    connection.commit()
    return inserted_or_updated


def upsert_job(connection: sqlite3.Connection, job: dict[str, str]) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    existing = connection.execute(
        """
        SELECT id
        FROM jobs
        WHERE lower(title) = lower(?) AND lower(company) = lower(?)
        LIMIT 1
        """,
        (job["title"], job["company"]),
    ).fetchone()

    values = {
        "title": job["title"].strip(),
        "company": job["company"].strip(),
        "required_skills": normalize_skills(job["required_skills"]),
        "description": job["description"].strip(),
        "salary_range": job["salary_range"].strip(),
        "location": job["location"].strip(),
        "apply_url": job.get("apply_url", "").strip(),
    }
    values["skills"] = values["required_skills"]

    if existing:
        connection.execute(
            """
            UPDATE jobs
            SET title = ?,
                company = ?,
                required_skills = ?,
                skills = ?,
                description = ?,
                salary_range = ?,
                location = ?,
                apply_url = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                values["title"],
                values["company"],
                values["required_skills"],
                values["skills"],
                values["description"],
                values["salary_range"],
                values["location"],
                values["apply_url"],
                now,
                existing["id"],
            ),
        )
        return

    connection.execute(
        """
        INSERT INTO jobs (
            title,
            company,
            required_skills,
            skills,
            description,
            salary_range,
            location,
            apply_url,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            values["title"],
            values["company"],
            values["required_skills"],
            values["skills"],
            values["description"],
            values["salary_range"],
            values["location"],
            values["apply_url"],
            now,
            now,
        ),
    )


def normalize_skills(skills: str) -> str:
    parts = [part.strip().lower() for part in skills.split(",") if part.strip()]
    deduped = list(dict.fromkeys(parts))
    return ", ".join(deduped)


def fetch_jobs(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT title, company, required_skills, description, salary_range, location
        FROM jobs
        ORDER BY title
        """
    ).fetchall()
    return [dict(row) for row in rows]


def initialize_database(database_path: str | Path = DATABASE_PATH) -> dict[str, Any]:
    with get_connection(database_path) as connection:
        create_tables(connection)
        seeded_count = seed_jobs(connection)
        total_jobs = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    return {
        "database_path": str(Path(database_path).resolve()),
        "seeded_jobs": seeded_count,
        "total_jobs": total_jobs,
    }


def main() -> None:
    result = initialize_database()
    print("Database initialized successfully.")
    print(f"Path: {result['database_path']}")
    print(f"Seeded sample jobs: {result['seeded_jobs']}")
    print(f"Total jobs in database: {result['total_jobs']}")


if __name__ == "__main__":
    main()
