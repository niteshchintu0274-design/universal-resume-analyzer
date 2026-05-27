import json
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import anthropic
import httpx
import spacy
from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    current_app,
    flash,
    g,
    has_request_context,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from config import Config
from modules.ai_helper import (
    ClaudeConfigurationError,
    GeminiConfigurationError,
    build_gemini_payload,
    career_chatbot as claude_career_chatbot,
    extract_gemini_text,
    gemini_career_chatbot,
)
from modules.extractor import extract_pdf_content
from modules.job_apis import JobFilters, fetch_matching_jobs
from modules.rag_context import build_resume_rag_context
from modules.universal_analyzer import (
    analyze_resume_doctor,
    build_local_chat_response,
    compare_resume_versions,
    extract_contact_details,
)

load_dotenv()

IST = timezone(timedelta(hours=5, minutes=30))
SESSION_ID_KEY = "resume_analyzer_session_id"
CURRENT_ANALYSIS_ID_KEY = "current_analysis_id"


COMMON_SKILLS = {
    "python",
    "java",
    "javascript",
    "typescript",
    "c",
    "c++",
    "c#",
    "go",
    "rust",
    "sql",
    "html",
    "css",
    "react",
    "angular",
    "vue",
    "node.js",
    "express",
    "django",
    "flask",
    "fastapi",
    "spring",
    "tailwind",
    "bootstrap",
    "sqlite",
    "postgresql",
    "mysql",
    "mongodb",
    "redis",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "terraform",
    "git",
    "github",
    "gitlab",
    "ci/cd",
    "linux",
    "machine learning",
    "deep learning",
    "nlp",
    "data analysis",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "spacy",
    "pdfplumber",
    "rest api",
    "graphql",
    "microservices",
    "agile",
    "scrum",
    "testing",
    "pytest",
    "selenium",
    "power bi",
    "tableau",
    "excel",
}

EDUCATION_KEYWORDS = {
    "bachelor",
    "master",
    "phd",
    "b.tech",
    "m.tech",
    "b.e",
    "m.e",
    "bsc",
    "msc",
    "mba",
    "degree",
    "university",
    "college",
    "institute",
}

STOPWORDS_FOR_KEYWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "could",
    "during",
    "each",
    "from",
    "have",
    "into",
    "more",
    "most",
    "other",
    "over",
    "same",
    "such",
    "than",
    "that",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "under",
    "using",
    "with",
    "would",
}

def create_app(config_class: type[Config] = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)

    ensure_directories(app)

    with app.app_context():
        init_db()

    app.add_url_rule("/", "index", index, methods=["GET"])
    app.add_url_rule("/chatbot", "chatbot", chatbot, methods=["GET"])
    app.add_url_rule("/compare", "compare", compare, methods=["GET", "POST"])
    app.add_url_rule("/analyze", "analyze_resume", analyze_resume, methods=["POST"])
    app.add_url_rule("/analysis/<int:analysis_id>", "analysis_detail", analysis_detail, methods=["GET"])
    app.add_url_rule("/analysis/<int:analysis_id>/interview-prep", "interview_prep", interview_prep, methods=["GET"])
    app.add_url_rule("/analysis/<int:analysis_id>/download", "download_resume", download_resume, methods=["GET"])
    app.add_url_rule("/api/analyze", "api_analyze_resume", api_analyze_resume, methods=["POST"])
    app.add_url_rule("/api/career-chat", "api_career_chat", api_career_chat, methods=["POST"])
    app.add_url_rule("/api/rewrite-section", "api_rewrite_section", api_rewrite_section, methods=["POST"])
    app.add_url_rule("/api/job-match", "api_job_match", api_job_match, methods=["POST"])
    app.add_url_rule("/api/interview-questions/<int:analysis_id>", "api_interview_questions", api_interview_questions, methods=["GET", "POST"])
    app.add_url_rule("/api/jobs/<int:analysis_id>", "api_matching_jobs", api_matching_jobs, methods=["GET"])
    app.add_url_rule("/api/compare", "api_compare", api_compare, methods=["POST"])
    app.add_url_rule("/api/analysis/<int:analysis_id>", "api_analysis_detail", api_analysis_detail, methods=["GET"])

    app.teardown_appcontext(close_db)
    return app


def ensure_directories(app: Flask) -> None:
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["DATABASE_PATH"]).parent.mkdir(parents=True, exist_ok=True)


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE_PATH"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(error: Exception | None = None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    db = get_db()
    db.execute("PRAGMA foreign_keys = ON")
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            name TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id TEXT,
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
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS interview_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            user_id INTEGER,
            session_id TEXT,
            target_role TEXT,
            job_description TEXT,
            questions_json TEXT NOT NULL DEFAULT '[]',
            ai_provider TEXT NOT NULL DEFAULT 'local',
            created_at TEXT NOT NULL,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )
    ensure_columns(
        db,
        "analyses",
        {
            "user_id": "INTEGER",
            "session_id": "TEXT",
        },
    )
    ensure_interview_questions_anonymous_schema(db)
    db.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses(user_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_analyses_session_id ON analyses(session_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_analysis_id ON chat_messages(analysis_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_interview_questions_analysis_user ON interview_questions(analysis_id, user_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_interview_questions_analysis_session ON interview_questions(analysis_id, session_id)")
    db.commit()


def ensure_columns(db: sqlite3.Connection, table_name: str, column_definitions: dict[str, str]) -> None:
    existing_columns = {row["name"] for row in db.execute(f"PRAGMA table_info({table_name})").fetchall()}
    for column_name, definition in column_definitions.items():
        if column_name not in existing_columns:
            db.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def ensure_interview_questions_anonymous_schema(db: sqlite3.Connection) -> None:
    columns = {row["name"]: row for row in db.execute("PRAGMA table_info(interview_questions)").fetchall()}
    if not columns:
        return
    user_id_required = bool(columns.get("user_id") and columns["user_id"]["notnull"])
    if not user_id_required and "session_id" in columns:
        return

    has_session_id = "session_id" in columns
    session_expr = "session_id" if has_session_id else "NULL"
    db.execute("DROP TABLE IF EXISTS interview_questions_new")
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS interview_questions_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            user_id INTEGER,
            session_id TEXT,
            target_role TEXT,
            job_description TEXT,
            questions_json TEXT NOT NULL DEFAULT '[]',
            ai_provider TEXT NOT NULL DEFAULT 'local',
            created_at TEXT NOT NULL,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )
    db.execute(
        f"""
        INSERT INTO interview_questions_new (
            id,
            analysis_id,
            user_id,
            session_id,
            target_role,
            job_description,
            questions_json,
            ai_provider,
            created_at
        )
        SELECT
            id,
            analysis_id,
            user_id,
            {session_expr},
            target_role,
            job_description,
            questions_json,
            ai_provider,
            created_at
        FROM interview_questions
        """
    )
    db.execute("DROP TABLE interview_questions")
    db.execute("ALTER TABLE interview_questions_new RENAME TO interview_questions")


def index() -> str:
    return render_template("index.html")


def chatbot() -> str:
    latest = get_current_session_analysis()
    latest_result = normalize_ai_fallback_status(parse_json_field(latest["claude_result"], {})) if latest else {}
    return render_template("chatbot.html", latest=latest, latest_result=latest_result)


def compare():
    if request.method == "GET":
        return render_template("compare.html", comparison=None)

    try:
        comparison = handle_compare_submission(request.files.get("resume_v1"), request.files.get("resume_v2"), request.form.get("job_description", ""))
        return render_template("compare.html", comparison=comparison)
    except ValueError as exc:
        flash(str(exc), "error")
        return render_template("compare.html", comparison=None)
    except Exception as exc:
        current_app.logger.exception("Resume comparison failed")
        flash(friendly_user_error("comparison"), "error")
        return render_template("compare.html", comparison=None)


def analyze_resume():
    try:
        analysis_id = handle_resume_submission(request.files.get("resume"), request.form.get("job_description", ""))
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))
    except Exception as exc:
        current_app.logger.exception("Resume analysis failed")
        flash(friendly_user_error("analysis"), "error")
        return redirect(url_for("index"))

    flash("Resume analyzed successfully.", "success")
    return redirect(url_for("analysis_detail", analysis_id=analysis_id))


def api_analyze_resume():
    try:
        analysis_id = handle_resume_submission(request.files.get("resume"), request.form.get("job_description", ""))
        return jsonify({"ok": True, "analysis": serialize_analysis(get_analysis(analysis_id))}), 201
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        current_app.logger.exception("API resume analysis failed")
        return jsonify({"ok": False, "error": friendly_user_error("analysis")}), 500


def api_career_chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message") or data.get("user_message") or "").strip()
    if not message:
        return jsonify({"ok": False, "error": "Career question is required."}), 400

    analysis_id = parse_optional_int(data.get("analysis_id"))
    chat_history = data.get("chat_history") if isinstance(data.get("chat_history"), list) else []
    analysis = get_analysis(analysis_id) if analysis_id else None
    if analysis_id and analysis is None:
        return jsonify({"ok": False, "error": "Analysis not found."}), 404
    analysis_result = parse_json_field(analysis["claude_result"], {}) if analysis else {}
    analysis_result = normalize_ai_fallback_status(analysis_result)
    resume_text = str((analysis["resume_text"] if analysis else data.get("resume_text")) or "")
    job_description = str(data.get("job_description") or (analysis["job_description"] if analysis else "") or "")
    if analysis and not analysis_result.get("profile"):
        analysis_result = analyze_resume_doctor(resume_text, job_description=job_description)
    detected_role = str(data.get("target_role") or analysis_result.get("profile", {}).get("role") or "")

    if analysis_id and not chat_history:
        chat_history = get_chat_history(analysis_id, limit=12)

    try:
        rag_context = build_resume_rag_context(
            resume_text=resume_text,
            user_message=message,
            analysis_result=analysis_result,
            chat_history=chat_history,
            job_description=job_description,
            analysis_id=analysis_id,
            persist_dir=current_app.config["CHROMA_PERSIST_DIR"],
            provider=current_app.config["RAG_PROVIDER"],
        )
        reply = ""
        ai_provider = "local"

        if current_app.config.get("ANTHROPIC_API_KEY"):
            try:
                reply = claude_career_chatbot(
                    user_message=message,
                    resume_text=rag_context or resume_text,
                    chat_history=chat_history,
                    target_role=detected_role,
                    job_description=job_description,
                    profile_context=analysis_result.get("profile", {}),
                )
                ai_provider = "anthropic"
            except (ClaudeConfigurationError, Exception) as exc:
                current_app.logger.info("Claude chat unavailable; trying Gemini/local fallback: %s", exc)

        if not reply and current_app.config.get("GEMINI_API_KEY"):
            try:
                reply = gemini_career_chatbot(
                    user_message=message,
                    resume_text=rag_context or resume_text,
                    chat_history=chat_history,
                    target_role=detected_role,
                    job_description=job_description,
                    profile_context=analysis_result.get("profile", {}),
                )
                ai_provider = "gemini"
            except (GeminiConfigurationError, Exception) as exc:
                current_app.logger.info("Gemini chat unavailable; using local fallback: %s", exc)

        if not reply:
            if not analysis_result and resume_text:
                analysis_result = analyze_resume_doctor(resume_text, job_description=job_description)
            local_reply = build_local_chat_response(
                user_message=message,
                analysis_result=analysis_result,
                resume_text=rag_context or resume_text,
                chat_history=chat_history,
            )
            reply = local_reply
        if analysis_id:
            save_chat_message(analysis_id, "user", message)
            save_chat_message(analysis_id, "assistant", reply)
        return jsonify({"ok": True, "reply": reply, "ai_provider": ai_provider})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        current_app.logger.exception("Career chatbot request failed")
        return jsonify({"ok": False, "error": friendly_user_error("chat")}), 500


def api_rewrite_section():
    data = request.get_json(silent=True) or {}
    section = str(data.get("section") or "Summary").strip()
    before = str(data.get("text") or data.get("before") or "").strip()
    analysis_id = parse_optional_int(data.get("analysis_id"))
    analysis = get_analysis(analysis_id) if analysis_id else None
    if analysis_id and analysis is None:
        return jsonify({"ok": False, "error": "Analysis not found."}), 404
    result = parse_json_field(analysis["claude_result"], {}) if analysis else {}
    if analysis and not result.get("profile"):
        result = analyze_resume_doctor(analysis["resume_text"], job_description=analysis["job_description"] or "")
    profile = result.get("profile", {})

    if not before and result:
        for item in result.get("rewrites", []):
            if item.get("section") == section:
                before = item.get("before", "")
                break

    if not before:
        return jsonify({"ok": False, "error": "Section text is required."}), 400

    versions = generate_domain_rewrites(before, section, profile)
    return jsonify({"ok": True, "section": section, "before": before, "versions": versions})


def api_job_match():
    data = request.get_json(silent=True) or {}
    analysis_id = parse_optional_int(data.get("analysis_id"))
    job_description = str(data.get("job_description") or "").strip()
    if not analysis_id:
        return jsonify({"ok": False, "error": "analysis_id is required."}), 400
    if not job_description:
        return jsonify({"ok": False, "error": "Job description is required."}), 400

    analysis = get_analysis(analysis_id)
    if analysis is None:
        return jsonify({"ok": False, "error": "Analysis not found."}), 404

    result = analyze_resume_doctor(analysis["resume_text"], job_description=job_description)
    return jsonify({"ok": True, "job_match": result.get("job_match", {}), "scores": result.get("scores", {})})


def api_interview_questions(analysis_id: int):
    analysis = get_analysis(analysis_id)
    if analysis is None:
        return jsonify({"ok": False, "error": "Analysis not found."}), 404

    if request.method == "GET":
        return jsonify({"ok": True, "interview_questions": get_latest_interview_question_bundle(analysis_id)})

    data = request.get_json(silent=True) or {}
    analysis_result = normalize_ai_fallback_status(parse_json_field(analysis["claude_result"], {}))
    if not analysis_result.get("profile"):
        analysis_result = analyze_resume_doctor(analysis["resume_text"], job_description=analysis["job_description"] or "")

    target_role = str(data.get("target_role") or analysis_result.get("profile", {}).get("role") or "").strip() or None
    job_description = str(data.get("job_description") or analysis["job_description"] or "").strip()
    generated = generate_interview_questions(
        analysis["resume_text"],
        analysis_result,
        target_role=target_role,
        job_description=job_description,
    )
    saved = save_interview_question_bundle(
        analysis_id=analysis_id,
        target_role=generated.get("target_role") or target_role or "",
        job_description=job_description,
        questions=generated.get("questions", []),
        ai_provider=generated.get("ai_provider", "local"),
    )
    return jsonify({"ok": True, "interview_questions": saved})


def api_matching_jobs(analysis_id: int):
    analysis = get_analysis(analysis_id)
    if analysis is None:
        return jsonify({"ok": False, "error": "Analysis not found."}), 404

    result = parse_json_field(analysis["claude_result"], {})
    if not result.get("profile"):
        result = analyze_resume_doctor(analysis["resume_text"], job_description=analysis["job_description"] or "")
    filters = JobFilters(
        location=request.args.get("location", "").strip(),
        remote=request.args.get("remote", "").lower() in {"1", "true", "yes", "on"},
        salary_min=parse_optional_int(request.args.get("salary_min")),
        experience=request.args.get("experience", "").strip(),
        freshers=request.args.get("freshers", "").lower() in {"1", "true", "yes", "on"},
        country=normalize_country_code(request.args.get("country") or current_app.config.get("ADZUNA_COUNTRY") or "in"),
    )
    bundle = fetch_matching_jobs(
        analysis["resume_text"],
        profile=result.get("profile", {}),
        skills=result.get("extracted_skills", parse_json_field(analysis["extracted_skills"], [])),
        database_path=current_app.config["DATABASE_PATH"],
        filters=filters,
        limit=12,
    )
    return jsonify({"ok": True, **bundle})


def api_compare():
    try:
        comparison = handle_compare_submission(
            request.files.get("resume_v1"),
            request.files.get("resume_v2"),
            request.form.get("job_description", ""),
        )
        return jsonify({"ok": True, "comparison": comparison})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        current_app.logger.exception("API resume comparison failed")
        return jsonify({"ok": False, "error": friendly_user_error("comparison")}), 500


def analysis_detail(analysis_id: int) -> str:
    analysis = get_analysis(analysis_id)
    if analysis is None:
        abort(404)

    result = parse_json_field(analysis["claude_result"], {})
    result = normalize_ai_fallback_status(result)
    if not result.get("profile"):
        result = analyze_resume_doctor(analysis["resume_text"], analysis["job_description"] or "")
    skills = result.get("extracted_skills") or parse_json_field(analysis["extracted_skills"], [])
    job_description = analysis["job_description"] or ""

    score_data = build_score_data(result, fallback=analysis["score"] or 0)

    try:
        job_bundle = fetch_matching_jobs(
            analysis["resume_text"],
            profile=result.get("profile", {}),
            skills=skills,
            database_path=current_app.config["DATABASE_PATH"],
            filters=JobFilters(country=normalize_country_code(current_app.config.get("ADZUNA_COUNTRY") or "in")),
            limit=8,
        )
        job_recommendations = job_bundle["jobs"]
        job_provider_notes = job_bundle["provider_notes"]
        job_source = job_bundle["source"]
        job_recommendation_error = None
    except Exception as exc:
        current_app.logger.exception("Could not calculate job recommendations")
        job_recommendations = []
        job_provider_notes = []
        job_source = "unavailable"
        job_recommendation_error = friendly_user_error("jobs")

    ats_result = result.get("ats", {})
    missing_skills = result.get("missing_keywords", [])
    display_score = normalize_score(result.get("scores", {}).get("overall"), fallback=analysis["score"] or 0)
    chat_history = get_chat_history(analysis_id, limit=20)
    analysis_view = prepare_analysis_for_view(analysis)
    preview_lines = build_resume_preview_lines(analysis["resume_text"], mask=True)
    raw_preview_lines = build_resume_preview_lines(analysis["resume_text"], mask=False)
    interview_bundle = get_latest_interview_question_bundle(analysis_id)

    return render_template(
        "result.html",
        analysis=analysis_view,
        result=result,
        skills=skills,
        found_skills=skills,
        missing_skills=missing_skills,
        ats_result=ats_result,
        score_data=score_data,
        display_score=display_score,
        job_recommendations=job_recommendations,
        job_provider_notes=job_provider_notes,
        job_source=job_source,
        job_recommendation_error=job_recommendation_error,
        chat_history=chat_history,
        preview_lines=preview_lines,
        raw_preview_lines=raw_preview_lines,
        interview_bundle=interview_bundle,
    )


def interview_prep(analysis_id: int):
    analysis = get_analysis(analysis_id)
    if analysis is None:
        abort(404)
    return redirect(url_for("analysis_detail", analysis_id=analysis_id) + "#interview-prep")


def download_resume(analysis_id: int):
    analysis = get_analysis(analysis_id)
    if analysis is None:
        abort(404)
    stored_path = resolve_owned_upload_path(analysis)
    if stored_path is None:
        abort(404)
    return send_file(
        stored_path,
        as_attachment=True,
        download_name=analysis["original_filename"],
    )


def api_analysis_detail(analysis_id: int):
    analysis = get_analysis(analysis_id)
    if analysis is None:
        return jsonify({"ok": False, "error": "Analysis not found"}), 404
    return jsonify({"ok": True, "analysis": serialize_analysis(analysis)})


def handle_resume_submission(file_storage, job_description: str) -> int:
    session_id = get_or_create_session_id()

    if not file_storage or not file_storage.filename:
        raise ValueError("Please upload a resume PDF.")

    if not allowed_file(file_storage.filename):
        raise ValueError("Only PDF files are supported.")

    original_filename = secure_filename(file_storage.filename) or "resume.pdf"
    stored_filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex}.pdf"
    upload_dir = get_session_upload_dir(session_id)
    saved_path = upload_dir / stored_filename
    stored_relative_path = Path(f"session_{session_id}") / stored_filename
    file_storage.save(saved_path)

    extracted = extract_pdf_content(saved_path, enable_ocr=True)
    resume_text = extracted["text"]
    diagnostics = extracted["diagnostics"]
    if len(resume_text) < current_app.config["MIN_RESUME_TEXT_LENGTH"]:
        saved_path.unlink(missing_ok=True)
        if diagnostics.get("ocr_attempted") and not diagnostics.get("ocr_available"):
            raise ValueError("Could not extract enough text from this PDF. OCR support needs PyMuPDF, pytesseract, and the Tesseract binary installed.")
        raise ValueError("Could not extract enough text from this PDF. Try a clearer text-based or OCR-readable resume PDF.")

    local_features = extract_resume_features(resume_text)
    doctor_result = analyze_resume_doctor(
        resume_text,
        job_description=job_description,
        local_features=local_features,
        diagnostics=diagnostics,
    )
    claude_result = analyze_with_claude(
        resume_text,
        job_description,
        {
            **local_features,
            "detected_profile": doctor_result.get("profile", {}),
            "domain_skills": doctor_result.get("extracted_skills", []),
        },
    )
    doctor_result = merge_ai_notes(doctor_result, claude_result)
    score = normalize_score(doctor_result.get("match_score"), fallback=doctor_result.get("scores", {}).get("overall", 0))
    extracted_skills = doctor_result.get("extracted_skills") or local_features.get("skills", [])

    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO analyses (
            user_id,
            session_id,
            original_filename,
            stored_filename,
            candidate_name,
            email,
            phone,
            score,
            extracted_skills,
            resume_text,
            job_description,
            claude_result,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            None,
            session_id,
            original_filename,
            stored_relative_path.as_posix(),
            local_features.get("candidate_name"),
            local_features.get("email"),
            local_features.get("phone"),
            score,
            json.dumps(extracted_skills),
            resume_text,
            job_description.strip(),
            json.dumps(doctor_result),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    db.commit()
    analysis_id = int(cursor.lastrowid)
    remember_current_analysis(analysis_id)
    return analysis_id


def handle_compare_submission(file_v1, file_v2, job_description: str = "") -> dict[str, Any]:
    session_id = get_or_create_session_id()

    if not file_v1 or not file_v1.filename:
        raise ValueError("Please upload Resume V1.")
    if not file_v2 or not file_v2.filename:
        raise ValueError("Please upload Resume V2.")
    if not allowed_file(file_v1.filename) or not allowed_file(file_v2.filename):
        raise ValueError("Only PDF files are supported for comparison.")

    saved_paths: list[Path] = []
    try:
        texts: list[str] = []
        diagnostics: list[dict[str, Any]] = []
        for file_storage in (file_v1, file_v2):
            stored_filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex}.pdf"
            saved_path = get_session_upload_dir(session_id) / stored_filename
            file_storage.save(saved_path)
            saved_paths.append(saved_path)
            extracted = extract_pdf_content(saved_path, enable_ocr=True)
            if len(extracted["text"]) < current_app.config["MIN_RESUME_TEXT_LENGTH"]:
                raise ValueError(f"Could not extract enough text from {secure_filename(file_storage.filename)}.")
            texts.append(extracted["text"])
            diagnostics.append(extracted["diagnostics"])

        return compare_resume_versions(
            texts[0],
            texts[1],
            job_description=job_description,
            diagnostics_v1=diagnostics[0],
            diagnostics_v2=diagnostics[1],
        )
    finally:
        for path in saved_paths:
            path.unlink(missing_ok=True)


def merge_ai_notes(doctor_result: dict[str, Any], ai_result: dict[str, Any]) -> dict[str, Any]:
    if not ai_result:
        return doctor_result
    if ai_result.get("ai_provider"):
        doctor_result["ai_provider"] = ai_result["ai_provider"]
    if ai_result.get("ai_status"):
        doctor_result["ai_status"] = ai_result["ai_status"]
    if ai_result.get("error"):
        doctor_result["ai_status"] = "Resume analysis completed."

    doctor_result["ai_notes"] = {
        "summary": ai_result.get("summary", ""),
        "strengths": ai_result.get("strengths", []),
        "gaps": ai_result.get("gaps", []),
        "recommendations": ai_result.get("recommendations", []),
    }
    return doctor_result


def normalize_ai_fallback_status(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    error = str(result.get("error") or "")
    if "ANTHROPIC_API_KEY" in error or "Claude analysis is not configured" in error:
        result = {**result}
        result.pop("error", None)
        result["ai_status"] = "Resume analysis completed."
        result.setdefault("ai_provider", "local")
    return result


def build_score_data(result: dict[str, Any], fallback: int = 0) -> dict[str, Any]:
    scores = result.get("scores", {})
    overall = normalize_score(scores.get("overall"), fallback=fallback)
    return {
        "total_score": overall,
        "max_score": 100,
        "grade": grade_label(overall),
        "breakdown": result.get("score_breakdown", {}),
        "recommendations": result.get("recommendations", []),
    }


def grade_label(score: int) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Strong"
    if score >= 55:
        return "Needs focused improvement"
    return "Needs major improvement"


def prepare_analysis_for_view(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    contact_details = extract_contact_details(str(data.get("resume_text") or ""))
    data["created_at_display"] = format_timestamp(data.get("created_at"))
    data["created_at_relative"] = relative_timestamp(data.get("created_at"))
    data["email_masked"] = mask_email(contact_details.get("email") or data.get("email") or "")
    data["phone_masked"] = mask_phone(contact_details.get("phone") or data.get("phone") or "")
    data["contact_details"] = contact_details
    return data


def format_timestamp(value: Any) -> str:
    parsed = parse_datetime(value)
    if not parsed:
        return str(value or "")
    local_dt = parsed.astimezone(IST)
    return local_dt.strftime("%d %b %Y, %I:%M %p").lstrip("0")


def relative_timestamp(value: Any) -> str:
    parsed = parse_datetime(value)
    if not parsed:
        return ""
    now = datetime.now(timezone.utc)
    seconds = max(0, int((now - parsed.astimezone(timezone.utc)).total_seconds()))
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    if days < 30:
        return f"{days} day{'s' if days != 1 else ''} ago"
    return format_timestamp(value)


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def friendly_user_error(context: str) -> str:
    messages = {
        "analysis": "We could not analyze this resume right now. Please try a clear PDF or upload again in a moment.",
        "comparison": "We could not compare these resumes right now. Please check both PDFs and try again.",
        "chat": "The assistant could not answer right now. Please try again in a moment.",
        "jobs": "Job matches are running in demo mode right now. You can still review sample roles and filters.",
        "interview": "Interview questions could not be generated right now. Please try again in a moment.",
    }
    return messages.get(context, "Something went wrong. Please try again.")


INTERVIEW_SECTIONS = [
    "Most likely interview questions",
    "Technical questions from skills",
    "Project-based questions",
    "HR and behavioral questions",
    "Questions about weak areas or resume gaps",
    "Questions based on target job description",
]
INTERVIEW_CATEGORIES = {"HR", "Technical", "Project", "Experience", "Behavioral", "Resume Gap", "Role Specific"}
INTERVIEW_DIFFICULTIES = {"Easy", "Medium", "Hard"}
CATEGORY_SECTION_MAP = {
    "HR": "HR and behavioral questions",
    "Technical": "Technical questions from skills",
    "Project": "Project-based questions",
    "Experience": "Most likely interview questions",
    "Behavioral": "HR and behavioral questions",
    "Resume Gap": "Questions about weak areas or resume gaps",
    "Role Specific": "Questions based on target job description",
}


def generate_interview_questions(
    resume_text: str,
    analysis_data: dict[str, Any],
    target_role: str | None = None,
    job_description: str | None = None,
) -> dict[str, Any]:
    analysis_data = analysis_data if isinstance(analysis_data, dict) else {}
    profile = analysis_data.get("profile") if isinstance(analysis_data.get("profile"), dict) else {}
    resolved_role = str(target_role or profile.get("role") or "target role").strip()
    jd = str(job_description or "").strip()

    if current_app.config.get("GEMINI_API_KEY"):
        try:
            prompt = build_interview_questions_prompt(resume_text, analysis_data, resolved_role, jd)
            questions = call_gemini_interview_questions(prompt)
            if questions:
                return {
                    "target_role": resolved_role,
                    "questions": questions,
                    "sections": group_interview_questions(questions),
                    "ai_provider": "gemini",
                }
        except Exception:
            current_app.logger.info("Gemini interview question generation unavailable; using local fallback.")

    questions = build_local_interview_questions(resume_text, analysis_data, resolved_role, jd)
    return {
        "target_role": resolved_role,
        "questions": questions,
        "sections": group_interview_questions(questions),
        "ai_provider": "local",
    }


def build_interview_questions_prompt(
    resume_text: str,
    analysis_data: dict[str, Any],
    target_role: str,
    job_description: str,
) -> str:
    profile = analysis_data.get("profile") if isinstance(analysis_data.get("profile"), dict) else {}
    sections = analysis_data.get("sections") if isinstance(analysis_data.get("sections"), dict) else {}
    context = {
        "target_role": target_role,
        "candidate_level": analysis_data.get("candidate_level", ""),
        "profile": profile,
        "extracted_skills": ensure_string_list(analysis_data.get("extracted_skills"))[:16],
        "gaps": ensure_string_list(analysis_data.get("gaps"))[:8],
        "missing_sections": analysis_data.get("missing_sections", [])[:8] if isinstance(analysis_data.get("missing_sections"), list) else [],
        "projects": compact_text(str(sections.get("Projects", {}).get("excerpt", "")), 1200),
        "experience": compact_text(str(sections.get("Experience", {}).get("excerpt", "")), 1200),
        "education": compact_text(str(sections.get("Education", {}).get("excerpt", "")), 900),
        "certifications": compact_text(str(sections.get("Certifications", {}).get("excerpt", "")), 900),
    }
    return f"""
You are an interview coach for a Universal Resume Analyzer + Career Assistant.

Create personalized interview questions from the candidate's resume analysis and resume text.
Use the detected role, extracted skills, projects/case work, experience, education, certifications/licenses/training, weak areas, resume gaps, and target job description when present.

Return only a valid JSON array. Each item must use this shape:
[
  {{
    "section": "Most likely interview questions",
    "category": "Technical",
    "difficulty": "Medium",
    "question": "...",
    "why_asked": "...",
    "interviewer_checks": "...",
    "answer_approach": "...",
    "sample_answer": "...",
    "key_points": ["...", "..."],
    "mistake_to_avoid": "..."
  }}
]

Allowed section values:
{json.dumps(INTERVIEW_SECTIONS, ensure_ascii=False)}

Allowed category values:
["HR", "Technical", "Project", "Experience", "Behavioral", "Resume Gap", "Role Specific"]

Rules:
- Generate 16 to 22 questions.
- Include a practical mix across all relevant sections.
- Include "Questions based on target job description" only when a job description is provided.
- Keep questions domain-specific. Do not assume software/IT unless the resume profile is software/IT.
- Do not invent employers, degrees, projects, certifications, metrics, licenses, or dates.
- Personalize every answer using only evidence from the resume analysis, projects, experience, education, skills, gaps, and target job description.
- If project details are present, project-category sample answers must mention the relevant project/case-work details from the resume.
- If the candidate is fresher-level, sample answers must sound like a fresher and avoid claiming professional experience.
- If experience is present, sample answers should sound like an experienced candidate while staying within the resume evidence.
- If resume data is missing or weak, the sample answer should show how the candidate can answer honestly and positively.
- Behavioral sample answers must use STAR structure with Situation, Task, Action, and Result.
- Technical sample answers must be simple, practical, and interview-ready.
- Resume gap or weak-area sample answers must be honest, positive, and specific about improvement.
- sample_answer must sound like the candidate is speaking in the interview, not like a coach explaining what to say.
- Do not start sample_answer with "I would". Use natural first-person wording such as "My strongest example is...", "In this project...", or "A fair answer is...".
- Vary the wording and structure across questions. Do not reuse the same opening sentence for every answer.
- Suggested answer approaches should guide structure, not fabricate achievements.
- key_points must be a JSON array of short strings.
- Do not include markdown fences or commentary outside the JSON array.

Resume analysis context:
{json.dumps(context, ensure_ascii=False)}

Target job description:
{compact_text(job_description, 6000) or "No target job description provided."}

Resume text:
{compact_text(resume_text, 10000)}
""".strip()


def call_gemini_interview_questions(prompt: str) -> list[dict[str, Any]]:
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{current_app.config['GEMINI_MODEL']}:generateContent",
        params={"key": current_app.config["GEMINI_API_KEY"]},
        json=build_gemini_payload(
            system_prompt="Return only valid JSON for the requested interview question and answer schema.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max(current_app.config["GEMINI_MAX_TOKENS"], 7000),
            temperature=min(current_app.config["GEMINI_TEMPERATURE"], 0.35),
            response_mime_type="application/json",
        ),
        timeout=45.0,
    )
    response.raise_for_status()
    raw_text = extract_gemini_text(response.json())
    parsed = parse_interview_questions_json(raw_text)
    return normalize_interview_question_items(parsed)


def parse_interview_questions_json(raw_text: str) -> list[dict[str, Any]]:
    if not raw_text:
        return []
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    parsed: Any
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        questions = parsed.get("questions") or parsed.get("items") or parsed.get("interview_questions")
        if isinstance(questions, list):
            return [item for item in questions if isinstance(item, dict)]
        if parsed.get("question"):
            return [parsed]
    return []


def build_local_interview_questions(
    resume_text: str,
    analysis_data: dict[str, Any],
    target_role: str,
    job_description: str,
) -> list[dict[str, Any]]:
    profile = analysis_data.get("profile") if isinstance(analysis_data.get("profile"), dict) else {}
    role = target_role or str(profile.get("role") or "target role")
    candidate_level = str(analysis_data.get("candidate_level") or "candidate").replace("-", " ")
    skills = ensure_string_list(analysis_data.get("extracted_skills"))[:8]
    if not skills:
        skills = ensure_string_list(profile.get("target_skills"))[:6]
    sections = analysis_data.get("sections") if isinstance(analysis_data.get("sections"), dict) else {}
    project_excerpt = compact_text(str(sections.get("Projects", {}).get("excerpt", "")), 600)
    experience_excerpt = compact_text(str(sections.get("Experience", {}).get("excerpt", "")), 600)
    education_excerpt = compact_text(str(sections.get("Education", {}).get("excerpt", "")), 500)
    certification_excerpt = compact_text(str(sections.get("Certifications", {}).get("excerpt", "")), 500)
    gaps = ensure_string_list(analysis_data.get("gaps"))[:6]
    missing_sections = analysis_data.get("missing_sections", []) if isinstance(analysis_data.get("missing_sections"), list) else []

    def split_resume_points(excerpt: str, limit: int = 3) -> list[str]:
        points: list[str] = []
        for part in re.split(r"(?:\n|;|\.\s+|\u2022)", excerpt):
            cleaned = re.sub(r"\s+", " ", part).strip(" -:\t")
            if len(cleaned.split()) >= 3:
                points.append(compact_text(cleaned, 220))
            if len(points) >= limit:
                break
        return points

    def first_resume_detail(excerpt: str, fallback: str) -> str:
        points = split_resume_points(excerpt, limit=1)
        return points[0].rstrip(" .") if points else fallback

    skill_text = ", ".join(skills[:3]) if skills else "the role-relevant skills I can honestly discuss"
    project_detail = first_resume_detail(project_excerpt, "the project or practical work shown in my resume")
    experience_detail = first_resume_detail(experience_excerpt, "the responsibility or work exposure shown in my resume")
    education_detail = first_resume_detail(education_excerpt, "my education, coursework, training, or practical learning")
    certification_detail = first_resume_detail(certification_excerpt, "the certification, license, or training listed in my resume")
    has_experience = bool(experience_excerpt)
    is_fresher = candidate_level == "fresher"

    def best_evidence() -> str:
        if has_experience and not is_fresher:
            return experience_detail
        if project_excerpt:
            return project_detail
        if education_excerpt:
            return education_detail
        if certification_excerpt:
            return certification_detail
        return f"my verified resume evidence and skills such as {skill_text}"

    def local_sample_answer(category: str, focus: str = "", question_text: str = "") -> str:
        focus_text = focus.strip()
        if category == "Project":
            return (
                f"In my project work, the strongest example I can talk about is {project_detail}. "
                "The problem was to build or improve something practical, and my contribution was focused on the part I handled personally. "
                "I can explain the tools or methods I used, what I learned during the work, and the final deliverable or result that is actually visible in my resume. "
                "If a measurable result is not mentioned, I will keep it honest and talk about the deliverable, feedback, deployment status, grade, or learning outcome instead of adding numbers."
            )
        if category == "Technical":
            skill_focus = focus_text or "this skill"
            evidence = project_detail if project_excerpt else best_evidence()
            return (
                f"My practical exposure to {skill_focus} comes from {evidence}. "
                "I used it to handle a specific task, understand the flow, and solve the part I was responsible for. "
                "I can discuss what I did step by step, one challenge or decision from that work, and what the output taught me. "
                "If my exposure is project-based or academic, I will say that clearly and focus on the hands-on part I can defend."
            )
        if category == "Behavioral":
            evidence = best_evidence()
            return (
                f"Situation: In {evidence}, I had to handle a task that required learning, feedback, or adjustment. "
                "Task: My goal was to complete the work accurately and keep improving the quality. "
                "Action: I broke the task into smaller steps, used the resources and skills shown in my resume, asked for clarification or feedback where needed, and applied the learning. "
                "Result: I completed the deliverable or learning outcome I can honestly verify, and I will not add a number unless my resume already supports it."
            )
        if category == "Resume Gap":
            gap_label = focus_text or "this area"
            return (
                f"That is a fair area to ask about. My resume is still developing in {gap_label}, so I want to be honest about it. "
                f"The closest proof I can connect is {best_evidence()}. "
                "I have started strengthening this area through practice and learning, and my next step is to build clearer proof around it. "
                "I do not want to pretend I have experience I do not have, but I can show that I understand the gap and am working on it seriously."
            )
        if category == "Role Specific":
            return (
                f"The part of this role that matches me best is connected to {best_evidence()}. "
                f"My resume also supports this through skills like {skill_text}. "
                "For any requirement I have not used directly, I will be transparent and explain the closest related work I have done, then share how I would ramp up practically."
            )
        if category == "HR":
            level_phrase = "as a fresher" if is_fresher or not has_experience else "at this stage of my career"
            return (
                f"I am interested in {role} roles {level_phrase} because my resume points toward this direction through {best_evidence()} and skills such as {skill_text}. "
                "I enjoy work where I can apply these strengths practically and keep improving. "
                "This role feels like the right next step because it matches what I have already worked on and gives me a chance to contribute while learning more."
            )
        if "walk me through" in question_text.lower():
            if is_fresher or not has_experience:
                return (
                    f"I am a fresher-level candidate aiming for {role} roles. My strongest proof comes from {best_evidence()}, and I have built relevant skills such as {skill_text}. "
                    "I have tried to connect my learning with practical work, so I can explain what I built or practiced, what I learned, and where I want to improve next. "
                    f"That is why I see {role} as a good fit for my current direction."
                )
            return (
                f"My background is aligned with {role} roles through {experience_detail}. "
                f"Along with that experience, my resume shows skills such as {skill_text}. "
                "The main reason I see myself as a fit is that I can connect my practical work, ownership, and learning to the responsibilities of this role."
            )
        if "achievement" in question_text.lower() or "responsibility" in question_text.lower():
            return (
                f"The strongest example from my resume is {best_evidence()}. "
                "I chose this because it shows actual responsibility rather than just a keyword. "
                "My role was to understand the task, do my part clearly, and produce a result or learning outcome I can explain honestly. "
                "I would keep the answer focused on my ownership and avoid adding impact numbers that are not already supported."
            )
        if is_fresher or not has_experience:
            return (
                f"As a fresher, my readiness for {role} comes from {best_evidence()} and skills such as {skill_text}. "
                "I can talk about one concrete assignment, project, training item, or practical example, what I personally did, and what I learned from it. "
                "I am not claiming formal experience that is not in my resume; my focus is on learning quickly, communicating clearly, and applying the fundamentals well."
            )
        return (
            f"My resume shows experience around {experience_detail}. "
            "In that work, I focused on understanding the responsibility, taking clear action, and learning from the outcome. "
            f"The same strengths are relevant for {role}, especially when combined with skills such as {skill_text}. "
            "I will keep the result factual and mention only outcomes, tools, stakeholders, or metrics that my resume can support."
        )

    def local_key_points(category: str, focus: str = "") -> list[str]:
        focus_text = focus.strip()
        if category == "Project":
            return ["Project problem and goal", "Your exact contribution", "Tools or methods used", "Truthful result, deliverable, feedback, or learning"]
        if category == "Technical":
            return [focus_text or "The specific skill being tested", "Where you used or practiced it", "A practical challenge or decision", "Result or learning without unsupported claims"]
        if category == "Behavioral":
            return ["Situation", "Task", "Action you personally took", "Result or learning you can verify"]
        if category == "Resume Gap":
            return ["Brief honest acknowledgement", "Closest truthful resume evidence", "Improvement plan", "Positive role connection"]
        if category == "Role Specific":
            return ["Matched job requirement", "Resume proof", "Relevant skill or project", "Honest plan for weaker areas"]
        if is_fresher or not has_experience:
            return ["Fresher-level context", "Project, coursework, training, or practical proof", "Skills you can defend", "Learning mindset"]
        return ["Role or responsibility", "Your ownership", "Tools, methods, or stakeholders", "Truthful result or impact"]

    def local_mistake_to_avoid(category: str) -> str:
        mistakes = {
            "Project": "Do not claim full ownership of team work or add project metrics that are not in the resume.",
            "Technical": "Do not give only a textbook definition; connect the skill to hands-on resume evidence.",
            "Behavioral": "Do not give a vague story without Situation, Task, Action, and Result.",
            "Resume Gap": "Do not apologize at length, hide the gap, or pretend to have experience you do not have.",
            "Role Specific": "Do not force-fit job-description keywords that you cannot explain.",
            "HR": "Do not give a generic answer that could apply to any role.",
            "Experience": "Do not recite the resume line by line without explaining ownership and relevance.",
        }
        return mistakes.get(category, "Do not exaggerate achievements, numbers, or experience beyond the resume.")

    questions: list[dict[str, Any]] = []

    def add(
        section: str,
        category: str,
        difficulty: str,
        question: str,
        why_asked: str,
        interviewer_checks: str,
        answer_approach: str,
        focus: str = "",
    ) -> None:
        questions.append(
            {
                "section": section,
                "category": category,
                "difficulty": difficulty,
                "question": question,
                "why_asked": why_asked,
                "interviewer_checks": interviewer_checks,
                "answer_approach": answer_approach,
                "sample_answer": local_sample_answer(category, focus, question),
                "key_points": local_key_points(category, focus),
                "mistake_to_avoid": local_mistake_to_avoid(category),
            }
        )

    add(
        "Most likely interview questions",
        "Experience",
        "Easy",
        f"Walk me through your resume and explain why you are a fit for {role} roles.",
        "This is a common opening question to test clarity and role alignment.",
        "Communication, career direction, and whether your resume story matches the role.",
        "Use a 60-90 second structure: current profile, strongest resume evidence, relevant skills, and why this role is the logical next step.",
    )
    add(
        "Most likely interview questions",
        "Experience",
        "Medium",
        "Which achievement or responsibility on your resume best proves your readiness for this role?",
        "Interviewers want to see what you consider your strongest proof, not just a list of duties.",
        "Ownership, impact, judgment, and whether you can defend resume claims with details.",
        "Pick one concrete example. Explain the situation, your action, tools or methods used, and the result without adding unsupported numbers.",
    )
    if education_excerpt:
        add(
            "Most likely interview questions",
            "Experience",
            "Easy",
            f"How has your education prepared you for {role} responsibilities?",
            "Education is often used to validate fundamentals, especially for freshers and career switchers.",
            "Conceptual grounding, relevant coursework, and practical application.",
            "Connect one or two subjects, labs, clinical/practical work, or academic projects directly to the target role.",
        )

    for skill in skills[:5]:
        add(
            "Technical questions from skills",
            "Technical",
            "Medium",
            f"How have you used {skill} in a real project, internship, work task, or practical assignment?",
            "The skill appears in your resume, so the interviewer may check whether it is hands-on.",
            "Depth of skill, examples, constraints, and whether you can explain tradeoffs or outcomes.",
            f"Describe the context, what you personally did with {skill}, the result, and one lesson or improvement you would make next time.",
            focus=skill,
        )

    if skills:
        add(
            "Technical questions from skills",
            "Technical",
            "Hard",
            f"Which of your listed skills is strongest, and which one needs more practice for {role} interviews?",
            "Interviewers often probe self-awareness around skill depth.",
            "Honesty, prioritization, and a realistic learning plan.",
            "Name one strong skill with evidence, then one improving skill with a specific plan and recent practice example.",
            focus=", ".join(skills[:2]),
        )
    else:
        add(
            "Questions about weak areas or resume gaps",
            "Resume Gap",
            "Medium",
            "Your resume does not show many clear role-specific skills. Which skills are you ready to discuss confidently?",
            "A weak or unclear skills section usually triggers follow-up questions.",
            "Whether you can separate actual capability from keywords.",
            "Mention only skills you have used. Give a short example for each and explain what you are currently improving.",
        )

    if project_excerpt:
        add(
            "Project-based questions",
            "Project",
            "Medium",
            "Choose the strongest project or case work from your resume. What problem did it solve and what was your contribution?",
            "Project discussion reveals practical ownership beyond resume keywords.",
            "Problem understanding, individual contribution, tools/methods, and outcome clarity.",
            "Use problem, action, tools/methods, result. Be clear about your exact contribution versus team contribution.",
        )
        add(
            "Project-based questions",
            "Project",
            "Hard",
            "If you had to improve one project from your resume today, what would you change and why?",
            "This tests reflection, product thinking, engineering/role judgment, or process maturity.",
            "Ability to critique your work and choose meaningful improvements.",
            "Pick a realistic improvement: reliability, usability, documentation, validation, stakeholder impact, or measurable outcome.",
        )
    else:
        add(
            "Questions about weak areas or resume gaps",
            "Resume Gap",
            "Medium",
            f"Your resume has limited visible project or case-work evidence. How can you prove practical readiness for {role}?",
            "When project evidence is missing, interviewers look for alternate proof.",
            "Whether internships, practical training, coursework, freelance work, or work samples can support your readiness.",
            "Use any truthful practical work you have. Explain what you did, what you learned, and how it maps to the role.",
        )

    if experience_excerpt:
        add(
            "Most likely interview questions",
            "Experience",
            "Medium",
            "Tell me about one responsibility from your experience section where you had clear ownership.",
            "Experience bullets are often tested for real ownership.",
            "Scope, responsibility, collaboration, and result orientation.",
            "Choose one responsibility. State your role, stakeholders, actions, and the outcome or quality bar you met.",
        )
    elif candidate_level == "fresher":
        add(
            "Most likely interview questions",
            "Experience",
            "Easy",
            "As a fresher, how will you compensate for limited formal work experience?",
            "Freshers are often asked how they convert learning into job readiness.",
            "Practical mindset, learning speed, and confidence without overclaiming.",
            "Point to projects, internships, coursework, training, volunteering, or self-practice, then explain how you will ramp up.",
        )

    if certification_excerpt:
        add(
            "Most likely interview questions",
            "Experience",
            "Easy",
            "Which certification, license, or training on your resume is most relevant to this role?",
            "Certifications and training can validate readiness when they are role-relevant.",
            "Whether the credential reflects usable knowledge, not just a line on the resume.",
            "Explain what the certification covered, where you applied it, and how it supports the target role.",
        )
    elif any(str(item.get("section", "")).lower() == "certifications" for item in missing_sections if isinstance(item, dict)):
        add(
            "Questions about weak areas or resume gaps",
            "Resume Gap",
            "Medium",
            "Your resume does not clearly show certifications or licenses. Is that a gap for your target role?",
            "For some roles, missing certifications or licenses can affect shortlisting.",
            "Awareness of role requirements and a plan to close credential gaps.",
            "Be honest. If required, mention your plan and timeline. If not required, explain your strongest alternative proof.",
        )

    add(
        "HR and behavioral questions",
        "Behavioral",
        "Easy",
        "Tell me about a time you had to learn something quickly to complete a task.",
        "This checks adaptability and learning speed.",
        "How you approach unfamiliar work, ask for help, and apply feedback.",
        "Use a specific example from work, academics, projects, or training. Show the learning process and result.",
    )
    add(
        "HR and behavioral questions",
        "Behavioral",
        "Medium",
        "Describe a time you received feedback and changed your work because of it.",
        "Interviewers want evidence that you can improve without becoming defensive.",
        "Coachability, maturity, and quality mindset.",
        "Share the feedback, what you changed, and how the final work improved.",
    )
    add(
        "HR and behavioral questions",
        "HR",
        "Easy",
        f"Why are you interested in {role} roles at this stage of your career?",
        "This tests motivation and whether your goals fit the role.",
        "Role understanding, career intent, and stability.",
        "Connect your resume evidence to the role, then mention the kind of work environment or learning path you want.",
    )

    for gap in gaps[:4]:
        add(
            "Questions about weak areas or resume gaps",
            "Resume Gap",
            "Medium",
            f"Your resume suggests this possible gap: {gap}. How would you address it in an interview?",
            "Interviewers may ask directly about visible weak areas or missing proof.",
            "Self-awareness, honesty, and whether you have a practical improvement plan.",
            "Acknowledge the gap briefly, give any truthful evidence that reduces concern, and state the next step you are taking.",
            focus=gap,
        )

    if job_description.strip():
        missing_keywords = find_missing_keywords(resume_text, job_description)[:4]
        add(
            "Questions based on target job description",
            "Role Specific",
            "Medium",
            f"Which requirements in this target job description match your resume most strongly for {role}?",
            "JD-specific interviews test whether you understand the role beyond your resume.",
            "Role fit, keyword alignment, and ability to map evidence to responsibilities.",
            "Choose two or three JD requirements. Match each to a resume skill, project, experience, education, or training example.",
        )
        add(
            "Questions based on target job description",
            "Role Specific",
            "Hard",
            "Which part of this job description is least supported by your resume, and how are you closing that gap?",
            "Interviewers may test risk areas before deciding fit.",
            "Honesty, prioritization, and a realistic upskilling or proof-building plan.",
            "Name one weaker area, mention any adjacent experience, and describe your concrete next practice or learning step.",
        )
        for keyword in missing_keywords[:3]:
            add(
                "Questions based on target job description",
                "Role Specific",
                "Medium",
                f"The job description emphasizes {keyword}. What related experience or learning can you discuss?",
                "Missing JD terms can become interview probes.",
                "Adjacent experience, transferability, and whether the gap is manageable.",
                "Use the closest truthful example. If you have not used it, say so and explain how you would learn or apply it.",
                focus=keyword,
            )

    return normalize_interview_question_items(questions)


def normalize_interview_question_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        category = normalize_interview_category(item.get("category") or item.get("type"))
        difficulty = normalize_interview_difficulty(item.get("difficulty"))
        section = normalize_interview_section(item.get("section"), category)
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        why_asked = str(item.get("why_asked") or item.get("why") or "").strip()
        interviewer_checks = str(item.get("interviewer_checks") or item.get("what_interviewer_checks") or item.get("checks") or "").strip()
        answer_approach = str(item.get("answer_approach") or item.get("suggested_answer_approach") or item.get("approach") or "").strip()
        sample_answer = str(
            item.get("sample_answer")
            or item.get("sampleAnswer")
            or item.get("full_sample_answer")
            or item.get("fullSampleAnswer")
            or item.get("full_answer")
            or item.get("answer")
            or ""
        ).strip()
        key_points = normalize_interview_key_points(
            item.get("key_points")
            or item.get("keyPoints")
            or item.get("key_points_to_include")
            or item.get("keyPointsToInclude")
            or item.get("points_to_include")
            or item.get("answer_key_points")
        )
        mistake_to_avoid = str(
            item.get("mistake_to_avoid")
            or item.get("mistakeToAvoid")
            or item.get("common_mistake_to_avoid")
            or item.get("common_mistake")
            or item.get("mistake")
            or ""
        ).strip()
        normalized.append(
            {
                "section": section,
                "category": category,
                "difficulty": difficulty,
                "question": question,
                "why_asked": why_asked or "This follows from the resume, target role, or job-description context.",
                "interviewer_checks": interviewer_checks or "The interviewer wants to verify depth, clarity, and truthful resume evidence.",
                "answer_approach": answer_approach or "Answer with a concise example: context, your action, tools or methods, and the outcome or learning.",
                "sample_answer": sample_answer,
                "key_points": key_points,
                "mistake_to_avoid": mistake_to_avoid,
            }
        )
    return normalized[:24]


def normalize_interview_key_points(value: Any) -> list[str]:
    raw_points: list[Any]
    if isinstance(value, (list, tuple)):
        raw_points = list(value)
    elif isinstance(value, str):
        separator = r"\n|;"
        raw_points = re.split(separator, value)
    else:
        raw_points = []

    points: list[str] = []
    for point in raw_points:
        if isinstance(point, dict):
            text = str(point.get("point") or point.get("text") or point.get("label") or "").strip()
        else:
            text = str(point or "").strip()
        text = re.sub(r"^\s*[-*\d.)]+\s*", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            points.append(text[:180])
        if len(points) >= 8:
            break
    return points


def hydrate_missing_interview_answer_fields(questions: list[dict[str, Any]], analysis_id: int | None) -> list[dict[str, Any]]:
    if not questions or not any(
        not question.get("sample_answer") or not question.get("key_points") or not question.get("mistake_to_avoid")
        or looks_like_coach_style_interview_answer(question.get("sample_answer"))
        for question in questions
    ):
        return questions

    analysis = get_analysis(analysis_id) if analysis_id else None
    if analysis is None:
        return questions

    analysis_data = normalize_ai_fallback_status(parse_json_field(analysis["claude_result"], {}))
    if not analysis_data.get("profile"):
        analysis_data = analyze_resume_doctor(analysis["resume_text"], job_description=analysis["job_description"] or "")

    context = build_interview_answer_context(analysis_data)
    hydrated: list[dict[str, Any]] = []
    for question in questions:
        answer_fields = build_fallback_interview_answer_fields(question, context)
        should_replace_sample = not question.get("sample_answer") or looks_like_coach_style_interview_answer(question.get("sample_answer"))
        hydrated.append(
            {
                **question,
                "sample_answer": answer_fields["sample_answer"] if should_replace_sample else question.get("sample_answer"),
                "key_points": question.get("key_points") or answer_fields["key_points"],
                "mistake_to_avoid": question.get("mistake_to_avoid") or answer_fields["mistake_to_avoid"],
            }
        )
    return hydrated


def looks_like_coach_style_interview_answer(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if not text:
        return False
    coach_openers = (
        "i would discuss",
        "i would answer",
        "i would address",
        "i would map",
        "i would connect",
        "i would explain",
        "in the interview i would",
        "as a fresher, i would",
    )
    return text.startswith(coach_openers) or text.count("i would") >= 2


def build_interview_answer_context(analysis_data: dict[str, Any]) -> dict[str, Any]:
    profile = analysis_data.get("profile") if isinstance(analysis_data.get("profile"), dict) else {}
    role = str(profile.get("role") or "target role").strip() or "target role"
    candidate_level = str(analysis_data.get("candidate_level") or "candidate").replace("-", " ").strip().lower()
    skills = ensure_string_list(analysis_data.get("extracted_skills"))[:8]
    if not skills:
        skills = ensure_string_list(profile.get("target_skills"))[:6]
    sections = analysis_data.get("sections") if isinstance(analysis_data.get("sections"), dict) else {}
    gaps = ensure_string_list(analysis_data.get("gaps"))[:6]

    project_detail = first_interview_resume_detail(
        compact_text(str(sections.get("Projects", {}).get("excerpt", "")), 600),
        "the project or practical work shown in my resume",
    )
    experience_detail = first_interview_resume_detail(
        compact_text(str(sections.get("Experience", {}).get("excerpt", "")), 600),
        "the responsibility or work exposure shown in my resume",
    )
    education_detail = first_interview_resume_detail(
        compact_text(str(sections.get("Education", {}).get("excerpt", "")), 500),
        "my education, coursework, training, or practical learning",
    )
    certification_detail = first_interview_resume_detail(
        compact_text(str(sections.get("Certifications", {}).get("excerpt", "")), 500),
        "the certification, license, or training listed in my resume",
    )

    return {
        "role": role,
        "candidate_level": candidate_level,
        "is_fresher": candidate_level == "fresher",
        "skills": skills,
        "skill_text": ", ".join(skills[:3]) if skills else "the role-relevant skills I can honestly discuss",
        "gaps": gaps,
        "has_projects": bool(str(sections.get("Projects", {}).get("excerpt", "")).strip()),
        "has_experience": bool(str(sections.get("Experience", {}).get("excerpt", "")).strip()),
        "project_detail": project_detail,
        "experience_detail": experience_detail,
        "education_detail": education_detail,
        "certification_detail": certification_detail,
    }


def first_interview_resume_detail(excerpt: str, fallback: str) -> str:
    for part in re.split(r"(?:\n|;|\.\s+|\u2022)", excerpt):
        cleaned = re.sub(r"\s+", " ", part).strip(" -:\t.")
        if len(cleaned.split()) >= 3:
            return compact_text(cleaned, 220).rstrip(" .")
    return fallback


def build_fallback_interview_answer_fields(question: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    category = normalize_interview_category(question.get("category"))
    focus = infer_interview_answer_focus(question, context)
    role = context["role"]
    skill_text = context["skill_text"]
    evidence = best_interview_evidence(category, context)
    question_text = str(question.get("question") or "").lower()

    if category == "Project":
        sample_answer = (
            f"In my project work, the strongest example I can talk about is {context['project_detail']}. "
            "The main problem was to create or improve something practical, and my contribution was the part I handled personally. "
            "I can explain the tools or methods I used, the challenge I faced, and the final deliverable or learning that is actually visible from my resume. "
            "If a measurable result is not mentioned, I will keep the answer honest and talk about the truthful outcome instead of adding numbers."
        )
    elif category == "Technical":
        sample_answer = (
            f"My practical exposure to {focus or 'this skill'} comes from {evidence}. "
            "I used or practiced it in a real task, so I can explain the flow, the part I handled, and one decision or issue I worked through. "
            "If my exposure is project-based or academic, I will say that clearly and focus on the hands-on part I can defend."
        )
    elif category == "Behavioral":
        sample_answer = (
            f"Situation: In {evidence}, I had to handle a task that needed learning, feedback, or adjustment. "
            "Task: My goal was to complete the work properly and improve the quality. "
            f"Action: I used the resume evidence I can defend, including {skill_text}, broke the work into steps, and applied feedback where needed. "
            "Result: I completed the deliverable or learning outcome I can honestly verify, without adding unsupported numbers."
        )
    elif category == "Resume Gap":
        sample_answer = (
            f"That is a fair point. My resume is lighter on {focus or 'this area'}, so I want to address it honestly. "
            f"The closest evidence I can connect is {evidence}. "
            "I am strengthening this area through practice and clearer proof of work, and I am comfortable explaining what I can do today and what I am still improving."
        )
    elif category == "Role Specific":
        sample_answer = (
            f"The role requirement connects best with {evidence}. "
            f"My resume also supports this through skills like {skill_text}. "
            "For any requirement I have not used directly, I will be transparent, explain the closest related exposure, and share how I would ramp up practically."
        )
    elif category == "HR":
        level_phrase = "as a fresher" if context["is_fresher"] or not context["has_experience"] else "at this stage of my career"
        sample_answer = (
            f"I am interested in {role} roles {level_phrase} because my resume already points in that direction through {evidence} and skills such as {skill_text}. "
            "I enjoy work where I can apply those strengths practically and keep improving. "
            "This role feels like a logical next step because it matches what I have already worked on and gives me room to contribute while learning more."
        )
    elif "walk me through" in question_text:
        if context["is_fresher"] or not context["has_experience"]:
            sample_answer = (
                f"I am a fresher-level candidate aiming for {role} roles. My strongest proof comes from {evidence}, and I have built relevant skills such as {skill_text}. "
                "I have tried to connect my learning with practical work, so I can explain what I built or practiced, what I learned, and where I want to improve next. "
                f"That is why I see {role} as a good fit for my current direction."
            )
        else:
            sample_answer = (
                f"My background is aligned with {role} roles through {context['experience_detail']}. "
                f"Along with that experience, my resume shows skills such as {skill_text}. "
                "The main reason I see myself as a fit is that I can connect my practical work, ownership, and learning to the responsibilities of this role."
            )
    elif "achievement" in question_text or "responsibility" in question_text or "ownership" in question_text:
        sample_answer = (
            f"The strongest example from my resume is {evidence}. "
            "I chose this because it shows actual responsibility rather than just a keyword. "
            "My role was to understand the task, do my part clearly, and produce a result or learning outcome I can explain honestly. "
            "I will keep the answer focused on my ownership and avoid adding impact numbers that are not already supported."
        )
    elif context["is_fresher"] or not context["has_experience"]:
        sample_answer = (
            f"As a fresher, my readiness for {role} comes from {evidence} and skills such as {skill_text}. "
            "I can describe one concrete project, coursework, training item, or practical example, what I personally did, and what I learned from it. "
            "I am not claiming formal work experience that is not in my resume; I am showing that I can learn quickly and apply the fundamentals well."
        )
    else:
        sample_answer = (
            f"My resume shows experience around {context['experience_detail']}. "
            "In that work, I focused on understanding the responsibility, taking clear action, and learning from the outcome. "
            f"The same strengths are relevant for {role}, especially when combined with skills such as {skill_text}. "
            "I will keep the result factual and mention only outcomes, tools, stakeholders, or metrics that my resume supports."
        )

    return {
        "sample_answer": sample_answer,
        "key_points": fallback_interview_key_points(category, context, focus),
        "mistake_to_avoid": fallback_interview_mistake_to_avoid(category),
    }


def infer_interview_answer_focus(question: dict[str, Any], context: dict[str, Any]) -> str:
    question_text = str(question.get("question") or "").lower()
    for skill in context.get("skills", []):
        if str(skill).lower() in question_text:
            return str(skill)
    for gap in context.get("gaps", []):
        if str(gap).lower()[:32] in question_text:
            return str(gap)
    if question.get("category") == "Resume Gap" and context.get("gaps"):
        return str(context["gaps"][0])
    return ""


def best_interview_evidence(category: str, context: dict[str, Any]) -> str:
    if category in {"Project", "Technical"} and context["has_projects"]:
        return context["project_detail"]
    if context["has_experience"] and not context["is_fresher"]:
        return context["experience_detail"]
    if context["has_projects"]:
        return context["project_detail"]
    if context["education_detail"]:
        return context["education_detail"]
    return context["certification_detail"]


def fallback_interview_key_points(category: str, context: dict[str, Any], focus: str = "") -> list[str]:
    if category == "Project":
        return ["Project problem and goal", "Your exact contribution", "Tools or methods used", "Truthful result, deliverable, feedback, or learning"]
    if category == "Technical":
        return [focus or "Skill being tested", "Where you used or practiced it", "A practical challenge or decision", "Result or learning without unsupported claims"]
    if category == "Behavioral":
        return ["Situation", "Task", "Action you personally took", "Result or learning you can verify"]
    if category == "Resume Gap":
        return ["Brief honest acknowledgement", "Closest truthful resume evidence", "Improvement plan", "Positive role connection"]
    if category == "Role Specific":
        return ["Matched job requirement", "Resume proof", "Relevant skill or project", "Honest plan for weaker areas"]
    if context["is_fresher"] or not context["has_experience"]:
        return ["Fresher-level context", "Project, coursework, training, or practical proof", "Skills you can defend", "Learning mindset"]
    return ["Role or responsibility", "Your ownership", "Tools, methods, or stakeholders", "Truthful result or impact"]


def fallback_interview_mistake_to_avoid(category: str) -> str:
    mistakes = {
        "Project": "Do not claim full ownership of team work or add project metrics that are not in the resume.",
        "Technical": "Do not give only a textbook definition; connect the skill to hands-on resume evidence.",
        "Behavioral": "Do not give a vague story without Situation, Task, Action, and Result.",
        "Resume Gap": "Do not hide the gap or pretend to have experience you do not have.",
        "Role Specific": "Do not force-fit job-description keywords that you cannot explain.",
        "HR": "Do not give a generic answer that could apply to any role.",
        "Experience": "Do not recite the resume line by line without explaining ownership and relevance.",
    }
    return mistakes.get(category, "Do not exaggerate achievements, numbers, or experience beyond the resume.")


def normalize_interview_category(value: Any) -> str:
    text = str(value or "").replace("_", " ").replace("-", " ").strip().lower()
    mapping = {
        "technical": "Technical",
        "tech": "Technical",
        "project": "Project",
        "projects": "Project",
        "experience": "Experience",
        "work experience": "Experience",
        "behavioral": "Behavioral",
        "behavioural": "Behavioral",
        "hr": "HR",
        "resume gap": "Resume Gap",
        "gap": "Resume Gap",
        "role specific": "Role Specific",
        "jd": "Role Specific",
        "job description": "Role Specific",
    }
    return mapping.get(text, str(value).strip().title() if str(value).strip().title() in INTERVIEW_CATEGORIES else "HR")


def normalize_interview_difficulty(value: Any) -> str:
    text = str(value or "Medium").strip().lower()
    mapping = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}
    return mapping.get(text, "Medium")


def normalize_interview_section(value: Any, category: str) -> str:
    text = str(value or "").strip()
    for section in INTERVIEW_SECTIONS:
        if text.lower() == section.lower():
            return section
    lower = text.lower()
    if "technical" in lower or "skill" in lower:
        return "Technical questions from skills"
    if "project" in lower or "case" in lower:
        return "Project-based questions"
    if "behavior" in lower or "behaviour" in lower or "hr" in lower:
        return "HR and behavioral questions"
    if "gap" in lower or "weak" in lower:
        return "Questions about weak areas or resume gaps"
    if "job description" in lower or "jd" in lower or "role specific" in lower:
        return "Questions based on target job description"
    return CATEGORY_SECTION_MAP.get(category, "Most likely interview questions")


def group_interview_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = []
    for section in INTERVIEW_SECTIONS:
        section_questions = [question for question in questions if question.get("section") == section]
        if section_questions:
            grouped.append({"title": section, "questions": section_questions})
    return grouped


def ensure_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def compact_text(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def mask_email(email: str) -> str:
    if not email or "@" not in email:
        return ""
    name, domain = email.split("@", 1)
    visible = name[:2] if len(name) > 2 else name[:1]
    return f"{visible}{'*' * max(3, len(name) - len(visible))}@{domain}"


def mask_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) < 4:
        return ""
    return f"{'*' * max(4, len(digits) - 4)}{digits[-4:]}"


def mask_private_resume_text(text: str) -> str:
    details = extract_contact_details(text)
    masked = text
    if details.get("email"):
        masked = masked.replace(details["email"], mask_email(details["email"]) or "[email hidden]")
    if details.get("phone"):
        masked = masked.replace(details["phone"], mask_phone(details["phone"]) or "[phone hidden]")
    return masked


def build_resume_preview_lines(text: str, mask: bool = True, limit: int = 220) -> list[dict[str, Any]]:
    source = mask_private_resume_text(text) if mask else text
    heading_aliases = {
        "summary": "Summary",
        "profile": "Summary",
        "objective": "Summary",
        "skills": "Skills",
        "technical skills": "Skills",
        "experience": "Experience",
        "work experience": "Experience",
        "internship": "Experience",
        "projects": "Projects",
        "academic projects": "Projects",
        "education": "Education",
        "certifications": "Certifications",
        "achievements": "Achievements",
        "portfolio": "Portfolio",
        "links": "Portfolio",
    }
    preview: list[dict[str, Any]] = []
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = re.sub(r"[:|/\-]+$", "", line.lower()).strip()
        section = heading_aliases.get(normalized)
        preview.append({"text": line[:500], "is_heading": bool(section), "section": section or ""})
        if len(preview) >= limit:
            break
    return preview


def generate_domain_rewrites(before: str, section: str, profile: dict[str, Any]) -> list[str]:
    role = str(profile.get("role") or "target role")
    sub_domain = str(profile.get("sub_domain") or "your field").lower()
    skills = [str(skill) for skill in profile.get("target_skills", [])[:3]]
    skill_text = ", ".join(skills) if skills else "role-relevant strengths"
    before_clean = re.sub(r"\s+", " ", before).strip(" -.")
    has_metric = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:%|percent|x|k|m|hours|days|weeks|months|users|students|patients|projects)\b", before_clean, re.I))
    if section.lower() == "summary":
        return [
            f"{role} candidate focused on {sub_domain}, highlighting verified strengths in {skill_text} and clear fit for {role} roles.",
            f"Early-career {role} profile with hands-on {sub_domain} proof, {skill_text}, and a resume narrative recruiters can verify.",
        ]
    if section.lower() == "skills":
        return [
            f"Core {role} skills: {skill_text}. Add separate groups for tools, frameworks, domain methods, and supporting skills.",
            "Keep skills interview-ready: include only tools you have used in coursework, internships, projects, training, or work.",
        ]
    if section.lower() == "projects":
        result = "the existing measurable result" if has_metric else "a truthful result such as scope, users, deployment, grade, or learning outcome if available"
        return [
            f"Built {sub_domain} work using {skill_text}; clarify the problem, your role, the technology, and {result}.",
            "Use action + technology + scope + result. Do not add numbers unless the resume already supports them.",
        ]
    if section.lower() == "experience":
        return [
            f"{before_clean}; strengthen this with ownership, tools/methods, timeline, and a truthful result.",
            "If there is no formal job experience, frame internships, major projects, freelance work, hackathons, or practical training as experience alternatives.",
        ]
    return [
        f"Delivered {sub_domain} work using {skill_text}, clarifying scope, ownership, and a truthful outcome for the target {role} role.",
        f"{before_clean}; strengthened the bullet by adding action, context, tools, and a specific result.",
        "Add a metric only if truthful; otherwise state the observable deliverable, quality bar, or learning outcome.",
    ]


def merge_unique(first: list[Any], second: list[Any], limit: int = 10) -> list[str]:
    merged: list[str] = []
    for item in [*(first or []), *(second or [])]:
        value = str(item).strip()
        if value and value not in merged:
            merged.append(value)
        if len(merged) >= limit:
            break
    return merged


def parse_optional_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]


def extract_pdf_text(path: Path) -> str:
    return extract_pdf_content(path, enable_ocr=True)["text"]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_resume_features(text: str) -> dict[str, Any]:
    nlp = get_nlp(current_app.config["SPACY_MODEL"])
    doc = nlp(text[: nlp.max_length])
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    contact_details = extract_contact_details(text)
    email = contact_details.get("email") if contact_details.get("email_status") == "valid" else extract_email(text)
    phone = contact_details.get("phone") or extract_phone(text)
    skills = extract_skills(text)
    education = extract_education(lines)
    experience_years = extract_experience_years(text)

    return {
        "candidate_name": extract_candidate_name(doc, lines, email),
        "email": email,
        "phone": phone,
        "contact_details": contact_details,
        "skills": skills,
        "education": education,
        "experience_years": experience_years,
        "keywords": extract_keywords(text),
    }


@lru_cache(maxsize=2)
def get_nlp(model_name: str):
    try:
        nlp = spacy.load(model_name)
    except OSError:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    nlp.max_length = max(nlp.max_length, 2_000_000)
    return nlp


def extract_email(text: str) -> str | None:
    match = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text)
    return match.group(0).lower() if match else None


def extract_phone(text: str) -> str | None:
    match = re.search(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{4}", text)
    return match.group(0).strip() if match else None


def extract_candidate_name(doc, lines: list[str], email: str | None) -> str | None:
    for ent in doc.ents:
        if ent.label_ == "PERSON" and 2 <= len(ent.text.split()) <= 4:
            return ent.text.strip()

    for line in lines[:8]:
        lower_line = line.lower()
        if email and email in lower_line:
            continue
        if any(token in lower_line for token in ("resume", "curriculum vitae", "email", "phone", "linkedin")):
            continue
        words = re.findall(r"[A-Za-z]+", line)
        if 2 <= len(words) <= 4 and all(word[:1].isupper() for word in words):
            return " ".join(words)
    return None


def extract_skills(text: str) -> list[str]:
    normalized = text.lower()
    found = []
    for skill in COMMON_SKILLS:
        pattern = r"(?<![\w+#.])" + re.escape(skill.lower()) + r"(?![\w+#.])"
        if re.search(pattern, normalized):
            found.append(skill)
    return sorted(found)


def extract_education(lines: list[str]) -> list[str]:
    education_lines = []
    for line in lines:
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in EDUCATION_KEYWORDS):
            education_lines.append(line)
        if len(education_lines) >= 5:
            break
    return education_lines


def extract_experience_years(text: str) -> int | None:
    patterns = [
        r"(\d{1,2})\+?\s*(?:years|yrs)\s+(?:of\s+)?experience",
        r"experience\s*(?:of\s*)?(\d{1,2})\+?\s*(?:years|yrs)",
    ]
    values = []
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            values.append(int(match))
    return max(values) if values else None


def extract_keywords(text: str, limit: int = 25) -> list[str]:
    words = re.findall(r"\b[a-zA-Z][a-zA-Z+#.-]{2,}\b", text.lower())
    counts: dict[str, int] = {}
    for word in words:
        if word in STOPWORDS_FOR_KEYWORDS or word.isdigit():
            continue
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:limit]]


def analyze_with_claude(resume_text: str, job_description: str, local_features: dict[str, Any]) -> dict[str, Any]:
    prompt = build_claude_prompt(resume_text, job_description, local_features)

    if current_app.config.get("ANTHROPIC_API_KEY"):
        try:
            return call_claude_analysis(prompt)
        except Exception as exc:
            current_app.logger.info("Claude analysis unavailable; trying Gemini/local fallback: %s", exc)

    if current_app.config.get("GEMINI_API_KEY"):
        try:
            return call_gemini_analysis(prompt)
        except Exception as exc:
            current_app.logger.info("Gemini analysis unavailable; using local fallback: %s", exc)

    result = local_only_result(resume_text, job_description, local_features)
    result["ai_provider"] = "local"
    result["ai_status"] = "Resume analysis completed."
    return result


def call_claude_analysis(prompt: str) -> dict[str, Any]:
    client = anthropic.Anthropic(api_key=current_app.config["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=current_app.config["CLAUDE_MODEL"],
        max_tokens=current_app.config["CLAUDE_MAX_TOKENS"],
        temperature=current_app.config["CLAUDE_TEMPERATURE"],
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = extract_claude_text(message)
    parsed = parse_model_json(raw_text)
    if parsed is None:
        raise ValueError("Claude response was not valid JSON.")
    normalized = normalize_claude_result(parsed)
    normalized["ai_provider"] = "anthropic"
    return normalized


def call_gemini_analysis(prompt: str) -> dict[str, Any]:
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{current_app.config['GEMINI_MODEL']}:generateContent",
        params={"key": current_app.config["GEMINI_API_KEY"]},
        json=build_gemini_payload(
            system_prompt="Return only valid JSON for the requested resume analysis schema.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=current_app.config["GEMINI_MAX_TOKENS"],
            temperature=current_app.config["GEMINI_TEMPERATURE"],
            response_mime_type="application/json",
        ),
        timeout=25.0,
    )
    response.raise_for_status()
    raw_text = extract_gemini_text(response.json())
    parsed = parse_model_json(raw_text)
    if parsed is None:
        raise ValueError("Gemini response was not valid JSON.")
    normalized = normalize_claude_result(parsed)
    normalized["ai_provider"] = "gemini"
    return normalized


def build_claude_prompt(resume_text: str, job_description: str, local_features: dict[str, Any]) -> str:
    compact_resume = resume_text[:12000]
    compact_job = job_description.strip()[:6000]
    detected_profile = local_features.get("detected_profile", {})
    return f"""
You are a Universal Resume Analyzer + Career Assistant for every industry, not only IT.

Highest priority rule:
- First respect the detected industry and role.
- Never give IT/software recommendations unless the detected profile is actually software/IT.
- For doctors, teachers, HR, mechanical, civil, legal, finance, marketing, healthcare, research, and business profiles, use only that domain's language.

Analyze the resume against the job description. Return only valid JSON with this schema:
{{
  "summary": "2-4 sentence hiring-focused summary",
  "match_score": 0,
  "strengths": ["specific strength"],
  "gaps": ["specific gap"],
  "recommendations": ["specific resume improvement"],
  "matched_keywords": ["keyword"],
  "missing_keywords": ["keyword"],
  "suggested_roles": ["role title"]
}}

Rules:
- match_score must be an integer from 0 to 100.
- Be concrete and useful, not generic.
- If no job description is provided, score overall resume quality instead.
- Do not include markdown fences or text outside the JSON object.
- Do not invent degrees, employers, certifications, metrics, licenses, or projects.
- For freshers, treat internships, academic projects, freelance work, hackathons, practical training, and major projects as experience alternatives.
- If metrics are missing, suggest examples with "if truthful" language instead of inventing numbers.
- Detect invalid contact details separately from missing contact details.

Local parser features:
{json.dumps(local_features, ensure_ascii=False)}

Detected profile:
{json.dumps(detected_profile, ensure_ascii=False)}

Job description:
{compact_job or "No job description provided."}

Resume:
{compact_resume}
""".strip()


def extract_claude_text(message: Any) -> str:
    parts = []
    for block in getattr(message, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts).strip()


def parse_model_json(raw_text: str) -> dict[str, Any] | None:
    if not raw_text:
        return None

    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_claude_result(result: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "summary": "",
        "match_score": 0,
        "strengths": [],
        "gaps": [],
        "recommendations": [],
        "matched_keywords": [],
        "missing_keywords": [],
        "suggested_roles": [],
    }
    normalized = {**defaults, **result}
    normalized["match_score"] = normalize_score(normalized.get("match_score"), fallback=0)

    for key in ("strengths", "gaps", "recommendations", "matched_keywords", "missing_keywords", "suggested_roles"):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = [value]
        elif not isinstance(value, list):
            normalized[key] = []

    return normalized


NON_TECH_AVOID_TERMS = {
    "react",
    "node",
    "java",
    "javascript",
    "python",
    "docker",
    "kubernetes",
    "api",
    "backend",
    "frontend",
    "dsa",
    "github",
    "git",
    "leetcode",
}


def filter_domain_terms(items: list[str], profile: dict[str, Any]) -> list[str]:
    if str(profile.get("industry") or "").lower() in {"software & it", "data & analytics"}:
        return [item for item in items if str(item).strip()]

    filtered: list[str] = []
    for item in items:
        text = str(item).strip()
        normalized = text.lower()
        if not text:
            continue
        if any(re.search(rf"(?<![a-z0-9+#]){re.escape(term)}(?![a-z0-9+#])", normalized) for term in NON_TECH_AVOID_TERMS):
            continue
        filtered.append(text)
    return filtered


def local_only_result(resume_text: str, job_description: str, local_features: dict[str, Any]) -> dict[str, Any]:
    score = calculate_local_match_score(resume_text, job_description)
    profile = local_features.get("detected_profile") if isinstance(local_features.get("detected_profile"), dict) else {}
    role = str(profile.get("role") or "target role")
    industry = str(profile.get("industry") or "career field")
    skills = local_features.get("domain_skills") or local_features.get("skills", [])
    skills = filter_domain_terms([str(skill) for skill in skills], profile)
    summary_parts = [
        f"Detected a {role} profile in {industry} with {len(skills)} visible role-relevant skill(s).",
        "Compared resume terms with the job description using a local keyword matcher."
        if job_description.strip()
        else "No job description was provided, so this is a basic resume-quality analysis.",
    ]
    return normalize_claude_result(
        {
            "summary": " ".join(summary_parts),
            "match_score": score,
            "strengths": [f"Includes {skill} experience." for skill in skills[:5]],
            "gaps": filter_domain_terms(build_local_gaps(resume_text, job_description, profile), profile),
            "recommendations": filter_domain_terms(build_local_recommendations(profile, job_description), profile),
            "matched_keywords": filter_domain_terms(find_keyword_overlap(resume_text, job_description)[:15], profile),
            "missing_keywords": filter_domain_terms(find_missing_keywords(resume_text, job_description)[:15], profile),
            "suggested_roles": [role] if role != "target role" else [],
        }
    )


def calculate_local_match_score(resume_text: str, job_description: str) -> int:
    if not job_description.strip():
        score = 45
        if extract_email(resume_text):
            score += 10
        if extract_phone(resume_text):
            score += 10
        score += min(len(extract_skills(resume_text)) * 3, 25)
        return min(score, 90)

    job_keywords = set(extract_keywords(job_description, limit=60))
    if not job_keywords:
        return 50

    resume_words = set(extract_keywords(resume_text, limit=250))
    overlap = job_keywords & resume_words
    return max(5, min(95, round((len(overlap) / len(job_keywords)) * 100)))


def build_local_gaps(resume_text: str, job_description: str, profile: dict[str, Any]) -> list[str]:
    missing = find_missing_keywords(resume_text, job_description)
    if missing:
        return [f"Job keyword not clearly visible: {keyword}" for keyword in missing[:5]]
    role = str(profile.get("role") or "the target role")
    return [f"Local analysis did not find deeper {role} seniority gaps without an AI provider."]


def build_local_recommendations(profile: dict[str, Any], job_description: str) -> list[str]:
    role = str(profile.get("role") or "target role")
    industry = str(profile.get("industry") or "")
    certifications = [str(item) for item in profile.get("certifications", [])[:3]]

    recommendations = [
        f"Add measurable outcomes for {role} responsibilities only where truthful and provable.",
        "Mirror important job-description keywords naturally in the resume." if job_description.strip() else f"Add a concise summary tailored to {role} roles.",
    ]
    if industry == "Healthcare":
        recommendations.append("Highlight specialization, licenses, hospital exposure, patient volume, procedures, and clinical outcomes.")
    elif industry == "Education":
        recommendations.append("Show subject, grade level, student impact, assessment outcomes, lesson planning, and classroom initiatives.")
    elif industry == "Engineering":
        recommendations.append("Add tools, standards, design/manufacturing exposure, quality metrics, and project outcomes.")
    elif industry == "Human Resources":
        recommendations.append("Quantify hiring volume, time-to-fill, onboarding completion, offer acceptance, and employee engagement metrics.")
    elif industry == "Software & IT":
        recommendations.append("Group technical skills by language, framework, database, cloud, testing, and tooling.")
    else:
        recommendations.append(f"Group skills and achievements around the expectations of {role} roles.")

    if certifications:
        recommendations.append("Add relevant certifications or training such as " + ", ".join(certifications) + ".")
    return recommendations


def find_keyword_overlap(resume_text: str, job_description: str) -> list[str]:
    if not job_description.strip():
        return []
    resume_keywords = set(extract_keywords(resume_text, limit=250))
    job_keywords = extract_keywords(job_description, limit=60)
    return [keyword for keyword in job_keywords if keyword in resume_keywords]


def find_missing_keywords(resume_text: str, job_description: str) -> list[str]:
    if not job_description.strip():
        return []
    resume_keywords = set(extract_keywords(resume_text, limit=250))
    return [keyword for keyword in extract_keywords(job_description, limit=60) if keyword not in resume_keywords]


def normalize_score(value: Any, fallback: int) -> int:
    try:
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            value = match.group(0) if match else value
        score = int(float(value))
    except (TypeError, ValueError):
        score = int(fallback)
    return max(0, min(100, score))


def normalize_country_code(value: Any) -> str:
    country = str(value or "").strip().lower()
    aliases = {
        "india": "in",
        "ind": "in",
        "bharat": "in",
        "united states": "us",
        "usa": "us",
        "u.s.": "us",
        "u.s.a.": "us",
        "united kingdom": "gb",
        "uk": "gb",
    }
    normalized = aliases.get(country, country or "in")
    return re.sub(r"[^a-z]", "", normalized)[:2] or "in"


def get_or_create_session_id() -> str:
    if not has_request_context():
        return "local"
    session_id = str(session.get(SESSION_ID_KEY) or "").strip()
    if not re.fullmatch(r"[0-9a-f]{32}", session_id):
        session_id = uuid.uuid4().hex
        session[SESSION_ID_KEY] = session_id
    return session_id


def remember_current_analysis(analysis_id: int) -> None:
    if has_request_context():
        session[CURRENT_ANALYSIS_ID_KEY] = int(analysis_id)


def get_current_session_analysis() -> sqlite3.Row | None:
    analysis_id = parse_optional_int(session.get(CURRENT_ANALYSIS_ID_KEY)) if has_request_context() else None
    return get_analysis(analysis_id) if analysis_id else None


def get_session_upload_dir(session_id: str | None = None) -> Path:
    safe_session_id = session_id if session_id and re.fullmatch(r"[0-9a-f]{32}", session_id) else get_or_create_session_id()
    upload_dir = Path(current_app.config["UPLOAD_FOLDER"]) / f"session_{safe_session_id}"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def resolve_owned_upload_path(analysis: sqlite3.Row | dict[str, Any]) -> Path | None:
    data = dict(analysis)
    stored_filename = str(data.get("stored_filename") or "")
    if not stored_filename:
        return None

    upload_root = Path(current_app.config["UPLOAD_FOLDER"]).resolve()
    candidate = (upload_root / stored_filename).resolve()
    try:
        candidate.relative_to(upload_root)
    except ValueError:
        return None

    if candidate.exists():
        return candidate

    fallbacks: list[Path] = []
    user_id = parse_optional_int(data.get("user_id"))
    if user_id is not None:
        fallbacks.append((upload_root / f"user_{user_id}" / Path(stored_filename).name).resolve())
    session_id = str(data.get("session_id") or "").strip()
    if session_id:
        fallbacks.append((upload_root / f"session_{session_id}" / Path(stored_filename).name).resolve())
    for fallback in fallbacks:
        try:
            fallback.relative_to(upload_root)
        except ValueError:
            continue
        if fallback.exists():
            return fallback
    return None


def get_analysis(analysis_id: int, user_id: int | None = None) -> sqlite3.Row | None:
    if analysis_id is None:
        return None
    return get_db().execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()


def get_chat_history(analysis_id: int, limit: int = 20) -> list[dict[str, str]]:
    if get_analysis(analysis_id) is None:
        return []
    rows = get_db().execute(
        """
        SELECT chat_messages.role, chat_messages.content
        FROM chat_messages
        WHERE chat_messages.analysis_id = ?
        ORDER BY chat_messages.id DESC
        LIMIT ?
        """,
        (analysis_id, limit),
    ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def save_chat_message(analysis_id: int, role: str, content: str) -> None:
    if role not in {"user", "assistant"} or not content.strip():
        return
    if get_analysis(analysis_id) is None:
        return
    db = get_db()
    db.execute(
        """
        INSERT INTO chat_messages (analysis_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (analysis_id, role, content.strip(), datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    db.commit()


def get_latest_interview_question_bundle(analysis_id: int) -> dict[str, Any]:
    if get_analysis(analysis_id) is None:
        return empty_interview_question_bundle()
    row = get_db().execute(
        """
        SELECT *
        FROM interview_questions
        WHERE analysis_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (analysis_id,),
    ).fetchone()
    return serialize_interview_question_bundle(row)


def save_interview_question_bundle(
    analysis_id: int,
    target_role: str,
    job_description: str,
    questions: list[dict[str, Any]],
    ai_provider: str,
) -> dict[str, Any]:
    if get_analysis(analysis_id) is None:
        return empty_interview_question_bundle()
    session_id = get_or_create_session_id()
    normalized_questions = normalize_interview_question_items(questions)
    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO interview_questions (
            analysis_id,
            user_id,
            session_id,
            target_role,
            job_description,
            questions_json,
            ai_provider,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            analysis_id,
            None,
            session_id,
            target_role.strip() or None,
            job_description.strip(),
            json.dumps(normalized_questions),
            ai_provider if ai_provider in {"gemini", "local"} else "local",
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    db.commit()
    row = db.execute("SELECT * FROM interview_questions WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return serialize_interview_question_bundle(row)


def serialize_interview_question_bundle(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        return empty_interview_question_bundle()
    data = dict(row)
    questions = normalize_interview_question_items(parse_json_field(data.get("questions_json"), []))
    questions = hydrate_missing_interview_answer_fields(questions, parse_optional_int(data.get("analysis_id")))
    generated_at = data.get("created_at")
    return {
        "id": data.get("id"),
        "analysis_id": data.get("analysis_id"),
        "target_role": data.get("target_role") or "",
        "job_description_used": bool(str(data.get("job_description") or "").strip()),
        "questions": questions,
        "sections": group_interview_questions(questions),
        "categories": sorted({question["category"] for question in questions}),
        "difficulties": [difficulty for difficulty in ("Easy", "Medium", "Hard") if any(question["difficulty"] == difficulty for question in questions)],
        "ai_provider": data.get("ai_provider") or "local",
        "created_at": generated_at,
        "created_at_display": format_timestamp(generated_at),
    }


def empty_interview_question_bundle() -> dict[str, Any]:
    return {
        "id": None,
        "analysis_id": None,
        "target_role": "",
        "job_description_used": False,
        "questions": [],
        "sections": [],
        "categories": [],
        "difficulties": [],
        "ai_provider": "",
        "created_at": "",
        "created_at_display": "",
    }


def serialize_analysis(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    data["extracted_skills"] = parse_json_field(data.get("extracted_skills"), [])
    data["claude_result"] = normalize_ai_fallback_status(parse_json_field(data.get("claude_result"), {}))
    data["created_at_display"] = format_timestamp(data.get("created_at"))
    data["created_at_relative"] = relative_timestamp(data.get("created_at"))
    data["contact_details"] = extract_contact_details(str(data.get("resume_text") or ""))
    return data


def parse_json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def parse_skill_payload(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


app = create_app()


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"])

