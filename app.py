import json
import os
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
    jsonify,
    redirect,
    render_template,
    render_template_string,
    request,
    send_from_directory,
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

INDEX_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AI Resume Analyzer</title>
    <script src="{{ config.TAILWIND_CDN_URL }}"></script>
  </head>
  <body class="min-h-screen bg-slate-50 text-slate-950">
    <main class="mx-auto grid max-w-6xl gap-6 px-4 py-8 lg:grid-cols-[1fr_360px]">
      <section class="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <div class="mb-6">
          <h1 class="text-2xl font-semibold tracking-tight">AI Resume Analyzer</h1>
          <p class="mt-2 text-sm text-slate-600">Upload a PDF resume and compare it against a job description.</p>
        </div>

        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            <div class="mb-5 space-y-2">
              {% for category, message in messages %}
                <div class="rounded-md border px-3 py-2 text-sm {{ 'border-red-200 bg-red-50 text-red-800' if category == 'error' else 'border-emerald-200 bg-emerald-50 text-emerald-800' }}">
                  {{ message }}
                </div>
              {% endfor %}
            </div>
          {% endif %}
        {% endwith %}

        <form action="{{ url_for('analyze_resume') }}" method="post" enctype="multipart/form-data" class="space-y-5">
          <div>
            <label for="resume" class="block text-sm font-medium text-slate-700">Resume PDF</label>
            <input id="resume" name="resume" type="file" accept="application/pdf,.pdf" required class="mt-2 block w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm file:mr-4 file:rounded-md file:border-0 file:bg-slate-900 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white">
          </div>

          <div>
            <label for="job_description" class="block text-sm font-medium text-slate-700">Job Description</label>
            <textarea id="job_description" name="job_description" rows="10" placeholder="Paste the job description here..." class="mt-2 block w-full rounded-md border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-900 focus:ring-1 focus:ring-slate-900"></textarea>
          </div>

          <button type="submit" class="inline-flex items-center rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800">Analyze Resume</button>
        </form>
      </section>

      <aside class="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <h2 class="text-base font-semibold">Recent Analyses</h2>
        {% if analyses %}
          <div class="mt-4 space-y-3">
            {% for item in analyses %}
              <a href="{{ url_for('analysis_detail', analysis_id=item.id) }}" class="block rounded-md border border-slate-200 p-3 hover:bg-slate-50">
                <div class="flex items-center justify-between gap-3">
                  <p class="truncate text-sm font-medium">{{ item.original_filename }}</p>
                  <span class="rounded bg-slate-100 px-2 py-1 text-xs font-medium">{{ item.score or 0 }}%</span>
                </div>
                <p class="mt-1 text-xs text-slate-500">{{ item.created_at }}</p>
              </a>
            {% endfor %}
          </div>
        {% else %}
          <p class="mt-4 text-sm text-slate-500">No analyses yet.</p>
        {% endif %}
      </aside>
    </main>
  </body>
</html>
"""

DETAIL_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Resume Analysis</title>
    <script src="{{ config.TAILWIND_CDN_URL }}"></script>
  </head>
  <body class="min-h-screen bg-slate-50 text-slate-950">
    <main class="mx-auto max-w-5xl px-4 py-8">
      <a href="{{ url_for('index') }}" class="text-sm font-medium text-slate-600 hover:text-slate-950">Back to upload</a>
      <section class="mt-4 rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <div class="flex flex-col gap-4 border-b border-slate-200 pb-5 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h1 class="text-2xl font-semibold tracking-tight">{{ analysis.original_filename }}</h1>
            <p class="mt-1 text-sm text-slate-500">{{ analysis.created_at }}</p>
            <div class="mt-3 flex flex-wrap gap-2 text-xs">
              {% if analysis.candidate_name %}<span class="rounded bg-slate-100 px-2 py-1">{{ analysis.candidate_name }}</span>{% endif %}
              {% if analysis.email %}<span class="rounded bg-slate-100 px-2 py-1">{{ analysis.email }}</span>{% endif %}
              {% if analysis.phone %}<span class="rounded bg-slate-100 px-2 py-1">{{ analysis.phone }}</span>{% endif %}
            </div>
          </div>
          <div class="rounded-lg border border-slate-200 px-5 py-4 text-center">
            <div class="text-3xl font-semibold">{{ analysis.score or 0 }}%</div>
            <div class="text-xs uppercase tracking-wide text-slate-500">Match Score</div>
          </div>
        </div>

        {% if result.error %}
          <div class="mt-5 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">{{ result.error }}</div>
        {% endif %}

        <div class="mt-6 grid gap-5 lg:grid-cols-2">
          <div>
            <h2 class="text-base font-semibold">Summary</h2>
            <p class="mt-2 text-sm leading-6 text-slate-700">{{ result.summary or "No summary returned." }}</p>
          </div>
          <div>
            <h2 class="text-base font-semibold">Skills Found</h2>
            <div class="mt-2 flex flex-wrap gap-2">
              {% for skill in skills %}
                <span class="rounded bg-slate-100 px-2 py-1 text-xs font-medium">{{ skill }}</span>
              {% else %}
                <span class="text-sm text-slate-500">No skills detected.</span>
              {% endfor %}
            </div>
          </div>
          <div>
            <h2 class="text-base font-semibold">Strengths</h2>
            <ul class="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
              {% for item in result.strengths or [] %}<li>{{ item }}</li>{% else %}<li>No strengths returned.</li>{% endfor %}
            </ul>
          </div>
          <div>
            <h2 class="text-base font-semibold">Gaps</h2>
            <ul class="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
              {% for item in result.gaps or [] %}<li>{{ item }}</li>{% else %}<li>No gaps returned.</li>{% endfor %}
            </ul>
          </div>
          <div class="lg:col-span-2">
            <h2 class="text-base font-semibold">Recommendations</h2>
            <ul class="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
              {% for item in result.recommendations or [] %}<li>{{ item }}</li>{% else %}<li>No recommendations returned.</li>{% endfor %}
            </ul>
          </div>
        </div>
      </section>
    </main>
  </body>
</html>
"""


def create_app(config_class: type[Config] = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)

    ensure_directories(app)

    with app.app_context():
        init_db()

    app.add_url_rule("/", "index", index, methods=["GET"])
    app.add_url_rule("/chatbot", "chatbot", chatbot, methods=["GET"])
    app.add_url_rule("/compare", "compare", compare, methods=["GET", "POST"])
    app.add_url_rule("/analyze", "analyze_resume", analyze_resume, methods=["POST"])
    app.add_url_rule("/analysis/<int:analysis_id>", "analysis_detail", analysis_detail, methods=["GET"])
    app.add_url_rule("/analysis/<int:analysis_id>/download", "download_resume", download_resume, methods=["GET"])
    app.add_url_rule("/api/analyze", "api_analyze_resume", api_analyze_resume, methods=["POST"])
    app.add_url_rule("/api/career-chat", "api_career_chat", api_career_chat, methods=["POST"])
    app.add_url_rule("/api/rewrite-section", "api_rewrite_section", api_rewrite_section, methods=["POST"])
    app.add_url_rule("/api/job-match", "api_job_match", api_job_match, methods=["POST"])
    app.add_url_rule("/api/jobs/<int:analysis_id>", "api_matching_jobs", api_matching_jobs, methods=["GET"])
    app.add_url_rule("/api/compare", "api_compare", api_compare, methods=["POST"])
    app.add_url_rule("/api/analyses", "api_analyses", api_analyses, methods=["GET"])
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
    db.execute(
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
    db.commit()


def index() -> str:
    analyses = [prepare_analysis_for_view(row) for row in get_recent_analyses()]
    return render_template("index.html", analyses=analyses)


def chatbot() -> str:
    analyses = [prepare_analysis_for_view(row) for row in get_recent_analyses(limit=20)]
    latest = analyses[0] if analyses else None
    latest_result = normalize_ai_fallback_status(parse_json_field(latest["claude_result"], {})) if latest else {}
    return render_template("chatbot.html", analyses=analyses, latest=latest, latest_result=latest_result)


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
            reply = "Using the local resume analysis for this answer.\n\n" + local_reply
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
        country=request.args.get("country", "us").strip() or "us",
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
            filters=JobFilters(country="us"),
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
    )


def download_resume(analysis_id: int):
    analysis = get_analysis(analysis_id)
    if analysis is None:
        abort(404)
    return send_from_directory(
        current_app.config["UPLOAD_FOLDER"],
        analysis["stored_filename"],
        as_attachment=True,
        download_name=analysis["original_filename"],
    )


def api_analyses():
    return jsonify({"ok": True, "analyses": [serialize_analysis(row) for row in get_recent_analyses(limit=50)]})


def api_analysis_detail(analysis_id: int):
    analysis = get_analysis(analysis_id)
    if analysis is None:
        return jsonify({"ok": False, "error": "Analysis not found"}), 404
    return jsonify({"ok": True, "analysis": serialize_analysis(analysis)})


def handle_resume_submission(file_storage, job_description: str) -> int:
    if not file_storage or not file_storage.filename:
        raise ValueError("Please upload a resume PDF.")

    if not allowed_file(file_storage.filename):
        raise ValueError("Only PDF files are supported.")

    original_filename = secure_filename(file_storage.filename)
    stored_filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex}.pdf"
    saved_path = Path(current_app.config["UPLOAD_FOLDER"]) / stored_filename
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
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            original_filename,
            stored_filename,
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
    return int(cursor.lastrowid)


def handle_compare_submission(file_v1, file_v2, job_description: str = "") -> dict[str, Any]:
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
            saved_path = Path(current_app.config["UPLOAD_FOLDER"]) / stored_filename
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
        doctor_result["ai_status"] = "AI unavailable. Using local analysis."

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
        result["ai_status"] = "AI unavailable. Using local analysis."
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
    }
    return messages.get(context, "Something went wrong. Please try again.")


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
    result["ai_status"] = "AI unavailable. Using local analysis."
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


def get_recent_analyses(limit: int = 10) -> list[sqlite3.Row]:
    return get_db().execute(
        """
        SELECT id, original_filename, stored_filename, candidate_name, email, phone, score,
               extracted_skills, resume_text, job_description, claude_result, created_at
        FROM analyses
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def get_analysis(analysis_id: int) -> sqlite3.Row | None:
    return get_db().execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()


def get_chat_history(analysis_id: int, limit: int = 20) -> list[dict[str, str]]:
    rows = get_db().execute(
        """
        SELECT role, content
        FROM chat_messages
        WHERE analysis_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (analysis_id, limit),
    ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def save_chat_message(analysis_id: int, role: str, content: str) -> None:
    if role not in {"user", "assistant"} or not content.strip():
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
