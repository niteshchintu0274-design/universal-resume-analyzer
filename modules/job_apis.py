from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv
load_dotenv()

try:
    from .recommender import recommend_jobs
    from .universal_analyzer import build_target_keywords, keyword_in_text, normalize_text
except ImportError:
    from recommender import recommend_jobs
    from universal_analyzer import build_target_keywords, keyword_in_text, normalize_text


DEFAULT_TIMEOUT = 8.0


@dataclass(frozen=True)
class JobFilters:
    location: str = ""
    remote: bool = False
    salary_min: int | None = None
    experience: str = ""
    freshers: bool = False
    country: str = "in"


def fetch_matching_jobs(
    resume_text: str,
    profile: dict[str, Any],
    skills: list[str],
    database_path: str | None = None,
    filters: JobFilters | None = None,
    limit: int = 12,
) -> dict[str, Any]:
    filters = filters or JobFilters()
    query = build_query(profile, skills, filters)
    jobs: list[dict[str, Any]] = []
    provider_notes: list[str] = []

    providers = (
        ("JSearch", fetch_jsearch_jobs),
        ("Adzuna", fetch_adzuna_jobs),
        ("RapidAPI Jobs", fetch_generic_rapidapi_jobs),
        ("LinkedIn Partner", fetch_linkedin_partner_jobs),
    )

    for provider_name, provider in providers:
        try:
            provider_jobs = provider(query=query, filters=filters, limit=limit)
        except MissingProviderConfig as exc:
            provider_notes.append(str(exc))
            continue
        except Exception as exc:
            provider_notes.append(f"{provider_name} request failed: {exc}")
            continue
        jobs.extend(provider_jobs)
        if len(jobs) >= limit:
            break

    source = "live-api"
    if not jobs:
        source = "demo"
        jobs = demo_jobs(profile, filters)
        provider_notes = ["Demo mode: live job APIs are not connected, so these sample roles are shown for the detected profile."]

    normalized = score_and_filter_jobs(jobs, resume_text, profile, skills, filters=filters, limit=limit)
    return {
        "jobs": normalized,
        "source": source,
        "query": query,
        "provider_notes": provider_notes,
        "filters": filters.__dict__,
    }


def demo_jobs(profile: dict[str, Any], filters: JobFilters) -> list[dict[str, Any]]:
    role = normalize_text(str(profile.get("role") or ""))
    if "frontend" in role or not role:
        jobs = [
            {
                "title": "Frontend Developer Intern",
                "company": "Demo SaaS Studio",
                "location": filters.location or "Remote",
                "work_mode": "Remote" if filters.remote else "Hybrid",
                "salary": "$800-$1,200/month",
                "experience": "Internship, fresher, 0-1 years",
                "apply_link": "",
                "description": "Build responsive UI screens, fix accessibility issues, and work with React, HTML, CSS, JavaScript, Git, and component-based workflows.",
                "source": "Demo",
            },
            {
                "title": "React Developer Fresher",
                "company": "Launchpad Products",
                "location": filters.location or "Bengaluru / Remote",
                "work_mode": "Remote" if filters.remote else "Hybrid",
                "salary": "$18k-$28k",
                "experience": "Fresher, entry level, 0-1 years",
                "apply_link": "",
                "description": "Create reusable React components, consume REST APIs, polish CSS layouts, and document project decisions with clean GitHub examples.",
                "source": "Demo",
            },
            {
                "title": "Web Developer Trainee",
                "company": "Northstar Digital",
                "location": filters.location or "Onsite / Hybrid",
                "work_mode": "Hybrid",
                "salary": "$15k-$22k",
                "experience": "Graduate trainee, no experience required",
                "apply_link": "",
                "description": "Train on HTML, CSS, JavaScript, responsive design, browser testing, and live project maintenance under senior developer review.",
                "source": "Demo",
            },
        ]
    else:
        title = str(profile.get("role") or "Career").strip()
        jobs = [
            {
                "title": f"{title} Intern",
                "company": "Demo Talent Network",
                "location": filters.location or "Remote",
                "work_mode": "Remote" if filters.remote else "Hybrid",
                "salary": "Market stipend",
                "experience": "Internship, fresher, 0-1 years",
                "apply_link": "",
                "description": f"Entry-level {title} role focused on supervised practical work, documentation, communication, and role-relevant learning outcomes.",
                "source": "Demo",
            },
            {
                "title": f"{title} Fresher",
                "company": "Career Launch Partners",
                "location": filters.location or "Flexible",
                "work_mode": "Hybrid",
                "salary": "Entry-level range",
                "experience": "Fresher, graduate, trainee",
                "apply_link": "",
                "description": f"Fresher-friendly {title} opening where academic projects, internships, practical training, and certifications can support shortlisting.",
                "source": "Demo",
            },
        ]
    return jobs


class MissingProviderConfig(RuntimeError):
    pass


def build_query(profile: dict[str, Any], skills: list[str], filters: JobFilters | None = None) -> str:
    role = str(profile.get("role") or "").strip()
    role = role.replace("/", " ").replace("|", " ")
    role = re.sub(r"\bprofessional\b", "", role, flags=re.IGNORECASE)
    role = re.sub(r"\s+", " ", role).strip()

    normalized_role = normalize_text(role)

    if "frontend" in normalized_role:
        return "frontend developer"
    if "web" in normalized_role and "developer" in normalized_role:
        return "web developer"
    if "nurse" in normalized_role or "healthcare" in normalized_role:
        return "nurse"
    if "software" in normalized_role:
        return "software developer"

    return role or " ".join(skills[:2]) or "fresher"


def build_filter_query_terms(filters: JobFilters) -> str:
    terms: list[str] = []
    if filters.remote:
        terms.append("remote")
    if filters.freshers:
        terms.append("fresher entry level graduate trainee")
    elif filters.experience:
        terms.append(filters.experience)
    return " ".join(terms)


def fetch_jsearch_jobs(query: str, filters: JobFilters, limit: int) -> list[dict[str, Any]]:
    api_key = os.getenv("JSEARCH_RAPIDAPI_KEY") or os.getenv("RAPIDAPI_KEY")
    if not api_key:
        raise MissingProviderConfig("JSearch API key missing. Set JSEARCH_RAPIDAPI_KEY or RAPIDAPI_KEY.")

    params = {
        "query": append_location(query, filters.location),
        "page": "1",
        "num_pages": "1",
        "country": filters.country or "us",
    }
    if filters.remote:
        params["remote_jobs_only"] = "true"

    response = httpx.get(
        "https://jsearch.p.rapidapi.com/search",
        headers={
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
        },
        params=params,
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    jobs = []
    for item in (data.get("data") or [])[:limit]:
        jobs.append(
            {
                "title": item.get("job_title") or "",
                "company": item.get("employer_name") or "",
                "location": item.get("job_city") or item.get("job_country") or item.get("job_location") or "",
                "work_mode": infer_work_mode(item.get("job_is_remote"), item.get("job_description", "")),
                "salary": format_salary(item.get("job_min_salary"), item.get("job_max_salary"), item.get("job_salary_currency")),
                "experience": item.get("job_required_experience", {}).get("required_experience_in_months") if isinstance(item.get("job_required_experience"), dict) else "",
                "apply_link": item.get("job_apply_link") or item.get("job_google_link") or "",
                "description": item.get("job_description") or "",
                "source": "JSearch",
            }
        )
    return jobs


def fetch_adzuna_jobs(query: str, filters: JobFilters, limit: int) -> list[dict[str, Any]]:
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise MissingProviderConfig("Adzuna credentials missing. Set ADZUNA_APP_ID and ADZUNA_APP_KEY.")

    country = (filters.country or os.getenv("ADZUNA_COUNTRY") or os.getenv("JOBS_COUNTRY") or "in").lower()
    params: dict[str, Any] = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": min(limit, 50),
        "what": query,
        "content-type": "application/json",
    }
    if filters.location:
        params["where"] = filters.location
    if filters.salary_min:
        params["salary_min"] = filters.salary_min

    response = httpx.get(
        f"https://api.adzuna.com/v1/api/jobs/{country}/search/1",
        params=params,
        headers={"Accept": "application/json"},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    jobs = []
    for item in (data.get("results") or [])[:limit]:
        jobs.append(
            {
                "title": item.get("title") or "",
                "company": (item.get("company") or {}).get("display_name", ""),
                "location": (item.get("location") or {}).get("display_name", ""),
                "work_mode": infer_work_mode(None, item.get("description", "") + " " + item.get("title", "")),
                "salary": format_salary(item.get("salary_min"), item.get("salary_max"), item.get("salary_currency") or ""),
                "experience": "",
                "apply_link": item.get("redirect_url") or "",
                "description": strip_html(item.get("description") or ""),
                "source": "Adzuna",
            }
        )
    return jobs


def fetch_generic_rapidapi_jobs(query: str, filters: JobFilters, limit: int) -> list[dict[str, Any]]:
    api_key = os.getenv("RAPIDAPI_JOBS_KEY")
    host = os.getenv("RAPIDAPI_JOBS_HOST")
    url = os.getenv("RAPIDAPI_JOBS_URL")
    if not api_key or not host or not url:
        raise MissingProviderConfig("Generic RapidAPI Jobs connector missing. Set RAPIDAPI_JOBS_KEY, RAPIDAPI_JOBS_HOST, and RAPIDAPI_JOBS_URL.")

    response = httpx.get(
        url,
        headers={"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": host},
        params={"query": append_location(query, filters.location), "limit": limit},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        items = payload.get("jobs", []) if isinstance(payload, dict) else []
    return [normalize_unknown_job(item, "RapidAPI Jobs") for item in items[:limit] if isinstance(item, dict)]


def fetch_linkedin_partner_jobs(query: str, filters: JobFilters, limit: int) -> list[dict[str, Any]]:
    url = os.getenv("LINKEDIN_JOBS_API_URL")
    token = os.getenv("LINKEDIN_JOBS_API_TOKEN")
    if not url or not token:
        raise MissingProviderConfig("LinkedIn job search requires an approved partner/proxy endpoint. Set LINKEDIN_JOBS_API_URL and LINKEDIN_JOBS_API_TOKEN.")

    response = httpx.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params={"keywords": query, "location": filters.location, "remote": str(filters.remote).lower(), "limit": limit},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("jobs") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        items = []
    return [normalize_unknown_job(item, "LinkedIn Partner") for item in items[:limit] if isinstance(item, dict)]


def local_jobs(resume_text: str, database_path: str, limit: int) -> list[dict[str, Any]]:
    jobs = recommend_jobs(resume_text, database_path=database_path, limit=limit)
    normalized = []
    for job in jobs:
        normalized.append(
            {
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", ""),
                "work_mode": infer_work_mode(None, job.get("location", "") + " " + job.get("description", "")),
                "salary": job.get("salary_range", ""),
                "experience": "",
                "apply_link": job.get("apply_url", ""),
                "description": job.get("description", ""),
                "source": "Local SQLite",
                "similarity": job.get("similarity", 0),
                "matched_skills": job.get("matched_skills", []),
                "missing_skills": job.get("missing_skills", []),
            }
        )
    return normalized


def score_and_filter_jobs(
    jobs: list[dict[str, Any]],
    resume_text: str,
    profile: dict[str, Any],
    resume_skills: list[str],
    filters: JobFilters,
    limit: int,
) -> list[dict[str, Any]]:
    seen = set()
    scored: list[dict[str, Any]] = []
    role_keywords = build_target_keywords_for_profile(profile)
    resume_skill_set = {normalize_text(skill) for skill in resume_skills}

    for job in jobs:
        if not passes_job_filters(job, filters):
            continue

        key = normalize_text(f"{job.get('title', '')} {job.get('company', '')} {job.get('location', '')}")
        if not key or key in seen:
            continue
        seen.add(key)

        job_text = normalize_text(" ".join(str(job.get(field, "")) for field in ("title", "description", "company", "location")))
        matched = [skill for skill in resume_skill_set if skill and keyword_in_text(skill, job_text)]
        target_missing = [keyword for keyword in role_keywords if keyword not in matched and keyword_in_text(keyword, job_text)]
        role_hits = [keyword for keyword in role_keywords if keyword_in_text(keyword, job_text)]
        base = float(job.get("similarity") or 0)
        if base <= 1:
            base *= 100
        skill_score = round((len(matched) / max(len(role_hits), 1)) * 100) if role_hits else min(75, len(matched) * 12)
        title_bonus = 12 if keyword_in_text(str(profile.get("role", "")), str(job.get("title", ""))) else 0
        match_percent = round(max(base, skill_score) * 0.82 + title_bonus)

        scored.append(
            {
                **job,
                "match_percent": max(0, min(98, match_percent)),
                "matched_skills": sorted(set(job.get("matched_skills") or matched))[:10],
                "missing_skills": sorted(set(job.get("missing_skills") or target_missing))[:10],
            }
        )

    scored.sort(key=lambda item: item["match_percent"], reverse=True)
    return scored[: max(0, limit)]


def passes_job_filters(job: dict[str, Any], filters: JobFilters) -> bool:
    text = normalize_text(" ".join(str(job.get(field, "")) for field in ("title", "description", "location", "work_mode", "experience", "salary")))
    if filters.remote and "remote" not in text:
        return False
    if filters.salary_min and not salary_meets_minimum(job.get("salary", ""), filters.salary_min):
        return False
    if filters.freshers and not matches_fresher_role(text):
        return False
    if filters.experience and not matches_experience_filter(text, filters.experience):
        return False
    return True


def salary_meets_minimum(salary: Any, minimum: int) -> bool:
    numbers = [float(value.replace(",", "")) for value in re.findall(r"\d[\d,]*(?:\.\d+)?", str(salary or ""))]
    if not numbers:
        return True
    return max(numbers) >= minimum


def matches_fresher_role(text: str) -> bool:
    if not text:
        return True
    return any(
        keyword in text
        for keyword in (
            "fresher",
            "entry level",
            "entry-level",
            "graduate",
            "trainee",
            "intern",
            "junior",
            "0-1",
            "0 to 1",
            "no experience",
        )
    )


def matches_experience_filter(text: str, experience: str) -> bool:
    normalized = normalize_text(experience)
    if not normalized:
        return True
    if normalized in text:
        return True
    if normalized in {"0-1", "0 to 1", "entry", "entry level"}:
        return matches_fresher_role(text)
    if normalized in {"1-3", "1 to 3"}:
        return bool(re.search(r"\b[123]\+?\s*(?:years|yrs)\b|\b1\s*-\s*3\b", text)) or not re.search(r"\b\d+\+?\s*(?:years|yrs)\b", text)
    if normalized in {"3-5", "3 to 5"}:
        return bool(re.search(r"\b[345]\+?\s*(?:years|yrs)\b|\b3\s*-\s*5\b", text)) or not re.search(r"\b\d+\+?\s*(?:years|yrs)\b", text)
    if normalized in {"5+", "5 plus", "senior"}:
        return bool(re.search(r"\b(?:5|6|7|8|9|10)\+?\s*(?:years|yrs)\b|senior|lead", text)) or not re.search(r"\b\d+\+?\s*(?:years|yrs)\b", text)
    return True


def build_target_keywords_for_profile(profile: dict[str, Any]) -> list[str]:
    skills = [str(skill) for skill in profile.get("target_skills", [])]
    role = str(profile.get("role") or "")
    synthetic_profile = type(
        "SyntheticProfile",
        (),
        {
            "skills": tuple(skills),
            "keywords": tuple([role, *skills]),
            "avoid_terms": tuple(),
        },
    )()
    return build_target_keywords(synthetic_profile)


def normalize_unknown_job(item: dict[str, Any], source: str) -> dict[str, Any]:
    title = first_present(item, ("title", "job_title", "position", "name"))
    company = first_present(item, ("company", "company_name", "employer_name", "organization"))
    location = first_present(item, ("location", "job_location", "city", "formatted_location"))
    description = first_present(item, ("description", "job_description", "snippet", "summary"))
    return {
        "title": title,
        "company": company,
        "location": location,
        "work_mode": infer_work_mode(item.get("remote"), f"{location} {description}"),
        "salary": first_present(item, ("salary", "salary_range", "compensation")),
        "experience": first_present(item, ("experience", "experience_level", "seniority")),
        "apply_link": first_present(item, ("apply_link", "url", "redirect_url", "job_apply_link")),
        "description": strip_html(description),
        "source": source,
    }


def first_present(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, dict):
            value = value.get("display_name") or value.get("name")
        if value not in (None, ""):
            return str(value)
    return ""


def append_location(query: str, location: str) -> str:
    return f"{query} in {location}".strip() if location else query


def infer_work_mode(remote_value: Any, text: str) -> str:
    if remote_value is True or str(remote_value).lower() == "true":
        return "Remote"
    normalized = normalize_text(text)
    if "remote" in normalized:
        return "Remote"
    if "hybrid" in normalized:
        return "Hybrid"
    if "onsite" in normalized or "on-site" in normalized:
        return "Onsite"
    return "Not specified"


def format_salary(min_salary: Any, max_salary: Any, currency: str = "") -> str:
    if not min_salary and not max_salary:
        return ""
    currency = str(currency or "").upper()
    low = format_number(min_salary)
    high = format_number(max_salary)
    if low and high:
        return f"{currency} {low}-{high}".strip()
    return f"{currency} {low or high}".strip()


def format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number >= 1000:
        return f"{number:,.0f}"
    return f"{number:g}"


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()
