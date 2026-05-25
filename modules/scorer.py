from __future__ import annotations

import re
from typing import Any

try:
    from .extractor import count_words
    from .skill_matcher import SKILLS, extract_skills, skill_match_summary
except ImportError:
    from extractor import count_words
    from skill_matcher import SKILLS, extract_skills, skill_match_summary


TOTAL_POINTS = 100
SKILLS_POINTS = 40
LENGTH_POINTS = 15
EDUCATION_POINTS = 15
EXPERIENCE_POINTS = 15
ATS_KEYWORDS_POINTS = 15

EDUCATION_TERMS = (
    "bachelor",
    "master",
    "phd",
    "doctorate",
    "b.tech",
    "m.tech",
    "b.e",
    "m.e",
    "bsc",
    "msc",
    "bca",
    "mca",
    "mba",
    "degree",
    "diploma",
)

INSTITUTION_TERMS = (
    "university",
    "college",
    "institute",
    "school",
    "academy",
)

EXPERIENCE_SECTION_TERMS = (
    "experience",
    "work experience",
    "employment",
    "professional experience",
    "internship",
    "internships",
    "academic projects",
    "major projects",
    "freelance",
    "hackathon",
    "practical training",
)

ACTION_VERBS = (
    "built",
    "created",
    "developed",
    "designed",
    "implemented",
    "improved",
    "optimized",
    "deployed",
    "managed",
    "led",
    "collaborated",
    "automated",
    "integrated",
    "analyzed",
)

DEFAULT_ATS_SECTION_KEYWORDS = (
    "skills",
    "experience",
    "education",
    "projects",
    "certifications",
    "summary",
    "objective",
    "achievements",
)

KEYWORD_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "and",
    "any",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "can",
    "could",
    "during",
    "each",
    "for",
    "from",
    "has",
    "have",
    "into",
    "more",
    "most",
    "our",
    "other",
    "over",
    "same",
    "such",
    "than",
    "that",
    "the",
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
    "will",
    "would",
    "you",
    "your",
}


def score_resume(
    resume_text: str,
    job_description: str = "",
    required_skills: list[str] | tuple[str, ...] | None = None,
    ats_keywords: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """
    Score a resume out of 100.

    Breakdown:
    - Skills: 40 points
    - Length: 15 points
    - Education: 15 points
    - Experience: 15 points
    - ATS keywords: 15 points
    """
    text = resume_text or ""

    breakdown = {
        "skills": score_skills(text, job_description=job_description, required_skills=required_skills),
        "length": score_length(text),
        "education": score_education(text),
        "experience": score_experience(text),
        "ats_keywords": score_ats_keywords(text, job_description=job_description, ats_keywords=ats_keywords),
    }

    total_score = sum(section["points"] for section in breakdown.values())

    return {
        "total_score": clamp_int(total_score, 0, TOTAL_POINTS),
        "max_score": TOTAL_POINTS,
        "grade": grade_score(total_score),
        "breakdown": breakdown,
        "recommendations": build_recommendations(breakdown),
    }


def calculate_resume_score(
    resume_text: str,
    job_description: str = "",
    required_skills: list[str] | tuple[str, ...] | None = None,
    ats_keywords: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Alias for score_resume."""
    return score_resume(
        resume_text=resume_text,
        job_description=job_description,
        required_skills=required_skills,
        ats_keywords=ats_keywords,
    )


def score_skills(
    resume_text: str,
    job_description: str = "",
    required_skills: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Score skill coverage out of 40 points."""
    summary = skill_match_summary(
        resume_text=resume_text,
        job_description=job_description,
        required_skills=required_skills,
    )
    match_score = int(summary["match_score"])
    points = round((match_score / 100) * SKILLS_POINTS)

    return {
        "label": "Skills",
        "points": clamp_int(points, 0, SKILLS_POINTS),
        "max_points": SKILLS_POINTS,
        "percentage": match_score,
        "matched_skills": summary["matched_skills"],
        "missing_skills": summary["missing_skills"],
        "resume_skills": summary["resume_skills"],
        "target_skills": summary["target_skills"],
    }


def score_length(resume_text: str) -> dict[str, Any]:
    """
    Score resume length out of 15 points.

    A concise one-page technical resume is usually around 350-900 words.
    """
    words = count_words(resume_text)

    if 350 <= words <= 900:
        points = 15
        status = "ideal"
    elif 250 <= words < 350 or 901 <= words <= 1100:
        points = 12
        status = "acceptable"
    elif 150 <= words < 250 or 1101 <= words <= 1400:
        points = 8
        status = "needs adjustment"
    elif 80 <= words < 150 or 1401 <= words <= 1800:
        points = 4
        status = "weak"
    else:
        points = 0
        status = "too short" if words < 80 else "too long"

    return {
        "label": "Length",
        "points": points,
        "max_points": LENGTH_POINTS,
        "word_count": words,
        "status": status,
        "ideal_range": "350-900 words",
    }


def score_education(resume_text: str) -> dict[str, Any]:
    """Score education evidence out of 15 points."""
    text = normalize_text(resume_text)
    detected_degrees = find_terms(text, EDUCATION_TERMS)
    detected_institutions = find_terms(text, INSTITUTION_TERMS)
    years = find_years(text)
    grade_mentions = re.findall(r"\b(?:cgpa|gpa|percentage|grade)\b", text, flags=re.IGNORECASE)

    points = 0
    if detected_degrees:
        points += 8
    if detected_institutions:
        points += 4
    if years or grade_mentions:
        points += 3

    return {
        "label": "Education",
        "points": clamp_int(points, 0, EDUCATION_POINTS),
        "max_points": EDUCATION_POINTS,
        "detected_degrees": detected_degrees,
        "detected_institutions": detected_institutions,
        "detected_years": years,
        "has_grade_or_cgpa": bool(grade_mentions),
    }


def score_experience(resume_text: str) -> dict[str, Any]:
    """Score professional experience evidence out of 15 points."""
    text = normalize_text(resume_text)
    section_terms = find_terms(text, EXPERIENCE_SECTION_TERMS)
    alternative_terms = find_terms(
        text,
        (
            "internship",
            "academic project",
            "major project",
            "freelance",
            "hackathon",
            "practical training",
            "industrial training",
        ),
    )
    years_of_experience = extract_years_of_experience(text)
    date_ranges = extract_date_ranges(text)
    action_verbs = find_terms(text, ACTION_VERBS)
    quantified_results = re.findall(r"\b\d+(?:\.\d+)?[ \t]*(?:%|percent|x|k|m|hours|days|users|customers|projects)\b", text)

    points = 0
    if section_terms:
        points += 4
    elif alternative_terms:
        points += 4
    if years_of_experience is not None or date_ranges:
        points += 5
    elif alternative_terms:
        points += 3
    if action_verbs:
        points += 3
    if quantified_results:
        points += 3

    return {
        "label": "Experience",
        "points": clamp_int(points, 0, EXPERIENCE_POINTS),
        "max_points": EXPERIENCE_POINTS,
        "section_terms": section_terms,
        "experience_alternatives": alternative_terms,
        "years_of_experience": years_of_experience,
        "date_ranges": date_ranges[:5],
        "action_verbs": action_verbs[:10],
        "quantified_results_count": len(quantified_results),
    }


def score_ats_keywords(
    resume_text: str,
    job_description: str = "",
    ats_keywords: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Score ATS keyword coverage out of 15 points."""
    resume_lower = normalize_text(resume_text)

    if ats_keywords:
        target_keywords = normalize_keyword_list(ats_keywords)
    elif job_description.strip():
        target_keywords = extract_ats_keywords(job_description)
    else:
        target_keywords = list(DEFAULT_ATS_SECTION_KEYWORDS) + list(SKILLS)

    matched_keywords = [keyword for keyword in target_keywords if keyword_in_text(keyword, resume_lower)]
    missing_keywords = [keyword for keyword in target_keywords if keyword not in matched_keywords]

    if not target_keywords:
        points = 0
        percentage = 0
    else:
        percentage = round((len(matched_keywords) / len(target_keywords)) * 100)
        points = round((percentage / 100) * ATS_KEYWORDS_POINTS)

    return {
        "label": "ATS Keywords",
        "points": clamp_int(points, 0, ATS_KEYWORDS_POINTS),
        "max_points": ATS_KEYWORDS_POINTS,
        "percentage": percentage,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "target_keywords": target_keywords,
    }


def extract_ats_keywords(text: str, limit: int = 25) -> list[str]:
    """
    Extract ATS-style keywords from a job description.

    This combines known technical skills with frequent meaningful words.
    """
    skills = extract_skills(text)
    words = re.findall(r"\b[a-zA-Z][a-zA-Z+#.-]{2,}\b", text.lower())

    counts: dict[str, int] = {}
    for word in words:
        word = word.strip(".-").lower()
        if not word or word in KEYWORD_STOPWORDS:
            continue
        if word in set(" ".join(skills).split()):
            continue
        counts[word] = counts.get(word, 0) + 1

    ranked_words = sorted(counts, key=lambda word: (-counts[word], word))
    return normalize_keyword_list([*skills, *ranked_words[:limit]])


def build_recommendations(breakdown: dict[str, dict[str, Any]]) -> list[str]:
    """Generate concise recommendations from the score breakdown."""
    recommendations: list[str] = []

    skills = breakdown["skills"]
    if skills["missing_skills"]:
        recommendations.append("Add or highlight missing target skills: " + ", ".join(skills["missing_skills"][:6]) + ".")

    length = breakdown["length"]
    if length["points"] < LENGTH_POINTS:
        if length["word_count"] < 350:
            recommendations.append("Expand the resume with stronger project, experience, and impact details.")
        elif length["word_count"] > 900:
            recommendations.append("Shorten the resume by removing repeated details and low-impact bullets.")

    education = breakdown["education"]
    if education["points"] < EDUCATION_POINTS:
        recommendations.append("Add degree, institution, graduation year, or CGPA/grade details in the education section.")

    experience = breakdown["experience"]
    if experience["points"] < EXPERIENCE_POINTS:
        recommendations.append("Use action verbs, dates, and measurable outcomes in experience or project bullets.")

    ats = breakdown["ats_keywords"]
    if ats["missing_keywords"]:
        recommendations.append("Include important ATS keywords naturally: " + ", ".join(ats["missing_keywords"][:6]) + ".")

    if not recommendations:
        recommendations.append("Resume score is strong. Keep tailoring keywords and measurable outcomes for each job.")

    return recommendations


def grade_score(score: int) -> str:
    """Convert score into a simple label."""
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Average"
    if score >= 40:
        return "Needs Improvement"
    return "Weak"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def normalize_keyword_list(keywords: list[str] | tuple[str, ...]) -> list[str]:
    normalized: list[str] = []
    for keyword in keywords:
        value = normalize_text(str(keyword))
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def keyword_in_text(keyword: str, text: str) -> bool:
    escaped_parts = [re.escape(part) for part in normalize_text(keyword).split()]
    pattern = r"[\s_.-]+".join(escaped_parts)
    return bool(re.search(rf"(?<![a-z0-9+#]){pattern}(?![a-z0-9+#])", text, flags=re.IGNORECASE))


def find_terms(text: str, terms: list[str] | tuple[str, ...]) -> list[str]:
    return [term for term in terms if keyword_in_text(term, text)]


def find_years(text: str) -> list[str]:
    return sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", text)))


def extract_years_of_experience(text: str) -> int | None:
    patterns = (
        r"(\d{1,2})\+?\s*(?:years|yrs)\s+(?:of\s+)?experience",
        r"experience\s*(?:of\s*)?(\d{1,2})\+?\s*(?:years|yrs)",
    )
    values: list[int] = []
    for pattern in patterns:
        values.extend(int(match) for match in re.findall(pattern, text, flags=re.IGNORECASE))
    return max(values) if values else None


def extract_date_ranges(text: str) -> list[str]:
    month = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
    year = r"(?:19|20)\d{2}"
    present = r"(?:present|current|now)"
    pattern = rf"\b(?:{month}\s+)?{year}\s*(?:-|to|\u2013)\s*(?:(?:{month}\s+)?{year}|{present})\b"
    return re.findall(pattern, text, flags=re.IGNORECASE)


def clamp_int(value: int | float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, round(value)))
