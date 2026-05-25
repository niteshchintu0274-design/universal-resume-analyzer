from __future__ import annotations

import re
from functools import lru_cache

import spacy
from spacy.matcher import PhraseMatcher


SKILLS: tuple[str, ...] = (
    "python",
    "java",
    "javascript",
    "react",
    "flask",
    "django",
    "sql",
    "mongodb",
    "machine learning",
    "deep learning",
    "nlp",
    "pandas",
    "numpy",
    "git",
    "docker",
    "aws",
    "html",
    "css",
    "tailwind",
)


def extract_skills(text: str, skills: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """
    Extract known technical skills from text using spaCy PhraseMatcher.

    The result uses canonical lowercase skill names from SKILLS.
    """
    if not text:
        return []

    skill_list = tuple(_normalize_skill(skill) for skill in (skills or SKILLS))
    skill_list = tuple(dict.fromkeys(skill for skill in skill_list if skill))

    nlp = get_nlp()
    matcher = build_phrase_matcher(skill_list)
    doc = nlp(text)

    found: set[str] = set()

    for match_id, start, end in matcher(doc):
        matched_skill = nlp.vocab.strings[match_id].lower()
        found.add(matched_skill)

    # Regex backup catches forms like "machine-learning" and "tailwind.css".
    normalized_text = text.lower()
    for skill in skill_list:
        if _skill_pattern(skill).search(normalized_text):
            found.add(skill)

    return sorted(found, key=lambda skill: skill_list.index(skill))


def match_skills(text: str, skills: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """Alias for extract_skills."""
    return extract_skills(text, skills=skills)


def skill_match_summary(
    resume_text: str,
    job_description: str = "",
    required_skills: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """
    Compare resume skills against required skills.

    If required_skills is not provided, skills are extracted from the job description.
    If neither required_skills nor job_description is provided, the full SKILLS list is used.
    """
    resume_skills = extract_skills(resume_text)

    if required_skills is not None:
        target_skills = normalize_skill_list(required_skills)
    elif job_description.strip():
        target_skills = extract_skills(job_description)
    else:
        target_skills = list(SKILLS)

    matched_skills = [skill for skill in target_skills if skill in resume_skills]
    missing_skills = [skill for skill in target_skills if skill not in resume_skills]
    match_score = calculate_skill_score(matched_skills, target_skills)

    return {
        "resume_skills": resume_skills,
        "target_skills": target_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "match_score": match_score,
    }


def calculate_skill_score(
    matched_skills: list[str] | tuple[str, ...],
    target_skills: list[str] | tuple[str, ...],
) -> int:
    """Return a 0-100 percentage score for matched target skills."""
    unique_targets = normalize_skill_list(target_skills)
    if not unique_targets:
        return 0

    unique_matches = set(normalize_skill_list(matched_skills))
    score = (len(unique_matches.intersection(unique_targets)) / len(unique_targets)) * 100
    return round(score)


def find_missing_skills(
    resume_text: str,
    job_description: str = "",
    required_skills: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Return target skills that are not present in the resume."""
    return skill_match_summary(
        resume_text=resume_text,
        job_description=job_description,
        required_skills=required_skills,
    )["missing_skills"]


def normalize_skill_list(skills: list[str] | tuple[str, ...]) -> list[str]:
    """Normalize, deduplicate, and keep only skills from the known skill set."""
    allowed = set(SKILLS)
    normalized: list[str] = []

    for skill in skills:
        value = _normalize_skill(skill)
        if value in allowed and value not in normalized:
            normalized.append(value)

    return normalized


@lru_cache(maxsize=1)
def get_nlp(model_name: str = "en_core_web_sm"):
    """Load spaCy model with a lightweight blank fallback."""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    nlp.max_length = max(nlp.max_length, 2_000_000)
    return nlp


@lru_cache(maxsize=16)
def build_phrase_matcher(skills: tuple[str, ...] = SKILLS) -> PhraseMatcher:
    """Build a cached spaCy PhraseMatcher for the provided skills."""
    nlp = get_nlp()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    for skill in skills:
        matcher.add(skill, [nlp.make_doc(skill)])

    return matcher


@lru_cache(maxsize=64)
def _skill_pattern(skill: str) -> re.Pattern[str]:
    escaped_parts = [re.escape(part) for part in skill.split()]
    body = r"[\s_.-]+".join(escaped_parts)
    return re.compile(rf"(?<![a-z0-9+#]){body}(?![a-z0-9+#])", re.IGNORECASE)


def _normalize_skill(skill: str) -> str:
    return re.sub(r"\s+", " ", skill.strip().lower())
