from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import httpx
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_MAX_TOKENS = 1200
DEFAULT_TEMPERATURE = 0.3
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class ClaudeConfigurationError(RuntimeError):
    """Raised when Claude API configuration is missing or invalid."""


class ClaudeResponseError(RuntimeError):
    """Raised when Claude returns an unusable response."""


class GeminiConfigurationError(RuntimeError):
    """Raised when Gemini API configuration is missing or invalid."""


class GeminiResponseError(RuntimeError):
    """Raised when Gemini returns an unusable response."""


@dataclass(frozen=True)
class ClaudeSettings:
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    @classmethod
    def from_env(cls) -> "ClaudeSettings":
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY"),
            model=os.getenv("CLAUDE_MODEL", DEFAULT_MODEL),
            max_tokens=_as_int(os.getenv("CLAUDE_MAX_TOKENS"), DEFAULT_MAX_TOKENS),
            temperature=_as_float(os.getenv("CLAUDE_TEMPERATURE"), DEFAULT_TEMPERATURE),
        )


@dataclass(frozen=True)
class GeminiSettings:
    api_key: str | None = None
    model: str = DEFAULT_GEMINI_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    @classmethod
    def from_env(cls) -> "GeminiSettings":
        return cls(
            api_key=os.getenv("GEMINI_API_KEY"),
            model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            max_tokens=_as_int(os.getenv("GEMINI_MAX_TOKENS"), DEFAULT_MAX_TOKENS),
            temperature=_as_float(os.getenv("GEMINI_TEMPERATURE"), DEFAULT_TEMPERATURE),
        )


class ClaudeCareerHelper:
    """
    Claude-powered helper for career chat, interview prep, and resume tips.

    Set a Claude or Gemini key in .env before calling hosted AI features.
    """

    def __init__(
        self,
        settings: ClaudeSettings | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.settings = settings or ClaudeSettings.from_env()
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not self.settings.api_key:
                raise ClaudeConfigurationError(
                    "Claude API key is missing."
                )
            self._client = anthropic.Anthropic(api_key=self.settings.api_key)
        return self._client

    def career_chatbot(
        self,
        user_message: str,
        resume_text: str = "",
        chat_history: list[dict[str, str]] | None = None,
        target_role: str = "",
        job_description: str = "",
        profile_context: dict[str, Any] | None = None,
    ) -> str:
        """
        Career chatbot for resume, job search, interview, and skill guidance.

        Args:
            user_message: Current user question.
            resume_text: Optional resume context.
            chat_history: Optional list of {"role": "user"|"assistant", "content": "..."}.
            target_role: Optional target role.
            job_description: Optional job description context.

        Returns:
            Claude's career guidance as plain text.
        """
        if not user_message.strip():
            raise ValueError("user_message is required.")

        system_prompt = build_career_system_prompt()
        context = build_context_block(
            resume_text=resume_text,
            job_description=job_description,
            target_role=target_role,
            profile_context=profile_context,
        )
        messages = normalize_messages(chat_history)

        current_message = "\n\n".join(
            part
            for part in (
                context,
                f"User question:\n{user_message.strip()}",
            )
            if part
        )
        append_user_message(messages, current_message)

        return self.complete_text(system_prompt=system_prompt, messages=messages)

    def generate_interview_questions(
        self,
        role: str,
        experience_level: str = "entry-level",
        skills: list[str] | tuple[str, ...] | None = None,
        resume_text: str = "",
        job_description: str = "",
        question_count: int = 10,
    ) -> dict[str, Any]:
        """
        Generate role-specific interview questions as structured JSON.

        Returns a dictionary with role, experience_level, questions, and practice_plan.
        """
        if not role.strip():
            raise ValueError("role is required.")

        count = clamp_int(question_count, 1, 30)
        skill_text = ", ".join(skills or [])
        system_prompt = build_interview_system_prompt()
        user_prompt = f"""
Generate {count} interview questions for this candidate.

Return only valid JSON with this exact shape:
{{
  "role": "{role.strip()}",
  "experience_level": "{experience_level.strip() or "entry-level"}",
  "questions": [
    {{
      "id": 1,
      "type": "technical",
      "difficulty": "medium",
      "question": "Question text",
      "what_good_answer_covers": ["point 1", "point 2"],
      "follow_up": "Follow-up question"
    }}
  ],
  "practice_plan": ["practice step"]
}}

Requirements:
- Include a mix of technical, behavioral, project-based, and resume-based questions.
- Include system design questions only when appropriate for the experience level.
- Make questions concrete for the role and listed skills.
- Do not include markdown fences or commentary outside JSON.

Target role:
{role.strip()}

Experience level:
{experience_level.strip() or "entry-level"}

Skills:
{skill_text or "Not provided"}

Job description:
{truncate_text(job_description, 6000) or "Not provided"}

Resume:
{truncate_text(resume_text, 9000) or "Not provided"}
""".strip()

        raw_response = self.complete_text(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max(self.settings.max_tokens, 1800),
            temperature=0.25,
        )
        parsed = parse_json_response(raw_response)

        if parsed is None:
            return {
                "role": role.strip(),
                "experience_level": experience_level.strip() or "entry-level",
                "questions": [],
                "practice_plan": [],
                "raw_response": raw_response,
                "error": "Claude response was not valid JSON.",
            }

        return normalize_interview_questions(parsed, role, experience_level)

    def get_resume_improvement_tips(
        self,
        resume_text: str,
        job_description: str = "",
        target_role: str = "",
        max_tips: int = 10,
    ) -> dict[str, Any]:
        """
        Generate resume improvement tips as structured JSON.

        Returns prioritized tips for content, ATS keywords, clarity, and impact.
        """
        if not resume_text.strip():
            raise ValueError("resume_text is required.")

        tip_count = clamp_int(max_tips, 3, 20)
        system_prompt = build_resume_tips_system_prompt()
        user_prompt = f"""
Review this resume and provide up to {tip_count} improvement tips.

Return only valid JSON with this exact shape:
{{
  "summary": "Short assessment of the resume",
  "overall_priority": ["highest priority fix"],
  "tips": [
    {{
      "section": "Skills",
      "issue": "Specific issue",
      "recommendation": "Specific improvement",
      "example_rewrite": "Optional rewritten bullet or phrase",
      "impact": "high"
    }}
  ],
  "missing_keywords": ["keyword"],
  "ats_fixes": ["ATS improvement"],
  "next_steps": ["next action"]
}}

Rules:
- Be direct, practical, and specific.
- Focus on measurable achievements, skill alignment, ATS readability, and role fit.
- If a job description is provided, tailor tips to it.
- Do not invent degrees, companies, dates, certifications, or employment history.
- Do not include markdown fences or commentary outside JSON.

Target role:
{target_role.strip() or "Not provided"}

Job description:
{truncate_text(job_description, 6000) or "Not provided"}

Resume:
{truncate_text(resume_text, 12000)}
""".strip()

        raw_response = self.complete_text(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max(self.settings.max_tokens, 1800),
            temperature=0.2,
        )
        parsed = parse_json_response(raw_response)

        if parsed is None:
            return {
                "summary": "",
                "overall_priority": [],
                "tips": [],
                "missing_keywords": [],
                "ats_fixes": [],
                "next_steps": [],
                "raw_response": raw_response,
                "error": "Claude response was not valid JSON.",
            }

        return normalize_resume_tips(parsed)

    def complete_text(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Call Claude Messages API and return plain text."""
        response = self.client.messages.create(
            model=self.settings.model,
            max_tokens=max_tokens or self.settings.max_tokens,
            temperature=self.settings.temperature if temperature is None else temperature,
            system=system_prompt,
            messages=messages,
        )
        text = extract_message_text(response)
        if not text:
            raise ClaudeResponseError("Claude returned an empty response.")
        return text


class GeminiCareerHelper:
    """Gemini-powered helper using the Google Generative Language REST API."""

    def __init__(self, settings: GeminiSettings | None = None) -> None:
        self.settings = settings or GeminiSettings.from_env()

    def career_chatbot(
        self,
        user_message: str,
        resume_text: str = "",
        chat_history: list[dict[str, str]] | None = None,
        target_role: str = "",
        job_description: str = "",
        profile_context: dict[str, Any] | None = None,
    ) -> str:
        if not user_message.strip():
            raise ValueError("user_message is required.")

        system_prompt = build_career_system_prompt()
        context = build_context_block(
            resume_text=resume_text,
            job_description=job_description,
            target_role=target_role,
            profile_context=profile_context,
        )
        messages = normalize_messages(chat_history)
        current_message = "\n\n".join(
            part
            for part in (
                context,
                f"User question:\n{user_message.strip()}",
            )
            if part
        )
        append_user_message(messages, current_message)
        return self.complete_text(system_prompt=system_prompt, messages=messages)

    def complete_text(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        if not self.settings.api_key:
            raise GeminiConfigurationError("Gemini API key is missing.")

        response = httpx.post(
            GEMINI_ENDPOINT.format(model=self.settings.model),
            params={"key": self.settings.api_key},
            json=build_gemini_payload(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=max_tokens or self.settings.max_tokens,
                temperature=self.settings.temperature if temperature is None else temperature,
            ),
            timeout=25.0,
        )
        response.raise_for_status()
        text = extract_gemini_text(response.json())
        if not text:
            raise GeminiResponseError("Gemini returned an empty response.")
        return text


def career_chatbot(
    user_message: str,
    resume_text: str = "",
    chat_history: list[dict[str, str]] | None = None,
    target_role: str = "",
    job_description: str = "",
    profile_context: dict[str, Any] | None = None,
    helper: ClaudeCareerHelper | None = None,
) -> str:
    """Convenience function for career chatbot responses."""
    service = helper or ClaudeCareerHelper()
    return service.career_chatbot(
        user_message=user_message,
        resume_text=resume_text,
        chat_history=chat_history,
        target_role=target_role,
        job_description=job_description,
        profile_context=profile_context,
    )


def gemini_career_chatbot(
    user_message: str,
    resume_text: str = "",
    chat_history: list[dict[str, str]] | None = None,
    target_role: str = "",
    job_description: str = "",
    profile_context: dict[str, Any] | None = None,
    helper: GeminiCareerHelper | None = None,
) -> str:
    """Convenience function for Gemini career chatbot responses."""
    service = helper or GeminiCareerHelper()
    return service.career_chatbot(
        user_message=user_message,
        resume_text=resume_text,
        chat_history=chat_history,
        target_role=target_role,
        job_description=job_description,
        profile_context=profile_context,
    )


def gemini_complete_text(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int | None = None,
    temperature: float | None = None,
    helper: GeminiCareerHelper | None = None,
) -> str:
    """Complete a single prompt with Gemini."""
    service = helper or GeminiCareerHelper()
    return service.complete_text(
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )


def generate_interview_questions(
    role: str,
    experience_level: str = "entry-level",
    skills: list[str] | tuple[str, ...] | None = None,
    resume_text: str = "",
    job_description: str = "",
    question_count: int = 10,
    helper: ClaudeCareerHelper | None = None,
) -> dict[str, Any]:
    """Convenience function for interview question generation."""
    service = helper or ClaudeCareerHelper()
    return service.generate_interview_questions(
        role=role,
        experience_level=experience_level,
        skills=skills,
        resume_text=resume_text,
        job_description=job_description,
        question_count=question_count,
    )


def get_resume_improvement_tips(
    resume_text: str,
    job_description: str = "",
    target_role: str = "",
    max_tips: int = 10,
    helper: ClaudeCareerHelper | None = None,
) -> dict[str, Any]:
    """Convenience function for resume improvement tips."""
    service = helper or ClaudeCareerHelper()
    return service.get_resume_improvement_tips(
        resume_text=resume_text,
        job_description=job_description,
        target_role=target_role,
        max_tips=max_tips,
    )


def interview_question_generator(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for generate_interview_questions."""
    return generate_interview_questions(*args, **kwargs)


def resume_improvement_tips(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for get_resume_improvement_tips."""
    return get_resume_improvement_tips(*args, **kwargs)


def build_career_system_prompt() -> str:
    return """
You are a Universal Resume Analyzer + Career Assistant, ATS expert, and career assistant for any industry.
Always ground answers in the uploaded resume context and detected profile first.
Name the resume section used when possible, for example "Based on your Projects section..." or "Based on your ATS section...".
Never assume the candidate is in IT or software unless the profile context says so.
For doctors, teachers, HR, marketing, finance, mechanical, civil, legal, healthcare, research, design, business, and other non-IT profiles, avoid software-only advice such as React, DSA, GitHub, Docker, APIs, or backend roadmaps unless those terms appear in the profile.
Help with resume strategy, ATS readiness, role fit, job matching, skill gaps, certifications, projects/case work, and career growth.
Be concise, specific, encouraging, and honest. Ask one clarifying question only when it would materially improve the advice.
Do not answer outside the uploaded profile context. Do not invent degrees, employers, certifications, metrics, licenses, or projects.
""".strip()


def build_interview_system_prompt() -> str:
    return """
You are a senior interviewer and hiring manager.
Create realistic interview questions that assess skill depth, project ownership, problem solving, communication, and role fit.
Return only valid JSON when asked for JSON.
""".strip()


def build_resume_tips_system_prompt() -> str:
    return """
You are an expert resume reviewer and ATS optimization specialist.
Give specific, ethical resume improvements based only on evidence in the resume and job description.
Return only valid JSON when asked for JSON.
""".strip()


def build_context_block(
    resume_text: str = "",
    job_description: str = "",
    target_role: str = "",
    profile_context: dict[str, Any] | None = None,
) -> str:
    parts = []

    if profile_context:
        parts.append(f"Detected profile:\n{json.dumps(profile_context, ensure_ascii=False)}")
    if target_role.strip():
        parts.append(f"Target role:\n{target_role.strip()}")
    if job_description.strip():
        parts.append(f"Job description:\n{truncate_text(job_description, 6000)}")
    if resume_text.strip():
        parts.append(f"Resume:\n{truncate_text(resume_text, 10000)}")

    return "\n\n".join(parts)


def normalize_messages(chat_history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """
    Normalize chat history for Anthropic messages.

    Keeps user/assistant turns, merges consecutive same-role turns, and skips empty content.
    """
    normalized: list[dict[str, str]] = []

    for item in chat_history or []:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()

        if role not in {"user", "assistant"} or not content:
            continue

        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] = normalized[-1]["content"] + "\n\n" + content
        else:
            normalized.append({"role": role, "content": content})

    if normalized and normalized[0]["role"] == "assistant":
        normalized.insert(0, {"role": "user", "content": "Continue our career coaching conversation."})

    return normalized


def append_user_message(messages: list[dict[str, str]], content: str) -> None:
    """Append a user message while preserving alternating user/assistant turns."""
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] = messages[-1]["content"] + "\n\n" + content
    else:
        messages.append({"role": "user", "content": content})


def build_gemini_payload(
    system_prompt: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    response_mime_type: str | None = None,
) -> dict[str, Any]:
    contents = []
    for message in messages:
        role = "model" if message["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message["content"]}]})

    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }
    if response_mime_type:
        generation_config["responseMimeType"] = response_mime_type

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation_config,
    }
    if system_prompt.strip():
        payload["systemInstruction"] = {"parts": [{"text": system_prompt.strip()}]}
    return payload


def extract_gemini_text(payload: dict[str, Any]) -> str:
    pieces: list[str] = []
    for candidate in payload.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            text = part.get("text")
            if text:
                pieces.append(str(text))
    return "\n".join(pieces).strip()


def extract_message_text(response: Any) -> str:
    """Extract text from an Anthropic Messages API response."""
    pieces: list[str] = []

    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            pieces.append(text)
            continue

        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            pieces.append(str(block["text"]))

    return "\n".join(pieces).strip()


def parse_json_response(raw_text: str) -> dict[str, Any] | None:
    """Parse a JSON object from Claude text, tolerating accidental markdown fences."""
    if not raw_text:
        return None

    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_interview_questions(
    data: dict[str, Any],
    role: str,
    experience_level: str,
) -> dict[str, Any]:
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        questions = []

    normalized_questions = []
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            continue

        normalized_questions.append(
            {
                "id": int(question.get("id") or index),
                "type": str(question.get("type") or "general"),
                "difficulty": str(question.get("difficulty") or "medium"),
                "question": str(question.get("question") or "").strip(),
                "what_good_answer_covers": ensure_list(question.get("what_good_answer_covers")),
                "follow_up": str(question.get("follow_up") or "").strip(),
            }
        )

    return {
        "role": str(data.get("role") or role).strip(),
        "experience_level": str(data.get("experience_level") or experience_level or "entry-level").strip(),
        "questions": [item for item in normalized_questions if item["question"]],
        "practice_plan": ensure_list(data.get("practice_plan")),
    }


def normalize_resume_tips(data: dict[str, Any]) -> dict[str, Any]:
    tips = data.get("tips", [])
    if not isinstance(tips, list):
        tips = []

    normalized_tips = []
    for tip in tips:
        if not isinstance(tip, dict):
            continue
        normalized_tips.append(
            {
                "section": str(tip.get("section") or "General").strip(),
                "issue": str(tip.get("issue") or "").strip(),
                "recommendation": str(tip.get("recommendation") or "").strip(),
                "example_rewrite": str(tip.get("example_rewrite") or "").strip(),
                "impact": normalize_impact(tip.get("impact")),
            }
        )

    return {
        "summary": str(data.get("summary") or "").strip(),
        "overall_priority": ensure_list(data.get("overall_priority")),
        "tips": normalized_tips,
        "missing_keywords": ensure_list(data.get("missing_keywords")),
        "ats_fixes": ensure_list(data.get("ats_fixes")),
        "next_steps": ensure_list(data.get("next_steps")),
    }


def ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def normalize_impact(value: Any) -> str:
    impact = str(value or "medium").strip().lower()
    return impact if impact in {"high", "medium", "low"} else "medium"


def truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() + "\n[truncated]"


def clamp_int(value: int | float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, round(value)))


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default
