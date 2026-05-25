from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Any

import pdfplumber

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None

try:
    import pytesseract
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None
    Image = None


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be read or parsed."""


def extract_text_from_pdf(file_path: str | Path, max_pages: int | None = None) -> str:
    """
    Extract clean text from a PDF resume using pdfplumber.

    Args:
        file_path: Path to the PDF file.
        max_pages: Optional page limit. Useful for very large PDFs.

    Returns:
        Cleaned text extracted from all readable pages.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF or max_pages is invalid.
        PDFExtractionError: If pdfplumber cannot parse the PDF.
    """
    return extract_pdf_content(file_path, max_pages=max_pages)["text"]


def extract_resume_text(file_path: str | Path, max_pages: int | None = None) -> str:
    """Alias for extract_text_from_pdf, named for resume-analyzer usage."""
    return extract_text_from_pdf(file_path, max_pages=max_pages)


def extract_pdf_content(
    file_path: str | Path,
    max_pages: int | None = None,
    enable_ocr: bool = True,
    min_words_for_ocr: int = 35,
) -> dict[str, Any]:
    """
    Extract resume PDF content with diagnostics.

    Extraction order:
    - pdfplumber text extraction
    - PyMuPDF text extraction fallback
    - Tesseract OCR fallback for image-based resumes when installed
    """
    path = _validate_pdf_path(file_path)

    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be a positive integer.")

    diagnostics: dict[str, Any] = {
        "filename": path.name,
        "page_count": 0,
        "character_count": 0,
        "word_count": 0,
        "table_count": 0,
        "image_count": 0,
        "multi_column_pages": 0,
        "extraction_methods": [],
        "ocr_attempted": False,
        "ocr_available": bool(pytesseract and Image and fitz),
        "warnings": [],
    }

    plumber_text, plumber_diagnostics = extract_with_pdfplumber(path, max_pages=max_pages)
    diagnostics.update(plumber_diagnostics)
    text = plumber_text
    if text:
        diagnostics["extraction_methods"].append("pdfplumber")

    if count_words(text) < min_words_for_ocr:
        pymupdf_text = extract_with_pymupdf(path, max_pages=max_pages)
        if count_words(pymupdf_text) > count_words(text):
            text = pymupdf_text
            diagnostics["extraction_methods"].append("pymupdf")

    if enable_ocr and count_words(text) < min_words_for_ocr:
        diagnostics["ocr_attempted"] = True
        if diagnostics["ocr_available"]:
            ocr_text = extract_with_tesseract(path, max_pages=max_pages)
            if count_words(ocr_text) > count_words(text):
                text = ocr_text
                diagnostics["extraction_methods"].append("tesseract_ocr")
        else:
            diagnostics["warnings"].append("OCR dependencies are not installed. Install PyMuPDF and pytesseract, plus the Tesseract binary.")

    text = clean_text(text)
    diagnostics["character_count"] = len(text)
    diagnostics["word_count"] = count_words(text)
    if not diagnostics["extraction_methods"]:
        diagnostics["extraction_methods"].append("none")

    return {"text": text, "diagnostics": diagnostics}


def extract_with_pdfplumber(path: Path, max_pages: int | None = None) -> tuple[str, dict[str, Any]]:
    page_texts: list[str] = []
    diagnostics = {
        "page_count": 0,
        "table_count": 0,
        "image_count": 0,
        "multi_column_pages": 0,
    }

    try:
        with pdfplumber.open(path) as pdf:
            pages = pdf.pages[:max_pages] if max_pages else pdf.pages
            diagnostics["page_count"] = len(pdf.pages)

            for page in pages:
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                text = text.strip()
                if text:
                    page_texts.append(text)

                diagnostics["image_count"] += len(getattr(page, "images", []) or [])

                try:
                    diagnostics["table_count"] += len(page.extract_tables() or [])
                except Exception:
                    pass

                try:
                    words = page.extract_words() or []
                    if is_probably_multicolumn(words, float(page.width or 0)):
                        diagnostics["multi_column_pages"] += 1
                except Exception:
                    pass
    except Exception as exc:
        raise PDFExtractionError(f"Could not extract text from PDF: {path.name}") from exc

    return clean_text("\n\n".join(page_texts)), diagnostics


def extract_with_pymupdf(path: Path, max_pages: int | None = None) -> str:
    if fitz is None:
        return ""

    page_texts: list[str] = []
    try:
        document = fitz.open(path)
        pages = range(min(len(document), max_pages)) if max_pages else range(len(document))
        for page_index in pages:
            page_text = document[page_index].get_text("text") or ""
            if page_text.strip():
                page_texts.append(page_text.strip())
        document.close()
    except Exception:
        return ""
    return clean_text("\n\n".join(page_texts))


def extract_with_tesseract(path: Path, max_pages: int | None = None) -> str:
    if fitz is None or pytesseract is None or Image is None:
        return ""

    page_texts: list[str] = []
    try:
        document = fitz.open(path)
        pages = range(min(len(document), max_pages)) if max_pages else range(len(document))
        for page_index in pages:
            page = document[page_index]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(BytesIO(pixmap.tobytes("png")))
            text = pytesseract.image_to_string(image) or ""
            if text.strip():
                page_texts.append(text.strip())
        document.close()
    except Exception:
        return ""
    return clean_text("\n\n".join(page_texts))


def extract_pdf_data(file_path: str | Path, max_pages: int | None = None) -> dict[str, Any]:
    """
    Extract text and simple metadata from a PDF.

    Returns a dictionary that is convenient for Flask JSON responses or database storage.
    """
    path = _validate_pdf_path(file_path)
    content = extract_pdf_content(path, max_pages=max_pages)
    return {
        "filename": path.name,
        "text": content["text"],
        **content["diagnostics"],
    }


def get_pdf_page_count(file_path: str | Path) -> int:
    """Return the number of pages in a PDF."""
    path = _validate_pdf_path(file_path)

    try:
        with pdfplumber.open(path) as pdf:
            return len(pdf.pages)
    except Exception as exc:
        raise PDFExtractionError(f"Could not read page count from PDF: {path.name}") from exc


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving useful paragraph breaks."""
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_words(text: str) -> int:
    """Count readable words in extracted resume text."""
    return len(re.findall(r"\b[\w+#.-]+\b", text))


def is_text_readable(text: str, min_words: int = 20) -> bool:
    """Return True when extracted text is likely usable for analysis."""
    return count_words(text) >= min_words


def is_probably_multicolumn(words: list[dict[str, Any]], page_width: float) -> bool:
    if not words or page_width <= 0 or len(words) < 35:
        return False

    left = 0
    middle = 0
    right = 0
    for word in words:
        x0 = float(word.get("x0", 0))
        if x0 < page_width * 0.42:
            left += 1
        elif x0 > page_width * 0.58:
            right += 1
        else:
            middle += 1

    return left >= 12 and right >= 12 and middle <= max(8, len(words) * 0.22)


def _validate_pdf_path(file_path: str | Path) -> Path:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        raise ValueError(f"PDF path is not a file: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    return path
