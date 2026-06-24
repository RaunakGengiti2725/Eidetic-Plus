"""Multimodal ingestion pipeline (dossier Section 11).

Turns any input into an IngestInput: the raw bytes (-> immutable substrate) plus an
embeddable text view (original text, OCR, transcription, or a model description).
Modalities Qwen cannot embed directly are stored raw and described; the raw object
stays ground truth. SHA-256 dedup happens in the engine BEFORE embedding to control cost.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import get_settings
from .dashscope_client import DashScopeClient
from .models import Modality

_IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
_PDF = {".pdf", ".docx", ".doc", ".pptx", ".md"}
_AUDIO = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".amr"}
_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_TEXT = {".txt", ".text", ".log", ".csv", ".json"}


@dataclass
class IngestInput:
    raw_bytes: bytes
    text: str
    modality: Modality
    is_described: bool
    source: str


def detect_modality(path: Path) -> Modality:
    ext = path.suffix.lower()
    if ext in _IMAGE:
        return Modality.IMAGE
    if ext == ".pdf" or ext in {".docx", ".doc", ".pptx"}:
        return Modality.PDF
    if ext in _AUDIO:
        return Modality.AUDIO
    if ext in _VIDEO:
        return Modality.VIDEO
    if ext in _TEXT or ext == ".md":
        return Modality.TEXT
    return Modality.BINARY


def from_text(text: str, source: str = "user") -> IngestInput:
    return IngestInput(raw_bytes=text.encode("utf-8"), text=text,
                       modality=Modality.TEXT, is_described=False, source=source)


def from_file(path: str, client: DashScopeClient, source: Optional[str] = None) -> IngestInput:
    p = Path(path)
    raw = p.read_bytes()
    source = source or p.name
    modality = detect_modality(p)

    if modality == Modality.TEXT:
        try:
            return IngestInput(raw, raw.decode("utf-8"), Modality.TEXT, False, source)
        except UnicodeDecodeError:
            modality = Modality.BINARY

    if modality == Modality.IMAGE:
        text = client.ocr_image(str(p)).strip()         # real qwen-vl-ocr
        described = False
        if len(text) < 8:                                # little/no text -> describe it
            text = client.describe_image(str(p)).strip()
            described = True
        return IngestInput(raw, text, Modality.IMAGE, described, source)

    if modality == Modality.PDF:
        text = client.read_document(str(p)).strip() if hasattr(client, "read_document") else ""
        if not text:
            text = _read_doc_via_describe(p, client)
        return IngestInput(raw, text, Modality.PDF, False, source)

    if modality == Modality.AUDIO:
        text = client.transcribe_audio(str(p)).strip()  # real qwen3-asr
        return IngestInput(raw, text, Modality.AUDIO, False, source)

    if modality == Modality.VIDEO:
        text = client.describe_video(str(p)).strip()     # real qwen-vl-plus
        return IngestInput(raw, text, Modality.VIDEO, True, source)

    # Un-embeddable: store raw, describe, embed the description.
    text = client.describe_binary(p.name, raw)
    return IngestInput(raw, text, Modality.BINARY, True, source)


def from_bytes(data: bytes, filename: str, client: DashScopeClient,
               source: Optional[str] = None) -> IngestInput:
    """For API uploads: persist to a temp file so model calls can read it, then route."""
    settings = get_settings()
    tmp_dir = settings.data_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=suffix, delete=False) as f:
        f.write(data)
        tmp_path = f.name
    try:
        return from_file(tmp_path, client, source=source or filename)
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


def _read_doc_via_describe(path: Path, client: DashScopeClient) -> str:
    """qwen-doc reader for PDFs/docs via the document model."""
    return client.read_document(str(path)) if hasattr(client, "read_document") else \
        client.describe_binary(path.name, path.read_bytes())
