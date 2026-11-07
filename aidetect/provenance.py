
"""Metadata provenance scanner.

Looks for declared evidence in image metadata and known metadata segments.
Metadata can be stripped or forged, so it is reported separately from
pixel-level classification.

Important: this scanner detects indications of Content Credentials; it does
not validate C2PA signatures or prove that a manifest is authentic.
"""
from __future__ import annotations

import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_SIG_PATH = Path(__file__).parent / "data" / "signatures.yaml"
_MAX_TEXT = 1 << 20

# C2PA Manifest Store UUID defined by the C2PA specification.
_C2PA_MANIFEST_STORE_UUID = bytes.fromhex(
    "6332706100110010800000aa00389b71"
)


@lru_cache(maxsize=1)
def signatures() -> dict:
    with open(_SIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _compile(words: list[str]) -> re.Pattern:
    parts = [
        r"(?<![a-z0-9])" + re.escape(word.lower()) + r"(?![a-z0-9])"
        for word in words
    ]
    return re.compile("|".join(parts), re.IGNORECASE)


@lru_cache(maxsize=1)
def _regexes() -> dict[str, re.Pattern]:
    sig = signatures()
    return {
        "generator": _compile(sig["ai_generators"]),
        "standard": _compile(sig["ai_standards"]),
        "credentials": _compile(sig["content_credentials"]),
    }


def _decode(data: bytes) -> str:
    return data[:_MAX_TEXT].decode("utf-8", errors="ignore")


def _is_c2pa_jpeg_app11(segment: bytes) -> bool:
    """Recognize the C2PA Manifest Store header in a JPEG APP11 segment.

    JPEG APP11 is also used by other standards. We therefore check the JP
    framing, JUMBF superbox, description-box type, C2PA UUID and label
    instead of treating every APP11 segment as Content Credentials.

    This is structural detection only, not cryptographic validation.
    """
    # APP11 payload begins with JP, instance number and sequence number.
    if len(segment) < 8 or segment[:2] != b"JP":
        return False

    jumbf = segment[8:]

    # JUMBF Superbox header followed by a JUMBF Description box.
    if len(jumbf) < 32:
        return False
    if jumbf[4:8] != b"jumb":
        return False
    if jumbf[12:16] != b"jumd":
        return False
    if jumbf[17:33] != _C2PA_MANIFEST_STORE_UUID:
        return False

    # The description label is null-terminated and should occur near the
    # start of the manifest store. Do not scan the whole binary payload.
    return b"c2pa\x00" in jumbf[33:129]


def _jpeg_segments(raw: bytes) -> list[tuple[str, bytes]]:
    """Extract selected JPEG metadata segments without scanning pixel data."""
    out: list[tuple[str, bytes]] = []
    pos, n = 2, len(raw)

    if not raw.startswith(b"\xff\xd8"):
        return out

    while pos + 4 <= n:
        if raw[pos] != 0xFF:
            break

        marker = raw[pos + 1]

        if marker == 0xFF:  # marker fill byte
            pos += 1
            continue

        # Standalone markers have no length field.
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            pos += 2
            continue

        # Image scan data starts at SOS; do not scan compressed pixels.
        if marker in (0xD9, 0xDA):
            break

        length = struct.unpack(">H", raw[pos + 2:pos + 4])[0]
        if length < 2:
            break

        segment = raw[pos + 4:pos + 2 + length]
        if len(segment) != length - 2:
            break

        if marker == 0xE1 and segment.startswith(
            b"http://ns.adobe.com/xap"
        ):
            out.append(("xmp", segment))
        elif marker == 0xFE:
            out.append(("comment", segment))
        elif marker == 0xEB:
            # Keep APP11 separate: arbitrary binary JUMBF data must not be
            # decoded as text and matched against generic words like "jumb".
            out.append(("app11", segment))

        pos += 2 + length

    return out


def _png_chunks(raw: bytes) -> list[str]:
    """Return PNG chunk names while respecting file bounds."""
    signature = b"\x89PNG\r\n\x1a\n"
    if not raw.startswith(signature):
        return []

    types: list[str] = []
    pos, n = 8, len(raw)

    while pos + 12 <= n:
        length = struct.unpack(">I", raw[pos:pos + 4])[0]
        ctype = raw[pos + 4:pos + 8].decode("latin-1")

        # A chunk must contain its header, data and CRC within the file.
        chunk_end = pos + 12 + length
        if chunk_end > n:
            break

        types.append(ctype)

        if ctype == "IEND":
            break

        pos = chunk_end

    return types


def _webp_segments(raw: bytes) -> list[tuple[str, bytes]]:
    """Extract WebP XMP metadata chunks."""
    out: list[tuple[str, bytes]] = []

    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WEBP":
        return out

    pos, n = 12, len(raw)

    while pos + 8 <= n:
        tag = raw[pos:pos + 4]
        size = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        data_start = pos + 8
        data_end = data_start + size

        if data_end > n:
            break

        if tag == b"XMP ":
            out.append(("xmp", raw[data_start:data_end]))

        pos = data_end + (size & 1)

    return out


def scan(
    raw: bytes | None,
    fmt: str | None,
    info: dict[str, str],
    exif: dict[str, Any],
) -> dict:
    """Scan metadata for declared provenance evidence.

    Findings are indicators, not proof of image origin. The function does
    not verify cryptographic Content Credentials or C2PA signatures.
    """
    sig, rx = signatures(), _regexes()
    markers: list[dict] = []
    credentials = False
    texts: list[tuple[str, str, str]] = []

    # Text metadata supplied by the image decoder.
    for key, value in info.items():
        if isinstance(value, str) and key.lower() not in (
            "exif", "icc_profile"
        ):
            texts.append(("info", key, value))

    # Human-readable EXIF fields.
    for key, value in exif.items():
        if isinstance(value, str):
            texts.append(("exif", key, value))

    # Examine metadata segments in the original bytes, not compressed pixels.
    if raw:
        try:
            if fmt == "JPEG" or raw.startswith(b"\xff\xd8"):
                for source, segment in _jpeg_segments(raw):
                    if source == "app11":
                        if _is_c2pa_jpeg_app11(segment):
                            credentials = True
                        # Never treat generic binary APP11 data as text.
                        continue

                    texts.append((source, source, _decode(segment)))

            elif fmt == "PNG" or raw.startswith(b"\x89PNG\r\n\x1a\n"):
                if "caBX" in _png_chunks(raw):
                    credentials = True

            elif fmt == "WEBP" or raw.startswith(b"RIFF"):
                for source, segment in _webp_segments(raw):
                    texts.append((source, source, _decode(segment)))

        except (struct.error, IndexError):
            # Malformed metadata should not prevent an otherwise readable
            # image from being analysed.
            pass

    ai_keys = {key.lower() for key in sig["ai_text_keys"]}
    seen: set[tuple[str, str, str, str]] = set()

    def add(source: str, key: str, match: str, kind: str) -> None:
        signature = (source, key, match.lower(), kind)
        if signature not in seen and len(markers) < 20:
            seen.add(signature)
            markers.append({
                "source": source,
                "key": key,
                "match": match[:80],
                "kind": kind,
            })

    for source, key, text in texts:
        if source == "info" and key.lower() in ai_keys and text.strip():
            add(
                source,
                key,
                f"text chunk '{key}'",
                "generation_parameters",
            )

        for match in rx["standard"].finditer(text):
            add(
                source,
                key,
                match.group(0),
                "declared_synthetic_media",
            )

        for match in rx["generator"].finditer(text):
            add(source, key, match.group(0), "generator_name")

        if rx["credentials"].search(text):
            credentials = True

    camera_tags = sig["camera_tags"]
    camera = {
        tag: str(exif[tag])[:60]
        for tag in camera_tags
        if tag in exif and str(exif[tag]).strip()
    }

    software = exif.get("Software") or info.get("Software")

    return {
        "declares_ai": bool(markers),
        "ai_markers": markers,
        "content_credentials": credentials,
        "camera_evidence": camera,
        "camera_like": (
            len(camera) >= 3
            and ("Make" in camera or "Model" in camera)
        ),
        "software": str(software)[:80] if software else None,
    }
