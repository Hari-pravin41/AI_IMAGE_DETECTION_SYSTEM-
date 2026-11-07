"""Render in-memory PDF case reports from existing forensic report data."""
from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from PIL import ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_EXIF_FIELDS = {
    "Make",
    "Model",
    "DateTime",
    "DateTimeOriginal",
    "DateTimeDigitized",
    "Software",
    "Orientation",
    "ExposureTime",
    "FNumber",
    "ISOSpeedRatings",
    "FocalLength",
    "LensModel",
    "ImageDescription",
}
_DECISIONS = {
    "ai_generated": "AI-GENERATED",
    "real": "LIKELY REAL",
    "uncertain": "UNCERTAIN",
}
_NAVY = colors.HexColor("#19324d")
_BLUE = colors.HexColor("#2f5d7c")
_PALE = colors.HexColor("#edf3f7")
_MUTED = colors.HexColor("#536273")


class InvalidReportError(ValueError):
    """Raised when supplied report data cannot safely form a case report."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidReportError(f"report field '{field}' must be an object")
    return value


def _required_text(value: Any, field: str, max_length: int = 2048) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise InvalidReportError(f"report field '{field}' must be non-empty text")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidReportError(f"report field '{field}' must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise InvalidReportError(f"report field '{field}' must be between 0 and 1")
    return number


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidReportError(f"report field '{field}' must be an integer")
    return value


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    win_ansi_text = str(text).encode("cp1252", errors="replace").decode("cp1252")
    return Paragraph(escape(win_ansi_text).replace("\n", "<br/>"), style)


def _format_score(value: float) -> str:
    return f"{value:.4f} ({value:.1%})"


def _format_analysis_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Not available"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except ValueError:
        return text


def _table(rows: list[tuple[str, Any]], styles: dict[str, ParagraphStyle]) -> Table:
    data = [[_p(label, styles["label"]), _p(value, styles["body"])] for label, value in rows]
    table = Table(data, colWidths=(1.55 * inch, 5.45 * inch), repeatRows=0, splitByRow=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), _PALE),
        ("TEXTCOLOR", (0, 0), (0, -1), _NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5df")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _preview_bytes(report: Mapping[str, Any], original_image: bytes) -> bytes:
    facts = _mapping(report.get("file_integrity"), "file_integrity")
    expected_hash = _required_text(facts.get("sha256"), "file_integrity.sha256", 64)
    expected_dimensions = _mapping(facts.get("dimensions_px"), "file_integrity.dimensions_px")
    try:
        width = int(expected_dimensions["width"])
        height = int(expected_dimensions["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidReportError("report image dimensions are invalid") from exc
    if hashlib.sha256(original_image).hexdigest() != expected_hash:
        raise InvalidReportError("preview image does not match the report SHA-256")
    try:
        with PILImage.open(io.BytesIO(original_image)) as source:
            source.verify()
        with PILImage.open(io.BytesIO(original_image)) as source:
            if source.size != (width, height):
                raise InvalidReportError("preview dimensions do not match the report")
            source = ImageOps.exif_transpose(source).convert("RGB")
            source.thumbnail((900, 650), PILImage.Resampling.LANCZOS)
            output = io.BytesIO()
            source.save(output, format="PNG", optimize=True)
            return output.getvalue()
    except InvalidReportError:
        raise
    except (OSError, ValueError) as exc:
        raise InvalidReportError("preview image is not a valid readable image") from exc


def create_pdf_report(
    case_report: Mapping[str, Any],
    original_image: bytes | None = None,
) -> bytes:
    """Create a downloadable PDF in memory; no report or upload is persisted."""
    report = _mapping(case_report, "root")
    case_id = _required_text(report.get("case_id"), "case_id", 128)
    timestamp = _required_text(
        report.get("analysis_timestamp_utc"), "analysis_timestamp_utc", 128
    )
    facts = _mapping(report.get("file_integrity"), "file_integrity")
    metadata = _mapping(report.get("metadata_examination"), "metadata_examination")
    analysis = _mapping(report.get("authenticity_analysis"), "authenticity_analysis")
    model = _mapping(analysis.get("model"), "authenticity_analysis.model")
    manipulation = _mapping(report.get("manipulation_analysis"), "manipulation_analysis")
    decision = analysis.get("decision")
    if not isinstance(decision, str) or decision not in _DECISIONS:
        raise InvalidReportError("report decision must be real, ai_generated, or uncertain")
    score = _number(analysis.get("score"), "authenticity_analysis.score")
    threshold = _number(analysis.get("threshold"), "authenticity_analysis.threshold")
    filename = _required_text(
        facts.get("original_filename"), "file_integrity.original_filename"
    )
    sha256 = _required_text(facts.get("sha256"), "file_integrity.sha256", 64)
    mime_type = _required_text(
        facts.get("detected_mime_type"), "file_integrity.detected_mime_type", 128
    )
    image_format = _required_text(facts.get("format"), "file_integrity.format", 32)
    file_size = _integer(facts.get("byte_size"), "file_integrity.byte_size")
    dimensions = _mapping(facts.get("dimensions_px"), "file_integrity.dimensions_px")
    width = _integer(dimensions.get("width"), "file_integrity.dimensions_px.width")
    height = _integer(dimensions.get("height"), "file_integrity.dimensions_px.height")
    if file_size < 0 or width <= 0 or height <= 0:
        raise InvalidReportError("report file size or dimensions are invalid")
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256.lower()):
        raise InvalidReportError("report SHA-256 must contain 64 hexadecimal characters")
    model_id = _required_text(model.get("identifier"), "authenticity_analysis.model.identifier")
    limitations = report.get("limitations")
    if not isinstance(limitations, list) or len(limitations) > 100:
        raise InvalidReportError("report field 'limitations' must be a list")
    if original_image is not None and not isinstance(original_image, bytes):
        raise InvalidReportError("preview image must be bytes")
    if original_image is not None and len(original_image) != file_size:
        raise InvalidReportError("preview image size does not match the report")
    thumbnail = _preview_bytes(report, original_image) if original_image is not None else None

    stylesheet = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "ReportTitle", parent=stylesheet["Title"], fontName="Helvetica-Bold",
            fontSize=21, leading=25, textColor=_NAVY, alignment=TA_CENTER, spaceAfter=5,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=stylesheet["Normal"], fontSize=9,
            leading=13, textColor=_MUTED, alignment=TA_CENTER, spaceAfter=14,
        ),
        "section": ParagraphStyle(
            "ReportSection", parent=stylesheet["Heading2"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, textColor=_BLUE, spaceBefore=12, spaceAfter=6,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ReportBody", parent=stylesheet["BodyText"], fontName="Helvetica",
            fontSize=8.5, leading=11, textColor=colors.HexColor("#202a35"),
        ),
        "label": ParagraphStyle(
            "ReportLabel", parent=stylesheet["BodyText"], fontName="Helvetica-Bold",
            fontSize=8.5, leading=11, textColor=_NAVY,
        ),
        "small": ParagraphStyle(
            "ReportSmall", parent=stylesheet["BodyText"], fontName="Helvetica",
            fontSize=8, leading=11, textColor=_MUTED,
        ),
        "decision": ParagraphStyle(
            "ReportDecision", parent=stylesheet["BodyText"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, textColor=_NAVY, alignment=TA_CENTER,
        ),
    }
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=letter, rightMargin=0.75 * inch, leftMargin=0.75 * inch,
        topMargin=0.65 * inch, bottomMargin=0.62 * inch,
        title=f"AI Image Detection System - {case_id[:64]}",
        author="AI Image Detection System",
    )
    story: list[Any] = [
        _p("AI Image Detection System", styles["title"]),
        _p("Forensic case report · generated from the submitted analysis", styles["subtitle"]),
        _p("Case overview", styles["section"]),
    ]
    decision_table = Table(
        [[_p(_DECISIONS[decision], styles["decision"])]],
        colWidths=(7 * inch,),
    )
    decision_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _PALE),
        ("BOX", (0, 0), (-1, -1), 0.8, _BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([
        decision_table,
        Spacer(1, 7),
        _table([
            ("Case ID", case_id),
            ("Analysis date (UTC)", timestamp),
            ("Uploaded filename", filename),
            ("Model score (AI class)", _format_score(score)),
            ("Decision threshold", _format_score(threshold)),
            ("Model identifier", model_id),
            ("Model type", analysis.get("model", {}).get("model_type", "Not specified")),
        ], styles),
    ])
    if thumbnail:
        story.extend([
            _p("Image preview", styles["section"]),
            Image(io.BytesIO(thumbnail), width=4.0 * inch, height=2.9 * inch, kind="proportional"),
        ])
    story.extend([
        _p("Uploaded file and integrity", styles["section"]),
        _table([
            ("SHA-256 (original bytes)", sha256),
            ("Detected file type", f"{image_format} · {mime_type}"),
            ("File size", f"{file_size:,} bytes"),
            ("Image dimensions", f"{width:,} × {height:,} pixels"),
        ], styles),
        _p("Metadata examination", styles["section"]),
    ])
    exif = metadata.get("exif")
    if not isinstance(exif, Mapping):
        raise InvalidReportError("report field 'metadata_examination.exif' must be an object")
    metadata_rows: list[tuple[str, Any]] = [
        ("EXIF status", "Available" if exif else "Not present"),
    ]
    metadata_rows.extend(
        (f"EXIF · {key}", str(value)[:256])
        for key, value in exif.items()
        if key in _EXIF_FIELDS
    )
    declared = _mapping(
        metadata.get("declared_ai_metadata"), "metadata_examination.declared_ai_metadata"
    )
    metadata_rows.append(("Declared AI metadata", declared.get("status", "Not specified")))
    markers = declared.get("markers", [])
    if not isinstance(markers, list):
        raise InvalidReportError("declared AI metadata markers must be a list")
    for marker in markers[:20]:
        if isinstance(marker, Mapping):
            metadata_rows.append((
                "Declared marker",
                " · ".join(str(marker.get(key, ""))[:100] for key in ("source", "kind", "match")),
            ))
    metadata_rows.append((
        "Content Credentials (C2PA)",
        "Present" if declared.get("content_credentials_present") else "Not reported",
    ))
    metadata_rows.append((
        "Interpretation",
        metadata.get("interpretation", "Metadata may be missing or altered and is not proof of image origin."),
    ))
    story.append(_table(metadata_rows, styles))

    external = report.get("external_provenance_verification")
    if external is not None:
        if not isinstance(external, Mapping):
            raise InvalidReportError("external provenance verification must be an object")
        status = external.get("status")
        status_labels = {
            "detected": "SUPPORTED OPENAI PROVENANCE SIGNAL DETECTED",
            "not_detected": "NO SUPPORTED OPENAI PROVENANCE SIGNAL DETECTED",
            "inconclusive": "INCONCLUSIVE",
        }
        story.extend([
            _p("External provenance verification", styles["section"]),
            _p(status_labels.get(status, "INCONCLUSIVE"), styles["body"]),
            _p(str(external.get("summary", "No summary available."))[:600], styles["body"]),
        ])
        checked_at = external.get("checked_at")
        if isinstance(checked_at, str):
            story.append(_p("Checked at (UTC): " + checked_at[:80], styles["small"]))
        signals = external.get("results", [])
        if isinstance(signals, list):
            for signal in signals[:5]:
                if not isinstance(signal, Mapping):
                    continue
                parts = [
                    str(signal.get("type", "signal"))[:30],
                    str(signal.get("outcome", "unknown"))[:30],
                ]
                for field in ("validation_state", "issuer", "model", "generated_at"):
                    value = signal.get(field)
                    if isinstance(value, str) and value:
                        parts.append(field + ": " + value[:100])
                story.append(_p(" | ".join(parts), styles["small"]))

    story.extend([
        _p("Interpretation and limitations", styles["section"]),
        _p(
            "The AI-class score is not necessarily a calibrated probability. A model prediction "
            "is an estimate, not proof of image origin or authenticity.",
            styles["body"],
        ),
        Spacer(1, 5),
        _p(
            "Image manipulation detection is not implemented. "
            + str(manipulation.get(
                "explanation",
                "This report does not determine whether the image was edited or tampered with.",
            )),
            styles["body"],
        ),
    ])
    for limitation in limitations:
        story.append(_p(f"• {limitation}", styles["small"]))
    external_for_conclusion = report.get("external_provenance_verification")
    if isinstance(external_for_conclusion, Mapping) and external_for_conclusion.get("status") == "detected":
        conclusion = (
            "EXTERNAL PROVENANCE FINDING: a supported OpenAI provenance signal was detected. "
            f"The separate visual-model decision is {_DECISIONS[decision]} "
            f"(AI-class score {_format_score(score)}; threshold {_format_score(threshold)}). "
            "These findings disagree. Do not interpret the visual model's label as confirmation "
            "that the image is real; the provenance finding is separate evidence about origin. "
            "The SHA-256 identifies the uploaded bytes; manipulation detection is not implemented."
        )
    else:
        conclusion = (
            f"Based on the submitted case report, the model decision is {_DECISIONS[decision]} "
            f"(AI-class score {_format_score(score)}; threshold {_format_score(threshold)}). "
            "This is a model estimate, not proof. The SHA-256 identifies the uploaded bytes; "
            "manipulation detection is not implemented."
        )
    story.extend([
        _p("Forensic conclusion", styles["section"]),
        _p(conclusion, styles["body"]),
    ])
    if metadata.get("exif_status") == "not_present":
        story.append(_p("No EXIF metadata was present in the case report. Missing metadata alone is not evidence of AI generation.", styles["small"]))

    def add_page_number(pdf_canvas: canvas.Canvas, _document: Any) -> None:
        pdf_canvas.saveState()
        pdf_canvas.setStrokeColor(colors.HexColor("#cbd5df"))
        pdf_canvas.setLineWidth(0.5)
        pdf_canvas.line(0.75 * inch, 0.45 * inch, 7.75 * inch, 0.45 * inch)
        pdf_canvas.setFont("Helvetica", 8)
        pdf_canvas.setFillColor(_MUTED)
        pdf_canvas.drawString(0.75 * inch, 0.28 * inch, "AI Image Detection System · assessment is not proof")
        pdf_canvas.drawRightString(7.75 * inch, 0.28 * inch, f"Page {pdf_canvas.getPageNumber()}")
        pdf_canvas.restoreState()

    def make_canvas(*args: Any, **kwargs: Any) -> canvas.Canvas:
        kwargs["pageCompression"] = 0
        return canvas.Canvas(*args, **kwargs)

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
        canvasmaker=make_canvas,
    )
    return buffer.getvalue()


def create_analysis_pdf_report(result: Mapping[str, Any], original_image: bytes) -> bytes:
    """Create a PDF for an image-analysis service result."""
    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "AnalysisTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=21, leading=25, textColor=_NAVY, alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "AnalysisSection", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, leading=15, textColor=_BLUE, spaceBefore=12, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "AnalysisBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9, leading=13, textColor=colors.HexColor("#202a35"),
    )
    document = SimpleDocTemplate(
        buffer, pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch,
        title="AI Image Detection System", author="AI Image Detection System",
    )
    story = [
        _p("AI Image Detection System", title_style),
        Spacer(1, 10),
    ]
    image_data = result.get("image_data", {})
    rows = [
        ("Report ID", result.get("analysis_id", "")),
        ("Uploaded filename", result.get("filename", "")),
        ("Analysis date and time", _format_analysis_timestamp(result.get("analysis_timestamp", ""))),
        ("Assessment", result.get("assessment", "")),
        ("Image dimensions", f"{image_data.get('width', '?')} × {image_data.get('height', '?')}"),
        ("File size", f"{image_data.get('file_size_bytes', 0):,} bytes" if image_data else "Not available"),
        ("Image format", image_data.get("format", "Not available")),
        ("SHA-256", image_data.get("sha256", "Not available")),
        ("Human review", {
            "confirmed": "Assessment confirmed by reviewer",
            "disputed": "Assessment disputed by reviewer",
        }.get(result.get("review_status"), "Not reviewed")),
    ]
    story.append(_table(rows, {"label": body_style, "body": body_style}))
    try:
        source = PILImage.open(io.BytesIO(original_image))
        source = ImageOps.exif_transpose(source).convert("RGB")
        source.thumbnail((450, 330), PILImage.Resampling.LANCZOS)
        preview = io.BytesIO()
        source.save(preview, format="PNG", optimize=True)
        story.extend([
            Spacer(1, 10),
            _p("Uploaded image preview", section_style),
            Image(io.BytesIO(preview.getvalue()), width=3.8 * inch, height=2.8 * inch, kind="proportional"),
        ])
    except (OSError, ValueError):
        pass
    story.append(_p("General image description", section_style))
    story.append(_p(result.get("description", "No description returned."), body_style))

    ai_indicators = result.get("possible_ai_indicators", [])
    real_indicators = result.get("possible_real_indicators", [])
    story.extend([
        _p("Evidence overview", section_style),
        _table([
            ("AI-generation indicators", len(ai_indicators) if isinstance(ai_indicators, list) else 0),
            ("Real-image indicators", len(real_indicators) if isinstance(real_indicators, list) else 0),
            ("Alternative explanations", len(result.get("uncertain_interpretations", []))
             if isinstance(result.get("uncertain_interpretations", []), list) else 0),
        ], {"label": body_style, "body": body_style}),
    ])

    technical = result.get("technical_evidence", {})
    if isinstance(technical, Mapping):
        pixel_facts = technical.get("pixel_facts", {})
        metadata = technical.get("metadata", {})
        technical_rows = [
            ("Technical inspection", technical.get("status", "Not available")),
            ("Estimated grayscale entropy",
             pixel_facts.get("estimated_entropy", "Not available") if isinstance(pixel_facts, Mapping) else "Not available"),
            ("EXIF observations", ", ".join(f"{key}: {value}" for key, value in metadata.items()) if metadata else "No relevant EXIF fields recorded"),
        ]
        provenance = technical.get("provenance", {})
        if isinstance(provenance, Mapping):
            technical_rows.extend([
                ("Declared AI marker", "Detected" if provenance.get("declares_ai") else "Not detected"),
                ("C2PA marker", "Detected (unverified)" if provenance.get("content_credentials_present") else "Not detected"),
                ("Camera-like evidence", "Detected" if provenance.get("camera_like") else "Not detected"),
                ("Provenance summary", provenance.get("summary", "No declared provenance signal was found.")),
            ])
        story.append(KeepTogether([
            _p("Verified technical image facts", section_style),
            _table(technical_rows, {"label": body_style, "body": body_style}),
        ]))
        embedded_signals = technical.get("embedded_signals", [])
        if isinstance(embedded_signals, list):
            for signal in embedded_signals:
                if isinstance(signal, str) and signal.strip():
                    story.append(_p(f"• {signal}", body_style))

    sections = [
        ("Direct visual observations", result.get("direct_observations", result.get("visual_observations", []))),
        ("Why this assessment was selected", result.get("assessment_reasoning", result.get("explanation", ""))),
        ("Possible AI-generated indicators", result.get("possible_ai_indicators", [])),
        ("Possible real-image indicators", result.get("possible_real_indicators", [])),
        ("Uncertain interpretations and alternatives", result.get("uncertain_interpretations", [])),
        ("Explanation", result.get("explanation", "")),
        ("Limitations", result.get("limitations", [])),
    ]
    for heading, value in sections:
        if isinstance(value, list):
            value = [item for item in value if isinstance(item, str) and item.strip()]
        if not value:
            continue
        section = [_p(heading, section_style)]
        if isinstance(value, list):
            for item in value:
                section.append(_p(f"• {item}", body_style))
        else:
            section.append(_p(value, body_style))
        story.append(KeepTogether(section))
    story.extend([
        _p("Scientific limitation and disclaimer", section_style),
        _p(
            "This result is an AI-assisted visual assessment based on visible image characteristics. "
            "It is not definitive forensic proof of whether an image was AI-generated or authentic.",
            body_style,
        ),
    ])

    def add_page_number(pdf_canvas: canvas.Canvas, document: Any) -> None:
        pdf_canvas.saveState()
        pdf_canvas.setStrokeColor(colors.HexColor("#cbd5df"))
        pdf_canvas.line(0.65 * inch, 0.48 * inch, 7.85 * inch, 0.48 * inch)
        pdf_canvas.setFont("Helvetica", 8)
        pdf_canvas.setFillColor(_MUTED)
        pdf_canvas.drawString(0.65 * inch, 0.30 * inch, "AI Image Detection System")
        pdf_canvas.drawRightString(7.85 * inch, 0.30 * inch, f"Page {pdf_canvas.getPageNumber()}")
        pdf_canvas.restoreState()

    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return buffer.getvalue()
