"""Backend-only integration for image analysis."""
from __future__ import annotations

import json
import os
import re
import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError

from .technical_evidence import inspect_image

SUPPORTED_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
MAX_BYTES = 10 * 1024 * 1024


class GeminiServiceError(RuntimeError):
    def __init__(self, message: str, status: int = 502, code: str = "analysis_unavailable") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def _root_env() -> Path:
    return Path(__file__).resolve().parents[1] / ".env"


load_dotenv(_root_env())


class GeminiImageService:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_MODEL", "").strip()

    def resolve_model(self) -> str:
        configured = self.model
        if configured:
            return configured
        if not self.api_key:
            raise GeminiServiceError(
                "Image analysis configuration is incomplete.", 503, "configuration_missing"
            )
        try:
            client = genai.Client(api_key=self.api_key)
            candidates = []
            for item in client.models.list():
                name = getattr(item, "name", "")
                actions = getattr(item, "supported_actions", None) or []
                methods = getattr(item, "supported_generation_methods", None) or []
                methods_text = " ".join(str(x) for x in list(actions) + list(methods)).lower()
                if "generatecontent" in methods_text and not methods_text:
                    continue
                if any(x in name.lower() for x in ("flash", "pro")):
                    candidates.append(name.split("/")[-1])
            preferred = next((x for x in candidates if "3" in x and "flash" in x.lower()), None)
            return preferred or next((x for x in candidates if "flash" in x.lower()), "gemini-2.0-flash")
        except Exception as exc:
            self._raise_sdk_error(exc)
        raise GeminiServiceError("Unable to determine an image analysis model.", 502, "model_unavailable")

    def _generate_content_models(self) -> list[str]:
        configured = self.model or self.resolve_model()
        fallbacks = ("gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-2.5-flash")
        return list(dict.fromkeys((configured, *(model for model in fallbacks if model != configured))))

    def analyze(self, filename: str, mime_type: str, image_bytes: bytes) -> dict[str, Any]:
        if not self.api_key:
            raise GeminiServiceError(
                "Image analysis configuration is incomplete.", 503, "configuration_missing"
            )
        if mime_type not in SUPPORTED_TYPES:
            raise GeminiServiceError(
                "Please upload a valid JPG, PNG or WEBP image.", 400, "unsupported_image"
            )
        if not image_bytes or len(image_bytes) > MAX_BYTES:
            raise GeminiServiceError(
                "The image is empty or exceeds the 10 MB upload limit.", 400, "invalid_image"
            )

        prompt = """You are an image analysis assistant for an AI Image Detection System.
Examine the uploaded image carefully. You must choose exactly one of two assessments: Likely AI-generated or Likely real.
Do not assume unusual features mean AI generation. Do not claim visual inspection definitively establishes provenance.
Provide observations based only on visible evidence and distinguish direct observations from possible alternative interpretations. Do not return Uncertain or any third assessment.
Do not invent metadata, camera information, source information, or forensic evidence.
Analyze: general description; objects/people; lighting/shadows; texture/fine details; background consistency; anatomical/structural irregularities; repeated patterns or unnatural elements; possible AI-generation signs; possible manipulation signs; characteristics consistent with real photography.
For every possible indicator, state the visible feature and the reason it could support or weaken that assessment. Include meaningful alternatives when they exist.
When evidence is subtle or balanced, choose the assessment supported by the stronger combination of visible indicators. Never return a middle or inconclusive category.
Return only valid JSON with this exact structure:
{"assessment":"Likely AI-generated|Likely real","description":"string","direct_observations":["specific visible facts"],"possible_ai_indicators":["observable detail and why it may support the assessment"],"possible_real_indicators":["observable detail and why it may support the assessment"],"uncertain_interpretations":["alternative explanations or uncertain findings"],"assessment_reasoning":"Explain why the selected assessment fits, while comparing both AI and real indicators. Include what evidence argues against the assessment.","explanation":"string","limitations":["string"]}"""

        try:
            client = genai.Client(api_key=self.api_key)
            response = None
            last_error = None
            for candidate in self._generate_content_models():
                try:
                    response = client.models.generate_content(
                        model=candidate,
                        contents=[
                            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                            prompt,
                        ],
                    )
                    break
                except Exception as exc:
                    last_error = exc
            if response is None:
                if last_error is not None:
                    self._raise_sdk_error(last_error)
                raise GeminiServiceError("Image analysis failed.", 502, "analysis_unavailable")
            text = getattr(response, "text", "") or ""
            parsed = self._parse_json(text)
            assessment = parsed.get("assessment")
            if assessment not in ("Likely AI-generated", "Likely real", "Uncertain"):
                raise GeminiServiceError(
                    "The image analysis response was malformed.", 502, "malformed_response"
                )
            if assessment == "Uncertain":
                ai_indicators = self._string_list(parsed.get("possible_ai_indicators"))
                real_indicators = self._string_list(parsed.get("possible_real_indicators"))
                assessment = "Likely AI-generated" if len(ai_indicators) > len(real_indicators) else "Likely real"
            parsed.setdefault("description", "")
            parsed["visual_observations"] = self._string_list(parsed.get("visual_observations"))
            parsed["possible_ai_indicators"] = self._string_list(parsed.get("possible_ai_indicators"))
            parsed["possible_real_indicators"] = self._string_list(parsed.get("possible_real_indicators"))
            parsed["direct_observations"] = self._string_list(parsed.get("direct_observations"))
            parsed["uncertain_interpretations"] = self._string_list(parsed.get("uncertain_interpretations"))
            parsed.setdefault("assessment_reasoning", parsed["explanation"])
            parsed.setdefault("explanation", "")
            parsed["limitations"] = self._string_list(parsed.get("limitations"))
            image_data = self._image_data(image_bytes)
            technical_evidence = inspect_image(image_bytes)
            return {
                "filename": filename,
                "assessment": assessment,
                "description": parsed["description"],
                "visual_observations": parsed["visual_observations"],
                "possible_ai_indicators": parsed["possible_ai_indicators"],
                "possible_real_indicators": parsed["possible_real_indicators"],
                "direct_observations": parsed["direct_observations"],
                "uncertain_interpretations": parsed["uncertain_interpretations"],
                "assessment_reasoning": parsed["assessment_reasoning"],
                "explanation": parsed["explanation"],
                "limitations": parsed["limitations"],
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "image_data": image_data,
                "technical_evidence": technical_evidence,
            }
        except GeminiServiceError:
            raise
        except Exception as exc:
            self._raise_sdk_error(exc)
        raise GeminiServiceError("Image analysis failed.", 502, "analysis_unavailable")

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if isinstance(item, (str, int, float)) and str(item).strip()][:30]

    @staticmethod
    def _image_data(image_bytes: bytes) -> dict[str, Any]:
        result: dict[str, Any] = {
            "file_size_bytes": len(image_bytes),
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
        }
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                result.update({
                    "width": image.width,
                    "height": image.height,
                    "format": image.format,
                    "mode": image.mode,
                })
        except (UnidentifiedImageError, OSError, ValueError):
            pass
        return result

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise GeminiServiceError(
                    "The image analysis response was malformed.", 502, "malformed_response"
                ) from exc
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as exc2:
                raise GeminiServiceError(
                    "The image analysis response was malformed.", 502, "malformed_response"
                ) from exc2
        if not isinstance(parsed, dict):
            raise GeminiServiceError(
                "The image analysis response was malformed.", 502, "malformed_response"
            )
        return parsed

    @staticmethod
    def _raise_sdk_error(exc: Exception) -> None:
        text = str(exc).lower()
        if "401" in text or "api key" in text or "unauthenticated" in text:
            raise GeminiServiceError(
                "Image analysis authentication failed. Please check the server configuration.",
                502,
                "authentication_failed",
            ) from exc
        if "429" in text or "quota" in text or "resource exhausted" in text:
            raise GeminiServiceError(
                "Image analysis is temporarily unavailable because the API usage limit has been reached.",
                429,
                "quota_exceeded",
            ) from exc
        if "404" in text or "not found" in text or "unavailable" in text:
            raise GeminiServiceError(
                "The configured image analysis model is unavailable. Please check server configuration.",
                503,
                "model_unavailable",
            ) from exc
        if "timeout" in text or "deadline" in text or "connection" in text or "network" in text:
            raise GeminiServiceError(
                "Unable to connect to the image analysis service. Please try again.",
                502,
                "network_error",
            ) from exc
        raise GeminiServiceError(
            "Image analysis is temporarily unavailable. Please try again.",
            502,
            "analysis_unavailable",
        ) from exc
