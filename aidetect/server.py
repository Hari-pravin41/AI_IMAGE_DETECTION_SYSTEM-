"""Local web UI + JSON API."""
from __future__ import annotations

import base64
import io
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import re
import smtplib
from email.message import EmailMessage

from flask import Flask, jsonify, make_response, redirect, render_template, request, send_file, session
from PIL import Image, UnidentifiedImageError

from .config import Config
from .detector import Detector
from .explain import render_panels
from .forensics import build_case_report
from .loader import ImageError, load_image
from .pdf_report import InvalidReportError, create_pdf_report
from .scope import SCOPE_WARNING, SUPPORTED_DOMAIN
from .gemini_service import GeminiImageService, GeminiServiceError, SUPPORTED_TYPES
from .history_store import HistoryStore
from .auth_store import AuthStore

_TEMPLATES = Path(__file__).parent / "templates"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_ANALYSIS_REQUESTS: dict[str, list[float]] = {}
_ANALYSIS_REQUESTS_LOCK = threading.Lock()


def _send_password_reset_email(email: str, token: str, app: Flask) -> bool:
    host = os.getenv("SMTP_HOST", "").strip()
    if not host:
        app.logger.warning(
            "SMTP is not configured; password reset token for %s was not emailed.",
            email,
        )
        if os.getenv("PASSWORD_RESET_DEV_LINKS", "").lower() in ("1", "true", "yes"):
            base_url = os.getenv("APP_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
            app.logger.warning("Development reset link: %s/reset-password?token=%s", base_url, token)
        return False
    message = EmailMessage()
    message["Subject"] = "Reset your AI Image Detection System password"
    message["From"] = os.getenv("SMTP_FROM", os.getenv("SMTP_USERNAME", "")).strip()
    message["To"] = email
    base_url = os.getenv("APP_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    reset_url = f"{base_url}/reset-password?token={token}"
    message.set_content(
        "A password reset was requested for your AI Image Detection System account.\n\n"
        f"Reset your password: {reset_url}\n\n"
        "This link expires in 30 minutes and can be used once. "
        "If you did not request it, you can ignore this email."
    )
    port = int(os.getenv("SMTP_PORT", "587"))
    try:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            if os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes"):
                smtp.starttls()
            username = os.getenv("SMTP_USERNAME", "").strip()
            password = os.getenv("SMTP_PASSWORD", "")
            if username:
                smtp.login(username, password)
            smtp.send_message(message)
        return True
    except (smtplib.SMTPException, OSError):
        app.logger.exception("Password reset email delivery failed")
        return False


def _send_verification_email(email: str, token: str, app: Flask) -> bool:
    host = os.getenv("SMTP_HOST", "").strip()
    if not host:
        if os.getenv("EMAIL_VERIFICATION_DEV_LINKS", "").lower() in ("1", "true", "yes"):
            base_url = os.getenv("APP_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
            app.logger.warning("Development verification link: %s/verify-email?token=%s", base_url, token)
        return False
    message = EmailMessage()
    message["Subject"] = "Verify your AI Image Detection System email"
    message["From"] = os.getenv("SMTP_FROM", os.getenv("SMTP_USERNAME", "")).strip()
    message["To"] = email
    base_url = os.getenv("APP_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    message.set_content(
        "Verify your email address for AI Image Detection System.\n\n"
        f"{base_url}/verify-email?token={token}\n\n"
        "This link expires in 60 minutes. If you did not create this account, ignore this email."
    )
    port = int(os.getenv("SMTP_PORT", "587"))
    try:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            if os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes"):
                smtp.starttls()
            username = os.getenv("SMTP_USERNAME", "").strip()
            if username:
                smtp.login(username, os.getenv("SMTP_PASSWORD", ""))
            smtp.send_message(message)
        return True
    except (smtplib.SMTPException, OSError):
        app.logger.exception("Verification email delivery failed")
        return False


def _configure_auth(app: Flask, auth_store: AuthStore) -> None:
    secret_key = os.getenv("APP_SECRET_KEY", "").strip()
    if not secret_key:
        secret_file = Path(__file__).resolve().parents[1] / "data" / ".app_secret"
        secret_file.parent.mkdir(parents=True, exist_ok=True)
        if not secret_file.exists():
            secret_file.write_text(__import__("secrets").token_hex(32), encoding="utf-8")
        secret_key = secret_file.read_text(encoding="utf-8").strip()
    app.secret_key = secret_key
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = os.getenv("SESSION_COOKIE_SECURE", "").lower() in ("1", "true", "yes")

    @app.get("/api/auth/me")
    def auth_me():
        if "user_id" not in session:
            return jsonify({"authenticated": False}), 200
        try:
            return jsonify({"authenticated": True, "user": auth_store.get_by_id(session["user_id"])})
        except KeyError:
            session.clear()
            return jsonify({"authenticated": False}), 200

    @app.post("/api/auth/register")
    def auth_register():
        payload = request.get_json(silent=True) or {}
        confirmation = str(payload.get("confirmation_password", ""))
        password = str(payload.get("password", ""))
        if confirmation and password != confirmation:
            return jsonify({"error": "Password and confirmation must match exactly. Passwords are case-sensitive.", "code": "password_mismatch"}), 400
        try:
            user = auth_store.create(str(payload.get("name", "")), str(payload.get("email", "")), password)
        except ValueError as exc:
            return jsonify({"error": str(exc), "code": "invalid_registration"}), 400
        session.clear()
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        verification_token = auth_store.create_verification_token(user["id"])
        _send_verification_email(user["email"], verification_token, app)
        return jsonify({"user": user}), 201

    @app.get("/verify-email")
    def verify_email_page():
        return render_template("verify_email.html", token=request.args.get("token", "").strip())

    @app.post("/api/auth/verify-email")
    def verify_email():
        payload = request.get_json(silent=True) or {}
        verified = auth_store.verify_email(str(payload.get("token", "")).strip())
        if not verified:
            return jsonify({"error": "This verification link is invalid or has expired.", "code": "invalid_verification_token"}), 400
        return jsonify({"message": "Email address verified successfully."}), 200

    @app.post("/api/auth/resend-verification")
    def resend_verification():
        if "user_id" not in session:
            return jsonify({"error": "Please sign in.", "code": "authentication_required"}), 401
        user = auth_store.get_by_id(session["user_id"])
        if user["email_verified"]:
            return jsonify({"message": "Your email address is already verified."}), 200
        token = auth_store.create_verification_token(user["id"])
        _send_verification_email(user["email"], token, app)
        return jsonify({"message": "Verification instructions have been sent if email delivery is configured."}), 200

    @app.post("/api/auth/login")
    def auth_login():
        payload = request.get_json(silent=True) or {}
        user = auth_store.authenticate(str(payload.get("email", "")), str(payload.get("password", "")))
        if user is None:
            return jsonify({"error": "Email or password is incorrect. Passwords are case-sensitive.", "code": "invalid_credentials"}), 401
        session.clear()
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        return jsonify({"user": user})

    @app.post("/api/auth/forgot-password")
    def auth_forgot_password():
        payload = request.get_json(silent=True) or {}
        token = auth_store.create_reset_token(str(payload.get("email", "")))
        if token:
            _send_password_reset_email(str(payload.get("email", "")).strip().lower(), token, app)
        return jsonify({
            "message": "If an account exists for this email, password-reset instructions will be sent."
        }), 200

    @app.get("/reset-password")
    def reset_password_page():
        return render_template("reset_password.html", token=request.args.get("token", "").strip())

    @app.post("/api/auth/reset-password")
    def auth_reset_password():
        payload = request.get_json(silent=True) or {}
        token = str(payload.get("token", "")).strip()
        password = str(payload.get("password", ""))
        if not token:
            return jsonify({"error": "This reset link is invalid.", "code": "invalid_reset_token"}), 400
        try:
            reset = auth_store.reset_password(token, password)
        except ValueError as exc:
            return jsonify({"error": str(exc), "code": "invalid_password"}), 400
        if not reset:
            return jsonify({"error": "This reset link is invalid or has expired.", "code": "invalid_reset_token"}), 400
        session.clear()
        return jsonify({"message": "Password updated. You can now sign in."}), 200

    @app.post("/api/auth/logout")
    def auth_logout():
        session.clear()
        return jsonify({"status": "logged_out"})


def _register_gemini_routes(app: Flask, history: HistoryStore) -> None:
    service = GeminiImageService()

    @app.get("/api/dashboard/summary")
    def dashboard_summary():
        if "user_id" not in session:
            return jsonify({"error": "Please sign in.", "code": "authentication_required"}), 401
        return jsonify(history.summary(session["user_id"])), 200

    @app.post("/api/reports/<analysis_id>/share")
    def create_report_share(analysis_id):
        if "user_id" not in session:
            return jsonify({"error": "Please sign in.", "code": "authentication_required"}), 401
        token = history.create_share(analysis_id, session["user_id"])
        if token is None:
            return jsonify({"error": "Analysis not found.", "code": "not_found"}), 404
        base_url = os.getenv("APP_BASE_URL", request.url_root.rstrip("/"))
        return jsonify({
            "share_url": f"{base_url}/shared-report/{token}",
            "expires_hours": 168,
        }), 201

    @app.get("/shared-report/<token>")
    def shared_report(token):
        analysis_id = history.shared_analysis(token)
        if analysis_id is None:
            return "This shared report link is invalid, expired, or revoked.", 404
        try:
            result = history.get(analysis_id)
            image_record = history.image(analysis_id)
        except KeyError:
            return "This shared report is no longer available.", 404
        if image_record is None:
            return "This shared report is no longer available.", 404
        from .pdf_report import create_analysis_pdf_report
        try:
            return send_file(
                io.BytesIO(create_analysis_pdf_report(result, image_record[0])),
                mimetype="application/pdf",
                as_attachment=False,
                download_name=f"AI-Image-Detection-System-{re.sub(r'[^A-Za-z0-9_-]', '', analysis_id)}.pdf",
            )
        except (InvalidReportError, OSError, ValueError):
            app.logger.exception("Shared report PDF generation failed")
            return "The shared report could not be generated.", 500

    @app.post("/api/analyze-image")
    def analyze_image():
        if "user_id" not in session:
            return jsonify({"error": "Please sign in to analyze an image.", "code": "authentication_required"}), 401
        limit = max(1, int(app.config.get("ANALYSIS_RATE_LIMIT", 10)))
        window = max(60, int(app.config.get("ANALYSIS_RATE_WINDOW_SECONDS", 3600)))
        now = time.monotonic()
        client_ip = request.remote_addr or "unknown"
        with _ANALYSIS_REQUESTS_LOCK:
            timestamps = [stamp for stamp in _ANALYSIS_REQUESTS.get(client_ip, []) if now - stamp < window]
            if len(timestamps) >= limit:
                return jsonify({
                    "error": "Analysis rate limit reached. Please wait before analyzing another image.",
                    "code": "rate_limit",
                }), 429
            timestamps.append(now)
            _ANALYSIS_REQUESTS[client_ip] = timestamps
        uploaded = request.files.get("image") or request.files.get("file")
        if uploaded is None:
            return jsonify({"error": "Select an image to analyze.", "code": "missing_image"}), 400
        if not uploaded.filename:
            return jsonify({"error": "Please upload a valid JPG, PNG or WEBP image.", "code": "invalid_image"}), 400
        data = uploaded.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            return jsonify({"error": "The image exceeds the 10 MB upload limit.", "code": "file_too_large"}), 413
        mime_type = (uploaded.mimetype or "").lower()
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"
        if mime_type not in SUPPORTED_TYPES:
            return jsonify({"error": "Please upload a valid JPG, PNG or WEBP image.", "code": "unsupported_image"}), 400
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            return jsonify({"error": "Please upload a valid JPG, PNG or WEBP image.", "code": "invalid_image"}), 400
        try:
            result = service.analyze(Path(uploaded.filename).name, mime_type, data)
            saved = history.add(result, data, mime_type, session["user_id"])
            return jsonify(saved), 200
        except GeminiServiceError as exc:
            return jsonify({"error": str(exc), "code": exc.code}), exc.status

    @app.get("/api/history/<analysis_id>")
    def analysis_detail(analysis_id):
        if "user_id" not in session:
            return jsonify({"error": "Please sign in.", "code": "authentication_required"}), 401
        try:
            return jsonify(history.get(analysis_id, session["user_id"]))
        except KeyError:
            return jsonify({"error": "Analysis not found.", "code": "not_found"}), 404

    @app.delete("/api/history/<analysis_id>")
    def analysis_delete(analysis_id):
        if "user_id" not in session or not history.delete(analysis_id, session["user_id"]):
            return jsonify({"error": "Analysis not found.", "code": "not_found"}), 404
        return jsonify({"status": "deleted", "analysis_id": analysis_id})

    @app.post("/api/history/<analysis_id>/review")
    def analysis_review(analysis_id):
        if "user_id" not in session:
            return jsonify({"error": "Please sign in.", "code": "authentication_required"}), 401
        payload = request.get_json(silent=True) or {}
        status = str(payload.get("status", "")).strip().lower()
        if not history.set_review(analysis_id, session["user_id"], status):
            return jsonify({"error": "The review could not be saved.", "code": "review_not_saved"}), 400
        return jsonify(history.get(analysis_id, session["user_id"])), 200

    @app.get("/api/reports/<analysis_id>/pdf")
    def analysis_pdf(analysis_id):
        if "user_id" not in session:
            return jsonify({"error": "Please sign in.", "code": "authentication_required"}), 401
        try:
            result = history.get(analysis_id, session["user_id"])
        except KeyError:
            return jsonify({"error": "Analysis not found.", "code": "not_found"}), 404
        image_record = history.image(analysis_id, session["user_id"])
        if image_record is None:
            return jsonify({"error": "Analysis not found.", "code": "not_found"}), 404
        from .pdf_report import create_analysis_pdf_report
        try:
            pdf_bytes = create_analysis_pdf_report(result, image_record[0])
        except (InvalidReportError, OSError, ValueError) as exc:
            app.logger.exception("Analysis PDF generation failed")
            return jsonify({"error": "The PDF report could not be generated.", "code": "report_generation_failed"}), 500
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", result["filename"])[:64].strip(".-_")
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"AI-Image-Detection-System-{safe_name or 'report'}.pdf",
        )


def _remember_results(app: Flask, results: list[dict]) -> None:
    history = app.config.setdefault("ANALYSIS_HISTORY", [])
    for result in results:
        report = result.get("forensic_report") if isinstance(result, dict) else None
        if not isinstance(report, dict):
            continue
        history.append({
            "case_id": report.get("case_id"),
            "filename": report.get("file_integrity", {}).get("original_filename", result.get("name")),
            "timestamp": report.get("analysis_timestamp_utc"),
            "label": result.get("label"),
            "prob_ai": result.get("prob_ai"),
            "threshold": result.get("threshold"),
        })
    app.config["ANALYSIS_HISTORY"] = history[-50:]


def _register_pdf_report_route(app: Flask, cfg: Config) -> None:
    @app.post("/api/report/pdf")
    def download_pdf_report():
        raw_report = request.form.get("report")
        if not raw_report:
            return jsonify({"error": "include the case report in the 'report' form field"}), 400
        try:
            report = json.loads(raw_report)
        except json.JSONDecodeError:
            return jsonify({"error": "the 'report' field must contain valid JSON"}), 400

        image_file = request.files.get("image")
        image_bytes = None
        if image_file:
            image_bytes = image_file.read(int(cfg.max_upload_mb * 1024 * 1024) + 1)
            if len(image_bytes) > cfg.max_upload_mb * 1024 * 1024:
                return jsonify({"error": f"file larger than {cfg.max_upload_mb:g} MB"}), 413
        try:
            pdf_bytes = create_pdf_report(report, image_bytes)
        except InvalidReportError as exc:
            return jsonify({"error": str(exc)}), 400
        case_id = report.get("case_id", "case") if isinstance(report, dict) else "case"
        safe_case_id = re.sub(r"[^A-Za-z0-9._-]+", "-", str(case_id))[:64].strip(".-_")
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"image-forensic-report-{safe_case_id or 'case'}.pdf",
        )


def create_app(detector: Detector) -> Flask:
    cfg = detector.cfg
    app = Flask(__name__, template_folder=str(_TEMPLATES))
    app.config["MAX_CONTENT_LENGTH"] = int(cfg.max_upload_mb * cfg.max_batch_files * 1024 * 1024)
    app.config["ANALYSIS_HISTORY"] = []
    app.config["ANALYSIS_RATE_LIMIT"] = int(os.getenv("ANALYSIS_RATE_LIMIT", "10"))
    app.config["ANALYSIS_RATE_WINDOW_SECONDS"] = int(os.getenv("ANALYSIS_RATE_WINDOW_SECONDS", "3600"))
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 1024
    history_store = HistoryStore()
    auth_store = AuthStore()
    _configure_auth(app, auth_store)
    _register_gemini_routes(app, history_store)
    _register_pdf_report_route(app, cfg)

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
        return response

    @app.get("/")
    def index():
        if "user_id" not in session:
            return render_template("landing.html")
        return render_template("index.html", info=detector.info(), max_mb=cfg.max_upload_mb,
                               max_files=cfg.max_batch_files)

    @app.get("/login")
    def login_page():
        if "user_id" in session:
            return redirect("/")
        response = make_response(render_template("login.html"))
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/model")
    def model():
        return jsonify({
            "model_type": "forensic",
            "supported_domain": SUPPORTED_DOMAIN,
            "scope_warning": SCOPE_WARNING,
            **detector.info(),
        })

    @app.post("/api/predict")
    def predict():
        files = request.files.getlist("files") or request.files.getlist("file")
        if not files:
            return jsonify({"error": "send one or more images as multipart field 'files'"}), 400
        if len(files) > cfg.max_batch_files:
            return jsonify({"error": f"at most {cfg.max_batch_files} files per request"}), 400
        explain = request.form.get("explain", "1").lower() not in ("0", "false", "no")
        out = []
        for f in files:
            data = f.read(int(cfg.max_upload_mb * 1024 * 1024) + 1)
            name = Path(f.filename or "upload").name
            if len(data) > cfg.max_upload_mb * 1024 * 1024:
                out.append({"name": name, "ok": False, "error": f"file larger than {cfg.max_upload_mb:g} MB"})
                continue
            try:
                img = load_image(data, cfg, name=name)
            except ImageError as e:
                out.append({"name": name, "ok": False, "error": str(e)})
                continue
            try:
                pred = detector.predict_loaded(img, explain=explain)
                d = pred.to_dict()
                d["forensic_report"] = build_case_report(data, img, d, detector, name)
                if explain:
                    d["panels"] = {k: "data:image/png;base64," + base64.b64encode(v).decode()
                                   for k, v in render_panels(img, pred).items()}
            except Exception:
                app.logger.exception("Image analysis failed")
                out.append({
                    "name": name,
                    "ok": False,
                    "error": "image analysis failed; see the local server log for details",
                })
                continue
            out.append(d)
        _remember_results(app, out)
        return jsonify({"results": out})

    @app.get("/api/history")
    def history():
        if "user_id" not in session:
            return jsonify({"error": "Please sign in.", "code": "authentication_required"}), 401
        return jsonify({"history": history_store.list(session["user_id"])})

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({"error": "upload too large"}), 413

    return app


def create_gpu_app(detector, cfg: Config | None = None) -> Flask:
    cfg = cfg or Config()
    app = Flask(__name__, template_folder=str(_TEMPLATES))
    app.config["MAX_CONTENT_LENGTH"] = int(cfg.max_upload_mb * cfg.max_batch_files * 1024 * 1024)
    app.config["ANALYSIS_HISTORY"] = []
    history_store = HistoryStore()
    auth_store = AuthStore()
    _configure_auth(app, auth_store)
    _register_gemini_routes(app, history_store)
    _register_pdf_report_route(app, cfg)

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
        return response

    @app.get("/")
    def index():
        if "user_id" not in session:
            return render_template("landing.html")
        return render_template("index.html", info={"model_type": "gpu"}, max_mb=cfg.max_upload_mb,
                               max_files=cfg.max_batch_files, model_type="gpu")

    @app.get("/login")
    def login_page():
        if "user_id" in session:
            return redirect("/")
        return render_template("login.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/model")
    def model():
        return jsonify({"model_type": "gpu", "architecture": "ResNet-18",
                        "threshold": detector.threshold, "device": str(detector.device),
                        "supported_domain": SUPPORTED_DOMAIN, "scope_warning": SCOPE_WARNING})

    @app.post("/api/verify-provenance")
    def verify_provenance():
        """Optional external check for supported OpenAI provenance signals."""
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return jsonify({"error": "OpenAI provenance verification is not configured on this server."}), 503

        uploaded = request.files.get("file")
        if uploaded is None:
            return jsonify({"error": "Select one image to verify."}), 400

        max_bytes = min(int(cfg.max_upload_mb * 1024 * 1024), 50 * 1024 * 1024)
        data = uploaded.read(max_bytes + 1)
        if not data:
            return jsonify({"error": "The uploaded file is empty."}), 400
        if len(data) > max_bytes:
            return jsonify({"error": "The image exceeds the provenance verification size limit."}), 413

        name = Path(uploaded.filename or "upload").name
        try:
            loaded = load_image(data, cfg, name=name)
        except ImageError as exc:
            return jsonify({"error": f"Invalid image: {exc}"}), 400

        media = {
            "PNG": ("image/png", "upload.png"),
            "JPEG": ("image/jpeg", "upload.jpg"),
            "WEBP": ("image/webp", "upload.webp"),
        }.get(str(loaded.format or "").upper())
        if media is None:
            return jsonify({"error": "External provenance checks support PNG, JPEG, and WebP images only."}), 400

        mime_type, safe_filename = media
        boundary = "----aidetect-" + secrets.token_hex(16)
        prefix = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("ascii")
        body = prefix + data + f"\r\n--{boundary}--\r\n".encode("ascii")

        req = Request(
            "https://api.openai.com/v1/content_provenance_checks",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=30) as response:
                response_bytes = response.read(2 * 1024 * 1024 + 1)
        except HTTPError as exc:
            app.logger.warning("OpenAI provenance API returned HTTP %s", exc.code)
            messages = {
                401: "OpenAI rejected the API key. Check the key in this server's terminal.",
                403: "The API key does not have permission for this operation.",
                404: "Content Provenance API access is unavailable for this account or endpoint.",
                429: "OpenAI provenance verification rate limit or quota reached.",
            }
            message = messages.get(exc.code, f"OpenAI provenance verification failed (HTTP {exc.code}).")
            return jsonify({"error": message}), 502
        except (URLError, TimeoutError, OSError) as exc:
            app.logger.warning("OpenAI provenance request failed: %s", type(exc).__name__)
            return jsonify({"error": "Could not reach OpenAI provenance verification. Check the network and try again."}), 502

        if len(response_bytes) > 2 * 1024 * 1024:
            return jsonify({"error": "Unexpectedly large response from the provenance service."}), 502
        try:
            payload = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return jsonify({"error": "OpenAI returned an unreadable provenance response."}), 502

        raw_results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw_results, list):
            return jsonify({"error": "OpenAI returned an unexpected provenance response."}), 502

        clean_results = []
        for item in raw_results[:5]:
            if not isinstance(item, dict) or item.get("type") not in ("c2pa", "synthid"):
                continue
            outcome = item.get("outcome")
            if outcome not in ("detected", "not_detected"):
                continue
            signal = {"type": item["type"], "outcome": outcome}
            for field in ("validation_state", "issuer", "model", "generated_at"):
                value = item.get(field)
                if isinstance(value, str):
                    signal[field] = value[:160]
            clean_results.append(signal)

        confirmed = any(
            (r["type"] == "synthid" and r["outcome"] == "detected")
            or (
                r["type"] == "c2pa"
                and r["outcome"] == "detected"
                and r.get("validation_state") in ("trusted", "valid")
                and "openai" in r.get("issuer", "").lower()
            )
            for r in clean_results
        )
        if confirmed:
            status = "detected"
            summary = (
                "A supported OpenAI provenance signal was detected. This is evidence about origin, "
                "not a guarantee that all image content is unmodified."
            )
        elif clean_results and all(r["outcome"] == "not_detected" for r in clean_results):
            status = "not_detected"
            summary = (
                "No supported OpenAI provenance signal was detected. This does not prove the image "
                "is genuine or rule out generation by another AI system."
            )
        else:
            status = "inconclusive"
            summary = "The service did not return a conclusive supported provenance signal."

        return jsonify({
            "result": {
                "service": "OpenAI Content Provenance API",
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "summary": summary,
                "results": clean_results,
            }
        })

    @app.post("/api/predict")
    def predict():
        files = request.files.getlist("files") or request.files.getlist("file")
        if not files:
            return jsonify({"error": "send one or more images as multipart field 'files'"}), 400
        if len(files) > cfg.max_batch_files:
            return jsonify({"error": f"at most {cfg.max_batch_files} files per request"}), 400
        out = []
        for f in files:
            data = f.read(int(cfg.max_upload_mb * 1024 * 1024) + 1)
            name = Path(f.filename or "upload").name
            if len(data) > cfg.max_upload_mb * 1024 * 1024:
                out.append({"name": name, "ok": False, "error": f"file larger than {cfg.max_upload_mb:g} MB"})
                continue
            try:
                image = load_image(data, cfg, name=name)
            except ImageError as e:
                out.append({"name": name, "ok": False, "error": str(e)})
                continue
            try:
                from PIL import Image
                result = detector.predict_image(Image.fromarray(image.rgb), name)
                result["format"] = image.format
                result["forensic_report"] = build_case_report(
                    data, image, result, detector, name
                )
            except Exception:
                app.logger.exception("GPU image analysis failed")
                out.append({
                    "name": name,
                    "ok": False,
                    "error": "image analysis failed; see the local server log for details",
                })
                continue
            out.append(result)
        _remember_results(app, out)
        return jsonify({"results": out})

    @app.get("/api/history")
    def history():
        if "user_id" not in session:
            return jsonify({"error": "Please sign in.", "code": "authentication_required"}), 401
        return jsonify({"history": history_store.list(session["user_id"])})

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({"error": "upload too large"}), 413

    return app


def serve(model_dir: str, host: str, port: int) -> None:
    from types import SimpleNamespace
    app = create_app(SimpleNamespace(cfg=Config(), info=lambda: {}))
    print(f"aidetect UI on http://{host}:{port}  (Ctrl+C to stop)")
    app.run(host=host, port=port, threaded=True, debug=False)


def serve_gpu(model_dir: str, host: str, port: int, device: str) -> None:
    from types import SimpleNamespace
    app = create_app(SimpleNamespace(cfg=Config(), info=lambda: {}))
    print(f"aidetect GPU UI on http://{host}:{port}  (Ctrl+C to stop)")
    app.run(host=host, port=port, threaded=True, debug=False)
