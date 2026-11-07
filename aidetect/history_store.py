"""Lightweight persistent analysis history stored by the Python backend."""
from __future__ import annotations

import json
import sqlite3
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class HistoryStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else Path(__file__).resolve().parents[1] / "data" / "analysis_history.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    assessment TEXT NOT NULL,
                    analysis_timestamp TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    image_blob BLOB NOT NULL,
                    image_mime TEXT NOT NULL
                )"""
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(analyses)")}
            if "user_id" not in columns:
                connection.execute("ALTER TABLE analyses ADD COLUMN user_id TEXT")
            if "review_status" not in columns:
                connection.execute("ALTER TABLE analyses ADD COLUMN review_status TEXT")
                columns.add("review_status")
            if "review_updated_at" not in columns:
                connection.execute("ALTER TABLE analyses ADD COLUMN review_updated_at TEXT")
                columns.add("review_updated_at")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS report_shares (
                    token_hash TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def add(self, result: dict[str, Any], image_bytes: bytes, mime_type: str, user_id: str | None = None) -> dict[str, Any]:
        analysis_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO analyses "
                "(id, filename, assessment, analysis_timestamp, result_json, image_blob, image_mime, user_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    analysis_id,
                    result["filename"],
                    result["assessment"],
                    result["analysis_timestamp"],
                    json.dumps(result, ensure_ascii=False),
                    image_bytes,
                    mime_type,
                    user_id,
                ),
            )
        return self.get(analysis_id, user_id)

    def list(self, user_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            params: tuple[Any, ...] = ()
            owner_sql = ""
            if user_id is not None:
                owner_sql = " WHERE user_id = ?"
                params = (user_id,)
            rows = connection.execute(
                "SELECT id, filename, assessment, analysis_timestamp, result_json, review_status, review_updated_at FROM analyses "
                + owner_sql
                + " ORDER BY analysis_timestamp DESC LIMIT 200",
                params,
            ).fetchall()
        return [self._public(row) for row in rows]

    def get(self, analysis_id: str, user_id: str | None = None) -> dict[str, Any]:
        with self._connect() as connection:
            params: tuple[Any, ...] = (analysis_id,)
            owner_sql = ""
            if user_id is not None:
                owner_sql = " AND user_id = ?"
                params = (analysis_id, user_id)
            row = connection.execute(
                "SELECT id, filename, assessment, analysis_timestamp, result_json, review_status, review_updated_at FROM analyses WHERE id = ?" + owner_sql,
                params,
            ).fetchone()
        if row is None:
            raise KeyError(analysis_id)
        return self._public(row)

    def image(self, analysis_id: str, user_id: str | None = None) -> tuple[bytes, str] | None:
        with self._connect() as connection:
            params: tuple[Any, ...] = (analysis_id,)
            owner_sql = ""
            if user_id is not None:
                owner_sql = " AND user_id = ?"
                params = (analysis_id, user_id)
            row = connection.execute(
                "SELECT image_blob, image_mime FROM analyses WHERE id = ?" + owner_sql,
                params,
            ).fetchone()
        if row is None:
            return None
        return bytes(row["image_blob"]), row["image_mime"]

    def delete(self, analysis_id: str, user_id: str | None = None) -> bool:
        with self._connect() as connection:
            params: tuple[Any, ...] = (analysis_id,)
            owner_sql = ""
            if user_id is not None:
                owner_sql = " AND user_id = ?"
                params = (analysis_id, user_id)
            cursor = connection.execute("DELETE FROM analyses WHERE id = ?" + owner_sql, params)
            return cursor.rowcount > 0

    def set_review(self, analysis_id: str, user_id: str, status: str) -> bool:
        if status not in {"confirmed", "disputed"}:
            return False
        from datetime import datetime, timezone
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE analyses SET review_status = ?, review_updated_at = ? "
                "WHERE id = ? AND user_id = ?",
                (status, datetime.now(timezone.utc).isoformat(), analysis_id, user_id),
            )
            return cursor.rowcount > 0

    def summary(self, user_id: str | None = None) -> dict[str, int | list[dict[str, int | str]]]:
        with self._connect() as connection:
            owner_sql = " WHERE user_id = ?" if user_id is not None else ""
            params = (user_id,) if user_id is not None else ()
            rows = connection.execute(
                "SELECT assessment, analysis_timestamp, review_status FROM analyses" + owner_sql,
                params,
            ).fetchall()
        counts = {"Likely AI-generated": 0, "Likely real": 0}
        confirmed = 0
        disputed = 0
        trend = {}
        today = datetime.now(timezone.utc).date()
        for offset in range(6, -1, -1):
            day = (today - timedelta(days=offset)).isoformat()
            trend[day] = {"date": day, "total": 0, "ai_generated": 0, "real": 0}
        for row in rows:
            assessment = str(row["assessment"])
            if assessment == "Inconclusive" or assessment == "Uncertain":
                assessment = "Likely real"
            if assessment in counts:
                counts[assessment] += 1
            if row["review_status"] == "confirmed":
                confirmed += 1
            elif row["review_status"] == "disputed":
                disputed += 1
            try:
                day = datetime.fromisoformat(str(row["analysis_timestamp"]).replace("Z", "+00:00")).date().isoformat()
            except (TypeError, ValueError):
                continue
            if day in trend:
                trend[day]["total"] += 1
                if assessment == "Likely AI-generated":
                    trend[day]["ai_generated"] += 1
                elif assessment == "Likely real":
                    trend[day]["real"] += 1
        total = len(rows)
        ai_total = counts["Likely AI-generated"]
        real_total = counts["Likely real"]
        return {
            "total": total,
            "ai_generated": ai_total,
            "real": real_total,
            "ai_percentage": round(ai_total * 100 / total, 1) if total else 0,
            "real_percentage": round(real_total * 100 / total, 1) if total else 0,
            "review_confirmed": confirmed,
            "review_disputed": disputed,
            "review_total": confirmed + disputed,
            "trend": list(trend.values()),
        }

    def create_share(self, analysis_id: str, user_id: str, lifetime_hours: int = 168) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id)
            ).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex + secrets.token_urlsafe(24)
            from datetime import datetime, timedelta, timezone
            expires = datetime.now(timezone.utc) + timedelta(hours=lifetime_hours)
            connection.execute(
                "INSERT INTO report_shares (token_hash, analysis_id, created_at, expires_at, revoked_at) "
                "VALUES (?, ?, datetime('now'), ?, NULL)",
                (hashlib.sha256(token.encode()).hexdigest(), analysis_id, expires.isoformat()),
            )
        return token

    def revoke_share(self, token: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE report_shares SET revoked_at = datetime('now') WHERE token_hash = ? AND revoked_at IS NULL",
                (hashlib.sha256(token.encode()).hexdigest(),),
            )
            return cursor.rowcount > 0

    def shared_analysis(self, token: str) -> str | None:
        from datetime import datetime, timezone
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT analysis_id, expires_at FROM report_shares WHERE token_hash = ? AND revoked_at IS NULL",
                (token_hash,),
            ).fetchone()
        if row is None or datetime.now(timezone.utc) >= datetime.fromisoformat(row["expires_at"]):
            return None
        return str(row["analysis_id"])

    @staticmethod
    def _public(row: sqlite3.Row) -> dict[str, Any]:
        result = json.loads(row["result_json"])
        result["analysis_id"] = row["id"]
        result["review_status"] = row["review_status"] if "review_status" in row.keys() else None
        result["review_updated_at"] = row["review_updated_at"] if "review_updated_at" in row.keys() else None
        if result.get("assessment") in ("Inconclusive", "Uncertain"):
            result["assessment"] = "Likely real"
            result["label"] = "real"
        return result
