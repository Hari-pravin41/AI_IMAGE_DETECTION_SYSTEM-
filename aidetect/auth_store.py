"""Local account storage with password hashing."""
from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
import hashlib
import secrets
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else Path(__file__).resolve().parents[1] / "data" / "accounts.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    email_verified INTEGER NOT NULL DEFAULT 0
                )"""
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(users)")}
            if "email_verified" not in columns:
                connection.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def create(self, name: str, email: str, password: str) -> dict[str, str]:
        name = name.strip()
        email = email.strip().lower()
        if len(name) < 2:
            raise ValueError("Enter your full name.")
        if not EMAIL_PATTERN.fullmatch(email):
            raise ValueError("Enter a valid email address.")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters.")
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
                raise ValueError("An account with this email already exists.")
            user_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO users (id, name, email, password_hash, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
                (user_id, name, email, generate_password_hash(password)),
            )
        return self.get_by_id(user_id)

    def create_reset_token(self, email: str, lifetime_minutes: int = 30) -> str | None:
        email = email.strip().lower()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM users WHERE email = ?", (email,)
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """CREATE TABLE IF NOT EXISTS password_resets (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT UNIQUE NOT NULL,
                    expires_at TEXT NOT NULL,
                    used_at TEXT
                )"""
            )
            connection.execute(
                "UPDATE password_resets SET used_at = datetime('now') WHERE user_id = ? AND used_at IS NULL",
                (row["id"],),
            )
            token = secrets.token_urlsafe(32)
            expires = datetime.now(timezone.utc) + timedelta(minutes=lifetime_minutes)
            connection.execute(
                "INSERT INTO password_resets (id, user_id, token_hash, expires_at, used_at) VALUES (?, ?, ?, ?, NULL)",
                (uuid.uuid4().hex, row["id"], hashlib.sha256(token.encode()).hexdigest(), expires.isoformat()),
            )
        return token

    def reset_password(self, token: str, new_password: str) -> bool:
        if len(new_password) < 8:
            raise ValueError("Password must be at least 8 characters.")
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS password_resets (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT UNIQUE NOT NULL,
                    expires_at TEXT NOT NULL,
                    used_at TEXT
                )"""
            )
            row = connection.execute(
                """SELECT user_id, expires_at FROM password_resets
                   WHERE token_hash = ? AND used_at IS NULL""",
                (token_hash,),
            ).fetchone()
            if row is None:
                return False
            expires = datetime.fromisoformat(row["expires_at"])
            if datetime.now(timezone.utc) >= expires:
                return False
            connection.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (generate_password_hash(new_password), row["user_id"]),
            )
            connection.execute(
                "UPDATE password_resets SET used_at = datetime('now') WHERE token_hash = ?",
                (token_hash,),
            )
        return True

    def reset_password_for_email(self, email: str, new_password: str) -> bool:
        if len(new_password) < 8:
            raise ValueError("Password must be at least 8 characters.")
        email = email.strip().lower()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM users WHERE email = ?", (email,)
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (generate_password_hash(new_password), row["id"]),
            )
        return True

    def create_verification_token(self, user_id: str, lifetime_minutes: int = 60) -> str:
        token = secrets.token_urlsafe(32)
        expires = datetime.now(timezone.utc) + timedelta(minutes=lifetime_minutes)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS email_verifications (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT UNIQUE NOT NULL,
                    expires_at TEXT NOT NULL,
                    used_at TEXT
                )"""
            )
            connection.execute(
                "UPDATE email_verifications SET used_at = datetime('now') WHERE user_id = ? AND used_at IS NULL",
                (user_id,),
            )
            connection.execute(
                "INSERT INTO email_verifications (id,user_id,token_hash,expires_at,used_at) VALUES (?, ?, ?, ?, NULL)",
                (uuid.uuid4().hex, user_id, hashlib.sha256(token.encode()).hexdigest(), expires.isoformat()),
            )
        return token

    def verify_email(self, token: str) -> bool:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as connection:
            row = connection.execute(
                """SELECT v.user_id, v.expires_at FROM email_verifications v
                   WHERE v.token_hash = ? AND v.used_at IS NULL""",
                (token_hash,),
            ).fetchone()
            if row is None:
                return False
            if datetime.now(timezone.utc) >= datetime.fromisoformat(row["expires_at"]):
                return False
            connection.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (row["user_id"],))
            connection.execute(
                "UPDATE email_verifications SET used_at = datetime('now') WHERE token_hash = ?",
                (token_hash,),
            )
        return True

    def authenticate(self, email: str, password: str) -> dict[str, str] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, name, email, password_hash, email_verified FROM users WHERE email = ?",
                (email.strip().lower(),),
            ).fetchone()
        if row is None or not check_password_hash(row["password_hash"], password):
            return None
        return {
            "id": row["id"], "name": row["name"], "email": row["email"],
            "email_verified": bool(row["email_verified"]) if "email_verified" in row.keys() else False,
        }

    def get_by_id(self, user_id: str) -> dict[str, str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, name, email, email_verified FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        if row is None:
            raise KeyError(user_id)
        return {
            "id": row["id"], "name": row["name"], "email": row["email"],
            "email_verified": bool(row["email_verified"]),
        }
