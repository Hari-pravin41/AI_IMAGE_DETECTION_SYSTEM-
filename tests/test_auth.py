from types import SimpleNamespace
import uuid

from aidetect.auth_store import AuthStore
from aidetect.config import Config
from aidetect.server import create_app


def test_registration_login_logout_and_protected_history(tmp_path):
    app = create_app(SimpleNamespace(cfg=Config(), info=lambda: {}))
    app.config.update(TESTING=True)
    client = app.test_client()
    email = f"user-{uuid.uuid4().hex}@example.com"

    assert b"Evidence-led image review" in client.get("/").data
    assert b"Welcome back" in client.get("/login").data
    assert client.get("/api/history").status_code == 401

    response = client.post("/api/auth/register", json={
        "name": "Test User",
        "email": email,
        "password": "StrongPassword123!",
    })
    assert response.status_code == 201
    assert response.json["user"]["email"] == email
    assert b"AI Image Detection System" in client.get("/").data
    assert client.get("/api/history").status_code == 200

    client.post("/api/auth/logout")
    assert client.get("/api/history").status_code == 401
    login = client.post("/api/auth/login", json={
        "email": email,
        "password": "StrongPassword123!",
    })
    assert login.status_code == 200


def test_auth_store_password_is_hashed_and_email_is_normalized(tmp_path):
    store = AuthStore(tmp_path / "accounts.db")
    user = store.create("Professional User", "User@Example.com", "StrongPassword123!")
    stored = store.authenticate("user@example.com", "StrongPassword123!")

    assert stored == user
    assert store.authenticate("user@example.com", "wrong-password") is None


def test_password_reset_token_is_single_use(tmp_path):
    store = AuthStore(tmp_path / "accounts.db")
    store.create("Reset User", "reset@example.com", "OldPassword123!")

    token = store.create_reset_token("reset@example.com")
    assert token
    assert store.reset_password(token, "NewPassword123!")
    assert store.authenticate("reset@example.com", "NewPassword123!")
    assert store.authenticate("reset@example.com", "OldPassword123!") is None
    assert store.reset_password(token, "AnotherPassword123!") is False


def test_registration_requires_matching_confirmation(tmp_path):
    app = create_app(SimpleNamespace(cfg=Config(), info=lambda: {}))
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.post("/api/auth/register", json={
        "name": "Confirmation User",
        "email": f"confirm-{uuid.uuid4().hex}@example.com",
        "password": "CaseSensitive123!",
        "confirmation_password": "casesensitive123!",
    })

    assert response.status_code == 400
    assert response.json["code"] == "password_mismatch"


def test_email_verification_token_is_single_use(tmp_path):
    store = AuthStore(tmp_path / "accounts.db")
    user = store.create("Verified User", "verified@example.com", "StrongPassword123!")
    token = store.create_verification_token(user["id"])

    assert store.verify_email(token) is True
    assert store.verify_email(token) is False
    assert store.get_by_id(user["id"])["email_verified"] is True
