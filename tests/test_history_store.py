import io

from PIL import Image

from aidetect.history_store import HistoryStore


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(output, "PNG")
    return output.getvalue()


def test_summary_includes_assessment_distribution_and_review_counts(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    common = {
        "analysis_timestamp": "2026-11-01T12:00:00+00:00",
        "image_data": {},
    }
    ai = store.add({**common, "filename": "ai.png", "assessment": "Likely AI-generated"}, image_bytes(), "image/png", "user-1")
    real = store.add({**common, "filename": "real.png", "assessment": "Likely real"}, image_bytes(), "image/png", "user-1")

    assert store.set_review(ai["analysis_id"], "user-1", "confirmed")
    assert store.set_review(real["analysis_id"], "user-1", "disputed")

    summary = store.summary("user-1")
    assert summary["total"] == 2
    assert summary["ai_generated"] == 1
    assert summary["real"] == 1
    assert summary["ai_percentage"] == 50.0
    assert summary["real_percentage"] == 50.0
    assert summary["review_total"] == 2
    assert summary["review_confirmed"] == 1
    assert summary["review_disputed"] == 1
