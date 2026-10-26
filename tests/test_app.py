import io
import os

from conftest import png_bytes


def upload(client, name, data, **kwargs):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data", **kwargs)


def test_health(client):
    body = client.get("/health").get_json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["database_connected"] is True


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Verify Synthetic Media Integrity" in r.data


def test_predict_json(client):
    r = upload(client, "red.png", png_bytes(), query_string={"format": "json"})
    assert r.status_code == 200
    body = r.get_json()
    assert 0.0 <= body["raw_score"] <= 1.0
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["filename"] == "red.png"


def test_predict_html_has_caveat(client):
    r = upload(client, "red.png", png_bytes())
    assert r.status_code == 200
    assert b"Mandatory AI advisory" in r.data


def test_rejects_wrong_extension(client):
    r = upload(client, "doc.pdf", b"%PDF-1.4 fake")
    assert r.status_code == 415


def test_rejects_wrong_content_with_image_extension(client):
    r = upload(client, "fake.png", b"this is not an image at all")
    assert r.status_code == 400


def test_rejects_truncated_image(client):
    r = upload(client, "cut.png", png_bytes((300, 300))[:80])
    assert r.status_code == 400


def test_rejects_oversized(client):
    r = upload(client, "big.png", os.urandom(2 * 1024 * 1024))
    assert r.status_code == 413


def test_missing_file(client):
    r = client.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_filename_is_sanitized(client):
    r = upload(client, "../../etc/pass wd.png", png_bytes(), query_string={"format": "json"})
    assert r.status_code == 200
    assert "/" not in r.get_json()["filename"]


def test_history_logs_predictions(client):
    upload(client, "a.png", png_bytes())
    upload(client, "b.png", png_bytes(color=(10, 10, 200)))
    rows = client.get("/history?format=json").get_json()
    assert [r["filename"] for r in rows] == ["b.png", "a.png"]
    page = client.get("/history")
    assert page.status_code == 200 and b"a.png" in page.data
