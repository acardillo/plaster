from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from main import app
from tests.conftest import make_image_bytes

client = TestClient(app)


class TestHealth:
    def test_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestEmbed:
    def test_returns_embedding(self):
        fake = [0.1] * 512
        with patch("main.get_embedding", new_callable=AsyncMock, return_value=fake):
            resp = client.post(
                "/embed",
                files={"image": ("poster.png", make_image_bytes(), "image/png")},
            )
        assert resp.status_code == 200
        assert resp.json()["embedding"] == fake

    def test_missing_file_returns_422(self):
        resp = client.post("/embed")
        assert resp.status_code == 422
