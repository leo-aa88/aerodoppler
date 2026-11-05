"""The API allows the Next.js dev server origins only (localhost, any port) (the built UI is same-origin)."""

from fastapi.testclient import TestClient

from aerodoppler.api.app import create_app
from aerodoppler.config import Settings


def test_cors_allows_only_the_local_dev_server(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, offline=True), web_dir=tmp_path / "no-web")
    with TestClient(app) as c:
        for origin in ("http://localhost:3000", "http://127.0.0.1:3001"):
            r = c.get("/health", headers={"Origin": origin})
            assert r.status_code == 200
            assert r.headers.get("access-control-allow-origin") == origin
        for origin in ("http://evil.example", "http://localhost.evil.example:3000"):
            r = c.get("/health", headers={"Origin": origin})
            assert "access-control-allow-origin" not in r.headers
