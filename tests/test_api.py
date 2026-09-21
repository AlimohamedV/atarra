"""API tests.

Only the endpoints that work offline are covered here. Anything that reads the
archive belongs in ``test_live.py``, marked ``network``, so a flaky catalogue never
breaks the default suite.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from atarra.api.main import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


class TestHealth:
    def test_health_reports_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_exposes_the_reflectance_convention(self, client):
        """The setting most likely to silently invalidate every index."""
        reflectance = client.get("/health").json()["reflectance"]
        assert reflectance["mode"] == "dn_scale"
        assert reflectance["offset"] == 0.0

    def test_health_reports_configuration(self, client):
        payload = client.get("/health").json()
        assert payload["collection"]
        assert len(payload["bands_8"]) == 8
        assert len(payload["bands_rgb"]) == 3
        assert "cache" in payload


class TestAreas:
    def test_lists_configured_areas(self, client):
        payload = client.get("/areas").json()
        assert payload["count"] == 3
        keys = {area["key"] for area in payload["areas"]}
        assert keys == {"burullus", "manzala", "kafr_elsheikh_canal"}

    def test_each_area_has_a_map_ready_geometry(self, client):
        for area in client.get("/areas").json()["areas"]:
            geometry = area["geometry"]
            assert geometry["type"] == "Polygon"
            ring = geometry["coordinates"][0]
            assert ring[0] == ring[-1]
            assert len(area["bbox_wgs84"]) == 4


class TestLegend:
    def test_legend_matches_the_renderer(self, client):
        payload = client.get("/legend/ndvi").json()
        assert payload["index"] == "ndvi"
        assert payload["stops"]
        for stop in payload["stops"]:
            assert stop["color"].startswith("#")
            assert payload["vmin"] <= stop["value"] <= payload["vmax"]

    def test_unknown_index_is_404(self, client):
        assert client.get("/legend/not_an_index").status_code == 404

    def test_all_indices_have_legends(self, client):
        for index in ("ndvi", "ndwi", "ndre", "ndmi"):
            assert client.get(f"/legend/{index}").status_code == 200


class TestValidation:
    def test_bad_date_is_rejected(self, client):
        response = client.get("/areas/burullus/summary", params={"date": "not-a-date"})
        assert response.status_code == 400
        assert "YYYY-MM-DD" in response.json()["detail"]

    def test_bad_index_is_rejected(self, client):
        response = client.get("/areas/burullus/summary", params={"indices": "ndvi,bogus"})
        assert response.status_code == 404

    def test_out_of_range_gsd_is_rejected(self, client):
        response = client.get("/areas/burullus/render.png", params={"gsd": 99999})
        assert response.status_code == 422  # FastAPI query validation

    def test_unknown_area_is_reported(self, client):
        response = client.get("/areas/atlantis/summary")
        assert response.status_code in {404, 500, 502}


class TestCacheEndpoint:
    def test_cache_stats_shape(self, client):
        payload = client.get("/cache").json()
        assert "bytes" in payload
        assert "max_bytes" in payload
        assert payload["max_bytes"] > 0
