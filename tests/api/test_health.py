from serving.api import create_app
from serving.runtime import UnavailableBackend
from tests.test_serving import FakeBackend, request, settings


def test_health_ready_and_metrics_aliases():
    app=create_app(FakeBackend(),settings=settings())
    assert request(app,"GET","/health").status_code == 200
    assert request(app,"GET","/ready").status_code == 200
    metrics=request(app,"GET","/metrics")
    assert metrics.status_code == 200
    assert "total_requests" in metrics.json()


def test_readiness_reports_unavailable():
    app=create_app(UnavailableBackend(),settings=settings())
    assert request(app,"GET","/health").json()["status"] == "ok"
    assert request(app,"GET","/ready").status_code == 503
