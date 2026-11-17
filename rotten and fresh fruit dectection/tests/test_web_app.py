"""Unit tests for live camera web app endpoints."""

import json
import urllib.request
import threading
import time
import base64
import numpy as np
import cv2
import pytest

from web_app import PORT, ThreadingHTTPServer, LiveCameraAPIHandler


@pytest.fixture(scope="module")
def web_server():
    """Start local web server on a test port in a background thread."""
    test_port = 8092
    server = ThreadingHTTPServer(("127.0.0.1", test_port), LiveCameraAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.5)
    yield f"http://127.0.0.1:{test_port}"
    server.shutdown()
    server.server_close()


def test_live_web_gui_index_page(web_server):
    """Verify that the home page HTML loads successfully with live camera components."""
    resp = urllib.request.urlopen(f"{web_server}/")
    assert resp.status == 200
    html = resp.read().decode("utf-8")
    assert "Fruit Freshness AI Detector" in html
    assert "videoFeed" in html
    assert "overlayCanvas" in html
    assert "Log Current Item" in html


def test_live_web_gui_status_api(web_server):
    """Verify status endpoint returns model info."""
    resp = urllib.request.urlopen(f"{web_server}/api/status")
    assert resp.status == 200
    data = json.loads(resp.read().decode("utf-8"))
    assert data["status"] == "ready"
    assert "MobileNetV2" in data["model"]


def test_live_web_gui_predict_api(web_server):
    """Verify prediction endpoint returns live prediction."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

    req_data = json.dumps({
        "image": b64,
        "threshold": 0.80,
        "filename": "camera_frame_test.jpg"
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{web_server}/api/predict",
        data=req_data,
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req)
    assert resp.status == 200
    res_json = json.loads(resp.read().decode("utf-8"))
    assert res_json["status"] == "success"
    assert res_json["verdict"] in ["FRESH", "ROTTEN", "UNCERTAIN"]
    assert "fresh_score" in res_json
    assert "rotten_score" in res_json
