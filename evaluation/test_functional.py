"""Offline end-to-end functional tests for the local website and API."""

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
WEB_DIR = PROJECT_ROOT / "web"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(WEB_DIR))

import server as server_module  # noqa: E402
import web_evidence as evidence_module  # noqa: E402
from banglish_normalizer import looks_banglish, normalize_banglish  # noqa: E402
from web_evidence import analyze_evidence, format_source_label  # noqa: E402


class FunctionalTests(unittest.TestCase):
    def test_local_server_rejects_an_untrusted_host_header(self) -> None:
        httpd = server_module.ReusableThreadingHTTPServer(
            ("127.0.0.1", 0), server_module.FakeNewsHandler
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
        request = urllib.request.Request(
            f"{base_url}/session-info",
            headers={"Host": f"attacker.example:{httpd.server_address[1]}"},
        )
        try:
            with self.assertRaises(urllib.error.HTTPError) as rejected_error:
                urllib.request.urlopen(request, timeout=10)
            self.assertEqual(rejected_error.exception.code, 403)
            payload = json.loads(rejected_error.exception.read().decode("utf-8"))
            rejected_error.exception.close()
            self.assertNotIn("shutdown_token", payload)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=10)

    def test_static_page_uses_private_security_headers(self) -> None:
        httpd = server_module.ReusableThreadingHTTPServer(
            ("127.0.0.1", 0), server_module.FakeNewsHandler
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base_url, timeout=10) as response:
                response.read()
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                self.assertEqual(
                    response.headers["Permissions-Policy"],
                    "camera=(), microphone=(), geolocation=()",
                )
                self.assertNotIn("Python", response.headers.get("Server", ""))
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=10)

    def test_session_info_rejects_a_malformed_page_id(self) -> None:
        httpd = server_module.ReusableThreadingHTTPServer(
            ("127.0.0.1", 0), server_module.FakeNewsHandler
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with self.assertRaises(urllib.error.HTTPError) as rejected_error:
                urllib.request.urlopen(
                    f"{base_url}/session-info?page_id=invalid%20page%20id",
                    timeout=10,
                )
            self.assertEqual(rejected_error.exception.code, 400)
            rejected_error.exception.close()
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=10)

    def test_matching_source_label_does_not_repeat_the_source_name(self) -> None:
        self.assertEqual(
            format_source_label("প্রথম আলো", "প্রথম আলো"),
            "প্রথম আলো",
        )
        self.assertEqual(
            format_source_label("নতুন শিক্ষানীতি ঘোষণা - প্রথম আলো", "প্রথম আলো"),
            "নতুন শিক্ষানীতি ঘোষণা - প্রথম আলো",
        )
        self.assertEqual(
            format_source_label("নতুন শিক্ষানীতি ঘোষণা", "প্রথম আলো"),
            "নতুন শিক্ষানীতি ঘোষণা - প্রথম আলো",
        )

    def test_banglish_trigger_and_conversion(self) -> None:
        text = "sorkar notun shikkha niti ghoshona koreche"
        self.assertTrue(looks_banglish(text))
        self.assertIn("সরকার", normalize_banglish(text))

    def test_four_way_evidence_category(self) -> None:
        evidence = analyze_evidence(
            "national cricket squad announced",
            [
                {
                    "source": "The Daily Star",
                    "source_url": "",
                    "title": "National cricket squad announced",
                    "snippet": "",
                    "url": "https://example.test/1",
                },
                {
                    "source": "bdnews24",
                    "source_url": "",
                    "title": "Board publishes national cricket squad",
                    "snippet": "",
                    "url": "https://example.test/2",
                },
            ],
            "real",
        )
        self.assertEqual(evidence["category"], "supporting")
        self.assertEqual(evidence["status"], "supported_by_sources")

    def test_news_feed_download_has_a_size_limit(self) -> None:
        class OversizedFeed:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size):
                return b"x" * size

        original_urlopen = evidence_module.urllib.request.urlopen
        evidence_module.urllib.request.urlopen = lambda *args, **kwargs: OversizedFeed()
        try:
            with self.assertRaisesRegex(ValueError, "too large"):
                evidence_module._read_google_news("test query")
        finally:
            evidence_module.urllib.request.urlopen = original_urlopen

    def test_connection_failure_returns_clear_internet_message(self) -> None:
        original_search = evidence_module._search_google_news
        evidence_module._search_google_news = lambda *args, **kwargs: (
            (_ for _ in ()).throw(urllib.error.URLError("offline"))
        )
        try:
            evidence = evidence_module.search_web_evidence(
                "সৌদি আরবের বিমানবন্দরে হামলায় নিহত ও আহত ২৬ বাংলাদেশি",
                "fake",
            )
        finally:
            evidence_module._search_google_news = original_search

        self.assertFalse(evidence["available"])
        self.assertEqual(evidence["failure_reason"], "connection_failed")
        self.assertEqual(evidence["status"], "web_search_unavailable")
        self.assertEqual(
            evidence["note"],
            (
                "Web evidence is unavailable. Check your internet connection "
                "and try again. The model prediction is still available."
            ),
        )

    def test_local_http_get_and_predict_post(self) -> None:
        original_search = server_module.search_web_evidence
        original_get_model = server_module.get_model

        class TestModel:
            def decision_function(self, texts):
                return [0.75 for _ in texts]

        server_module.get_model = lambda: TestModel()
        server_module.search_web_evidence = lambda text, prediction: {
            "available": True,
            "query": "offline functional test",
            "status": "not_enough_evidence",
            "category": "insufficient",
            "overall": "unverified",
            "note": "Controlled offline functional response.",
            "trusted_source_count": 0,
            "results": [],
        }
        httpd = server_module.ReusableThreadingHTTPServer(
            ("127.0.0.1", 0), server_module.FakeNewsHandler
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base_url, timeout=10) as response:
                page = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
            self.assertEqual(response.headers["X-Frame-Options"], "DENY")
            self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])
            self.assertIn("Fake News Detection", page)
            self.assertIn('role="dialog"', page)
            self.assertIn('id="evidenceStatus"', page)
            self.assertIn('id="sourceList"', page)
            self.assertIn('id="textCount"', page)
            self.assertIn('id="closeApp"', page)
            self.assertIn("Long text scrolls inside the box", page)

            with urllib.request.urlopen(
                f"{base_url}/project-info", timeout=10
            ) as response:
                project_info = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
            self.assertEqual(project_info["selected_model"], "Hybrid Linear SVM")
            self.assertIn("Hybrid Linear SVM", project_info["comparison_results"])

            body = json.dumps(
                {"text": "সরকার নতুন শিক্ষা নীতি ঘোষণা করেছে"},
                ensure_ascii=False,
            ).encode("utf-8")
            request = urllib.request.Request(
                f"{base_url}/predict",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
            self.assertIn(result["prediction"], {"fake", "real"})
            self.assertIsInstance(result["decision_score"], float)
            self.assertEqual(result["threshold"], 0.5)
            self.assertNotIn("web_evidence", result)
            self.assertIn("matching reports", result["message"])

            evidence_body = json.dumps(
                {
                    "text": "সরকার নতুন শিক্ষা নীতি ঘোষণা করেছে",
                    "prediction": result["prediction"],
                },
                ensure_ascii=False,
            ).encode("utf-8")
            evidence_request = urllib.request.Request(
                f"{base_url}/evidence",
                data=evidence_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(evidence_request, timeout=30) as response:
                evidence_result = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
            self.assertEqual(evidence_result["web_evidence"]["category"], "insufficient")

            malformed_request = urllib.request.Request(
                f"{base_url}/predict",
                data=b"{not-valid-json",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as malformed_error:
                urllib.request.urlopen(malformed_request, timeout=10)
            self.assertEqual(malformed_error.exception.code, 400)
            malformed_payload = json.loads(
                malformed_error.exception.read().decode("utf-8")
            )
            malformed_error.exception.close()
            self.assertEqual(malformed_payload["error"], "Invalid JSON request.")

            wrong_type_request = urllib.request.Request(
                f"{base_url}/predict",
                data=b"plain text",
                headers={"Content-Type": "text/plain"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as type_error:
                urllib.request.urlopen(wrong_type_request, timeout=10)
            self.assertEqual(type_error.exception.code, 415)
            type_error.exception.close()

            non_string_request = urllib.request.Request(
                f"{base_url}/predict",
                data=json.dumps({"text": {"unexpected": "object"}}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as text_error:
                urllib.request.urlopen(non_string_request, timeout=10)
            self.assertEqual(text_error.exception.code, 400)
            text_payload = json.loads(text_error.exception.read().decode("utf-8"))
            text_error.exception.close()
            self.assertEqual(text_payload["error"], "The text field must be a string.")
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=10)
            server_module.search_web_evidence = original_search
            server_module.get_model = original_get_model

    def test_confirmed_shutdown_stops_the_local_server(self) -> None:
        httpd = server_module.ReusableThreadingHTTPServer(
            ("127.0.0.1", 0), server_module.FakeNewsHandler
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            rejected_request = urllib.request.Request(
                f"{base_url}/shutdown",
                data=b"{}",
                headers={
                    "Content-Type": "application/json",
                    "Origin": base_url,
                    "X-Local-App-Action": "close",
                    "X-Shutdown-Token": "wrong-token",
                },
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as rejected_error:
                urllib.request.urlopen(rejected_request, timeout=10)
            self.assertEqual(rejected_error.exception.code, 403)
            rejected_error.exception.close()
            self.assertTrue(thread.is_alive())

            with urllib.request.urlopen(
                f"{base_url}/session-info", timeout=10
            ) as response:
                session_info = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
            self.assertTrue(session_info["shutdown_token"])

            request = urllib.request.Request(
                f"{base_url}/shutdown",
                data=b"{}",
                headers={
                    "Content-Type": "application/json",
                    "Origin": base_url,
                    "X-Local-App-Action": "close",
                    "X-Shutdown-Token": session_info["shutdown_token"],
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
            self.assertEqual(result["status"], "shutting_down")
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        finally:
            if thread.is_alive():
                httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=10)

    def test_browser_close_intent_stops_server_after_grace_period(self) -> None:
        httpd = server_module.ReusableThreadingHTTPServer(
            ("127.0.0.1", 0), server_module.FakeNewsHandler
        )
        httpd.close_grace_seconds = 0.05
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(
                f"{base_url}/session-info?page_id=page-one", timeout=10
            ) as response:
                session_info = json.loads(response.read().decode("utf-8"))

            request = urllib.request.Request(
                f"{base_url}/browser-close",
                data=json.dumps({"page_id": "page-one"}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Origin": base_url,
                    "X-Local-App-Action": "browser-close",
                    "X-Shutdown-Token": session_info["shutdown_token"],
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 202)
            self.assertEqual(result["status"], "close_pending")

            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        finally:
            if thread.is_alive():
                httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=10)

    def test_refresh_reconnect_cancels_browser_close_intent(self) -> None:
        httpd = server_module.ReusableThreadingHTTPServer(
            ("127.0.0.1", 0), server_module.FakeNewsHandler
        )
        httpd.close_grace_seconds = 0.1
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(
                f"{base_url}/session-info?page_id=page-before-refresh", timeout=10
            ) as response:
                session_info = json.loads(response.read().decode("utf-8"))

            request = urllib.request.Request(
                f"{base_url}/browser-close",
                data=json.dumps({"page_id": "page-before-refresh"}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Origin": base_url,
                    "X-Local-App-Action": "browser-close",
                    "X-Shutdown-Token": session_info["shutdown_token"],
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                self.assertEqual(response.status, 202)

            with urllib.request.urlopen(
                f"{base_url}/session-info?page_id=page-after-refresh", timeout=10
            ) as response:
                self.assertEqual(response.status, 200)

            time.sleep(0.25)
            self.assertTrue(thread.is_alive())
            with urllib.request.urlopen(base_url, timeout=10) as response:
                self.assertEqual(response.status, 200)
        finally:
            if thread.is_alive():
                httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=10)

    def test_model_prediction_returns_while_evidence_is_still_running(self) -> None:
        original_search = server_module.search_web_evidence
        original_get_model = server_module.get_model
        evidence_started = threading.Event()
        release_evidence = threading.Event()

        class TestModel:
            def decision_function(self, texts):
                return [0.75 for _ in texts]

        def slow_evidence(text, prediction):
            evidence_started.set()
            release_evidence.wait(timeout=10)
            return {
                "available": False,
                "query": "controlled test",
                "status": "web_search_unavailable",
                "category": "unavailable",
                "overall": "unverified",
                "note": "Controlled delayed response.",
                "trusted_source_count": 0,
                "results": [],
            }

        server_module.get_model = lambda: TestModel()
        server_module.search_web_evidence = slow_evidence
        httpd = server_module.ReusableThreadingHTTPServer(
            ("127.0.0.1", 0), server_module.FakeNewsHandler
        )
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
        evidence_result = {}

        def request_evidence() -> None:
            body = json.dumps(
                {"text": "পরীক্ষার সংবাদ", "prediction": "real"},
                ensure_ascii=False,
            ).encode("utf-8")
            request = urllib.request.Request(
                f"{base_url}/evidence",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                evidence_result.update(json.loads(response.read().decode("utf-8")))

        evidence_thread = threading.Thread(target=request_evidence, daemon=True)
        try:
            evidence_thread.start()
            self.assertTrue(evidence_started.wait(timeout=3))

            body = json.dumps(
                {"text": "পরীক্ষার সংবাদ"}, ensure_ascii=False
            ).encode("utf-8")
            request = urllib.request.Request(
                f"{base_url}/predict",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                prediction_result = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
            self.assertEqual(prediction_result["prediction"], "real")
            self.assertTrue(evidence_thread.is_alive())
        finally:
            release_evidence.set()
            evidence_thread.join(timeout=10)
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=10)
            server_module.search_web_evidence = original_search
            server_module.get_model = original_get_model

        self.assertEqual(
            evidence_result["web_evidence"]["note"],
            "Controlled delayed response.",
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
