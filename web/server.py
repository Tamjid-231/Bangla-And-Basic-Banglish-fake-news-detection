import importlib.util
import json
import re
import secrets
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from joblib import load


# ==================== PROJECT SETTINGS: START =======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "banfakenews_tfidf_svm.joblib"
PROJECT_INFO_PATH = PROJECT_ROOT / "evaluation" / "retraining" / "model_metrics.json"
WEB_DIR = PROJECT_ROOT / "web"
MAX_REQUEST_BYTES = 200_000
MAX_PAGE_ID_LENGTH = 128
PAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
MODEL = None
MODEL_LOCK = threading.Lock()
SHUTDOWN_TOKEN = secrets.token_urlsafe(32)
# ===================== PROJECT SETTINGS: END ========================

sys.path.insert(0, str(SRC_DIR))
from preprocess import clean_text  # noqa: E402
from model_decision import DEFAULT_DECISION_THRESHOLD, label_from_score  # noqa: E402


EVIDENCE_MODULE_PATH = SRC_DIR / "web_evidence.py"
EVIDENCE_SPEC = importlib.util.spec_from_file_location(
    "capstone_web_evidence_runtime",
    EVIDENCE_MODULE_PATH,
)
if EVIDENCE_SPEC is None or EVIDENCE_SPEC.loader is None:
    raise ImportError(f"Could not load evidence engine: {EVIDENCE_MODULE_PATH}")
EVIDENCE_MODULE = importlib.util.module_from_spec(EVIDENCE_SPEC)
EVIDENCE_SPEC.loader.exec_module(EVIDENCE_MODULE)
search_web_evidence = EVIDENCE_MODULE.search_web_evidence


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    close_grace_seconds = 1.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_page_id = ""
        self.page_lock = threading.Lock()

    def register_page(self, page_id: str) -> None:
        with self.page_lock:
            self.active_page_id = page_id

    def schedule_browser_close(self, page_id: str) -> None:
        def stop_if_page_did_not_return() -> None:
            with self.page_lock:
                should_stop = self.active_page_id == page_id
            if should_stop:
                self.shutdown()

        timer = threading.Timer(self.close_grace_seconds, stop_if_page_did_not_return)
        timer.daemon = True
        timer.start()


def get_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    with MODEL_LOCK:
        if MODEL is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
            MODEL = load(MODEL_PATH)
    return MODEL


def warm_model() -> None:
    try:
        get_model()
    except Exception as error:
        print(f"Model warm-up failed: {error}", file=sys.stderr, flush=True)


class FakeNewsHandler(SimpleHTTPRequestHandler):
    server_version = "BanglaFakeNews"
    sys_version = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def is_allowed_host(self) -> bool:
        port = self.server.server_address[1]
        host = self.headers.get("Host", "").strip().lower()
        return host in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def reject_untrusted_host(self) -> bool:
        if self.is_allowed_host():
            return False
        self.send_json(403, {"error": "This local request is not allowed."})
        return True

    @staticmethod
    def is_valid_page_id(page_id: object) -> bool:
        return (
            isinstance(page_id, str)
            and 1 <= len(page_id) <= MAX_PAGE_ID_LENGTH
            and PAGE_ID_PATTERN.fullmatch(page_id) is not None
        )

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=()",
        )
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'",
        )
        super().end_headers()

    def send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.reject_untrusted_host():
            return
        requested_url = urlsplit(self.path)
        if requested_url.path == "/session-info":
            page_id = parse_qs(requested_url.query).get("page_id", [""])[0].strip()
            if page_id and not self.is_valid_page_id(page_id):
                self.send_json(400, {"error": "The page session is not valid."})
                return
            if page_id:
                self.server.register_page(page_id)
            self.send_json(200, {"shutdown_token": SHUTDOWN_TOKEN})
            return
        if self.path == "/project-info":
            try:
                with PROJECT_INFO_PATH.open("r", encoding="utf-8") as report:
                    self.send_json(200, json.load(report))
            except (OSError, json.JSONDecodeError) as error:
                self.log_error("Could not read project information: %s", error)
                self.send_json(500, {"error": "Evaluation report unavailable."})
            return
        super().do_GET()

    def read_json_request(self) -> dict | None:
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            self.send_json(415, {"error": "Content-Type must be application/json."})
            return None

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"error": "Invalid Content-Length header."})
            return None

        if content_length < 0:
            self.send_json(400, {"error": "Invalid Content-Length header."})
            return None
        if content_length > MAX_REQUEST_BYTES:
            self.send_json(413, {"error": "The submitted text is too large."})
            return None

        try:
            raw_body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(raw_body or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, {"error": "Invalid JSON request."})
            return None

        if not isinstance(data, dict):
            self.send_json(400, {"error": "The request must be a JSON object."})
            return None
        return data

    def read_news_text(self, data: dict) -> tuple[str, str] | None:
        submitted_text = data.get("text", "")
        if not isinstance(submitted_text, str):
            self.send_json(400, {"error": "The text field must be a string."})
            return None
        text = submitted_text.strip()
        if not text:
            self.send_json(400, {"error": "Please enter news text first."})
            return None

        cleaned_text = clean_text(text)
        if not cleaned_text:
            self.send_json(
                400,
                {
                    "error": (
                        "The text is empty after cleaning. "
                        "Please enter valid Bangla news text."
                    )
                },
            )
            return None
        return text, cleaned_text

    def do_POST(self) -> None:
        if self.reject_untrusted_host():
            return
        if self.path == "/browser-close":
            host = self.headers.get("Host", "")
            supplied_token = self.headers.get("X-Shutdown-Token", "")
            is_authorized = (
                self.client_address[0] in {"127.0.0.1", "::1"}
                and self.headers.get("Origin") == f"http://{host}"
                and self.headers.get("X-Local-App-Action") == "browser-close"
                and secrets.compare_digest(supplied_token, SHUTDOWN_TOKEN)
            )
            if not is_authorized:
                self.send_json(403, {"error": "Browser close confirmation is required."})
                return
            data = self.read_json_request()
            if data is None:
                return
            page_id = data.get("page_id", "")
            if not self.is_valid_page_id(page_id):
                self.send_json(400, {"error": "A valid page session is required."})
                return
            self.server.schedule_browser_close(page_id)
            self.send_json(202, {"status": "close_pending"})
            return

        if self.path == "/shutdown":
            host = self.headers.get("Host", "")
            supplied_token = self.headers.get("X-Shutdown-Token", "")
            is_authorized = (
                self.client_address[0] in {"127.0.0.1", "::1"}
                and self.headers.get("Origin") == f"http://{host}"
                and self.headers.get("X-Local-App-Action") == "close"
                and secrets.compare_digest(supplied_token, SHUTDOWN_TOKEN)
            )
            if not is_authorized:
                self.send_json(403, {"error": "Shutdown confirmation is required."})
                return
            self.send_json(200, {"status": "shutting_down"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        if self.path not in {"/predict", "/evidence"}:
            self.send_json(404, {"error": "Not found"})
            return

        try:
            data = self.read_json_request()
            if data is None:
                return
            text_pair = self.read_news_text(data)
            if text_pair is None:
                return
            text, cleaned_text = text_pair

            if self.path == "/evidence":
                prediction = str(data.get("prediction", "")).lower()
                if prediction not in {"fake", "real"}:
                    self.send_json(400, {"error": "A valid model prediction is required."})
                    return
                self.send_json(
                    200,
                    {"web_evidence": search_web_evidence(text, prediction)},
                )
                return

            model = get_model()
            score = float(model.decision_function([cleaned_text])[0])
            prediction = label_from_score(score, DEFAULT_DECISION_THRESHOLD)

            self.send_json(
                200,
                {
                    "prediction": str(prediction).lower(),
                    "decision_score": score,
                    "threshold": DEFAULT_DECISION_THRESHOLD,
                    "message": (
                        "The model checks language patterns. Web evidence checks for "
                        "matching reports; neither replaces a complete manual fact-check."
                    ),
                },
            )
        except Exception as error:
            self.log_error("Prediction failed: %s", error)
            self.send_json(500, {"error": "Prediction could not be completed."})


def main() -> None:
    host = "127.0.0.1"
    port = 8502
    server = ReusableThreadingHTTPServer((host, port), FakeNewsHandler)
    print(f"Fake News Detection website running at http://{host}:{port}")
    print("Press CTRL+C to stop the server.")
    threading.Thread(target=warm_model, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
