import http.server
import threading
import unittest

from filament.config import MAX_SHELL_TIMEOUT_SECONDS, LLMConfig
from filament.llm import (
    MAX_ERROR_BODY_BYTES,
    MAX_RESPONSE_BYTES,
    AnthropicClient,
    OllamaClient,
    OpenAICompatibleClient,
    _read_body,
    _read_error_body,
    build_client,
    make_client,
)

SMALL_OK = b'{"choices": [{"message": {"content": "ok"}}]}'


class _FakeResponse:
    def __init__(self, data=b"", headers=None, max_chunk=None):
        self._data = data
        self._pos = 0
        self._max_chunk = max_chunk
        self.headers = headers or {}
        self.read_calls = []

    def read(self, size=-1):
        self.read_calls.append(size)
        end = len(self._data) if size is None or size < 0 else min(len(self._data), self._pos + size)
        if self._max_chunk is not None:
            end = min(end, self._pos + self._max_chunk)
        chunk = self._data[self._pos:end]
        self._pos = end
        return chunk


class _Server:
    def __init__(self, status=200, payload=b"", declare_length=False):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                if declare_length:
                    self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):  # type: ignore[override]
                pass

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.httpd.server_port}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class TestReadBody(unittest.TestCase):
    def test_body_under_limit(self):
        self.assertEqual(_read_body(_FakeResponse(b"abcd"), "test", limit=8), b"abcd")

    def test_body_exactly_at_limit(self):
        self.assertEqual(_read_body(_FakeResponse(b"12345678"), "test", limit=8), b"12345678")

    def test_body_over_limit_raises(self):
        with self.assertRaises(RuntimeError):
            _read_body(_FakeResponse(b"123456789"), "test", limit=8)

    def test_short_reads_still_reach_limit(self):
        response = _FakeResponse(b"123456789", max_chunk=3)
        with self.assertRaises(RuntimeError):
            _read_body(response, "test", limit=8)

    def test_declared_length_over_limit_is_rejected_before_read(self):
        response = _FakeResponse(b"", headers={"Content-Length": "9"})
        with self.assertRaises(RuntimeError):
            _read_body(response, "test", limit=8)
        self.assertEqual(response.read_calls, [])

    def test_invalid_content_length_is_ignored(self):
        response = _FakeResponse(b"abcd", headers={"Content-Length": "garbage"})
        self.assertEqual(_read_body(response, "test", limit=8), b"abcd")


class TestReadErrorBody(unittest.TestCase):
    def test_short_error_body_is_returned_whole(self):
        self.assertEqual(_read_error_body(_FakeResponse(b"boom")), "boom")  # type: ignore[arg-type]

    def test_oversized_error_body_is_truncated(self):
        response = _FakeResponse(b"E" * (MAX_ERROR_BODY_BYTES + 10))
        text = _read_error_body(response)  # type: ignore[arg-type]
        self.assertTrue(text.endswith(" [truncated]"))
        self.assertEqual(len(text), MAX_ERROR_BODY_BYTES + len(" [truncated]"))


class TestProviderResponseCap(unittest.TestCase):
    def setUp(self):
        self.server = _Server(payload=b"A" * (MAX_RESPONSE_BYTES + 64 * 1024))

    def tearDown(self):
        self.server.close()

    def test_all_providers_reject_oversized_response(self):
        clients = [
            OllamaClient(model="m", base_url=self.server.url, timeout=30),
            OpenAICompatibleClient(model="m", base_url=self.server.url, timeout=30),
            AnthropicClient(model="m", base_url=self.server.url, timeout=30),
        ]
        for client in clients:
            with self.subTest(client=type(client).__name__):
                with self.assertRaises(RuntimeError) as caught:
                    client.generate("hi")
                self.assertIn("exceeded", str(caught.exception))

    def test_declared_oversized_content_length_is_rejected(self):
        server = _Server(payload=b"A" * (MAX_RESPONSE_BYTES + 1), declare_length=True)
        self.addCleanup(server.close)
        client = OpenAICompatibleClient(model="m", base_url=server.url, timeout=30)
        with self.assertRaises(RuntimeError) as caught:
            client.generate("hi")
        self.assertIn("Content-Length", str(caught.exception))


class TestProviderSuccessPath(unittest.TestCase):
    def test_small_response_still_parses(self):
        server = _Server(payload=SMALL_OK)
        self.addCleanup(server.close)
        client = OpenAICompatibleClient(model="m", base_url=server.url, timeout=30)
        self.assertEqual(client.generate("hi"), "ok")


class TestHttpErrorBodyCap(unittest.TestCase):
    def test_error_body_is_truncated_in_message(self):
        server = _Server(status=500, payload=b"E" * (MAX_ERROR_BODY_BYTES + 1024), declare_length=True)
        self.addCleanup(server.close)
        client = OpenAICompatibleClient(model="m", base_url=server.url, timeout=30)
        with self.assertRaises(RuntimeError) as caught:
            client.generate("hi")
        message = str(caught.exception)
        self.assertIn("HTTP 500", message)
        self.assertIn("[truncated]", message)
        self.assertLess(len(message), MAX_ERROR_BODY_BYTES + 200)


class TestLLMTimeoutValidation(unittest.TestCase):
    def test_accepts_positive_whole_seconds(self):
        config = LLMConfig(timeout=30)
        config.validate()
        self.assertEqual(config.timeout, 30)
        self.assertIsInstance(config.timeout, int)

    def test_rejects_numeric_string(self):
        with self.assertRaises(ValueError):
            LLMConfig(timeout="120").validate()  # type: ignore[arg-type]

    def test_rejects_bool(self):
        with self.assertRaises(ValueError):
            LLMConfig(timeout=True).validate()

    def test_rejects_non_positive(self):
        for value in (0, -5, 0.5):
            with self.assertRaises(ValueError):
                LLMConfig(timeout=value).validate()  # type: ignore[arg-type]

    def test_rejects_fractional_seconds(self):
        with self.assertRaises(ValueError):
            LLMConfig(timeout=2.5).validate()  # type: ignore[arg-type]

    def test_rejects_beyond_upper_bound(self):
        with self.assertRaises(ValueError):
            LLMConfig(timeout=MAX_SHELL_TIMEOUT_SECONDS + 1).validate()

    def test_build_client_validates_timeout(self):
        with self.assertRaises(ValueError):
            build_client(LLMConfig(kind="mock", timeout="60"))  # type: ignore[arg-type]

    def test_make_client_validates_timeout(self):
        with self.assertRaises(ValueError):
            make_client(kind="openai", base_url="http://127.0.0.1:9", api_key="k", timeout="60")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            make_client(kind="openai", base_url="http://127.0.0.1:9", api_key="k", timeout=0.5)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
