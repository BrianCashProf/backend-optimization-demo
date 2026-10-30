"""Matching batch order pricing workloads for Locust (select ONE user class per run).

python -m locust -f locustfile.py RestUser --host http://127.0.0.1:8001
python -m locust -f locustfile.py WebSocketUser --host http://127.0.0.1:8003

Locust patches socket I/O with gevent before loading this module. websocket-client
uses that socket I/O, so each persistent connection can yield to other users.
"""

import json
import math
from time import perf_counter, time
from urllib.parse import urlsplit, urlunsplit

import websocket
from locust import HttpUser, User, constant, task
from locust.exception import StopUser


def pricing_for(revision):
    # Identical pricing scenarios per virtual user, for every backend and transport.
    return {"discount": (revision % 21) / 100, "tax_rate": 0.08, "shipping_fee": 5}


def validate(response):
    if not isinstance(response, dict) or "error" in response:
        raise ValueError(f"Server rejected pricing settings: {response}")
    for key in ("count", "total_amount", "average_total", "preview", "backend", "compute_ms"):
        if key not in response:
            raise ValueError(f"Response is missing {key}")
    if not all(math.isfinite(response[key]) for key in ("total_amount", "average_total", "compute_ms")):
        raise ValueError("Non-finite amount or timing")


class RestUser(HttpUser):
    wait_time = constant(0)  # Saturation test; try constant(1) for a paced demo.

    def on_start(self):
        self.revision = 0

    @task
    def quote(self):
        with self.client.post("/quote", json=pricing_for(self.revision),
                              name="/quote", timeout=30, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    validate(response.json())
                except (ValueError, TypeError, KeyError) as error:
                    response.failure(str(error))
            # Locust records non-2xx responses as failures automatically.
        self.revision += 1


class WebSocketUser(User):
    wait_time = constant(0)

    def on_start(self):
        self.revision = 0
        self.socket = None
        parts = urlsplit(self.host)
        if parts.scheme not in {"http", "https", "ws", "wss"}:
            raise ValueError("Use an http(s) or ws(s) --host")
        scheme = "wss" if parts.scheme in {"https", "wss"} else "ws"
        self.url = urlunsplit((scheme, parts.netloc, "/quote", "", ""))
        self.connect()

    def connect(self):
        started, wall_time = perf_counter(), time()
        error = None
        try:
            self.socket = websocket.create_connection(
                self.url, timeout=30, enable_multithread=False,
                http_no_proxy=["127.0.0.1", "localhost"],
            )
        except Exception as exc:
            error = exc
        self.environment.events.request.fire(
            request_type="WS", name="/connect", response_time=(perf_counter() - started) * 1000,
            response_length=0, exception=error, start_time=wall_time,
        )
        if error:
            raise StopUser() from error

    @task
    def quote(self):
        if self.socket is None:
            self.connect()
        payload = json.dumps(pricing_for(self.revision))
        started, wall_time = perf_counter(), time()
        error, length = None, 0
        try:
            self.socket.send(payload)
            raw = self.socket.recv()
            elapsed = (perf_counter() - started) * 1000
            length = len(raw.encode("utf-8") if isinstance(raw, str) else raw)
            validate(json.loads(raw))
        except Exception as exc:
            elapsed = (perf_counter() - started) * 1000
            error = exc
            self.on_stop()
        # One event per send/receive round trip, including failures.
        self.environment.events.request.fire(
            request_type="WS", name="/quote", response_time=elapsed,
            response_length=length, exception=error, start_time=wall_time,
        )
        self.revision += 1

    def on_stop(self):
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
