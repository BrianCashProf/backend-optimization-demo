"""Numerical, transport, and concurrent-buffer checks. python -m unittest -v test_demo

PRICING_TEST_CUDA=1 python -m unittest -v test_demo   # Also exercise real GPU hardware.
"""

import ast
import importlib
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import rest_optimized as fast
import rest_unoptimized as slow

PAYLOADS = [
    {},
    {"discount": 0.2, "tax_rate": 0.1, "shipping_fee": 8},
    {"discount": 0.5, "tax_rate": 0.3, "shipping_fee": 50},
]


def reference_response(payload, n=1003):
    pricing = slow.PricingRequest(**payload)
    totals = slow.price_python(*slow.make_orders(n), pricing.discount, pricing.tax_rate, pricing.shipping_fee)
    return totals


class DemoTests(unittest.TestCase):
    def test_known_order_totals(self):
        # 2 x $10 and 3 x $20, then 10% discount, 10% tax, and $5 shipping per order.
        orders = (np.array([10, 20], dtype=np.float32), np.array([2, 3], dtype=np.float32))
        for price in (slow.price_python, fast.price_numpy, fast.price_numba):
            np.testing.assert_allclose(price(*orders, 0.1, 0.1, 5.0), [24.8, 64.4], rtol=2e-6)

    def check_response(self, result, payload, n=1003):
        totals = reference_response(payload, n)
        self.assertEqual(result["count"], n)
        np.testing.assert_allclose(result["total_amount"], np.sum(totals, dtype=np.float64), rtol=2e-6, atol=1e-3)
        np.testing.assert_allclose(result["average_total"], np.mean(totals, dtype=np.float64), rtol=2e-6, atol=1e-3)
        np.testing.assert_allclose(result["preview"], totals[:8], rtol=2e-6, atol=1e-3)
        self.assertGreaterEqual(result["compute_ms"], 0)

    def test_cpu_full_arrays_and_tails(self):
        for n in (1, 127, 1003):
            for payload in PAYLOADS:
                pricing = slow.PricingRequest(**payload)
                args = tuple(np.float32(v) for v in (pricing.discount, pricing.tax_rate, pricing.shipping_fee))
                orders = fast.make_orders(n)
                expected = reference_response(payload, n)
                for price in (fast.price_numpy, fast.price_numba):
                    with self.subTest(n=n, price=price, payload=payload):
                        np.testing.assert_allclose(price(*orders, *args), expected, rtol=2e-6, atol=1e-3)

    def test_rest_endpoints_and_validation(self):
        for module, backend in ((slow, "python"), (fast, "numpy"), (fast, "numba")):
            with self.subTest(module=module.__name__, backend=backend):
                with patch.dict(os.environ, PRICING_COUNT="1003", PRICING_BACKEND=backend):
                    with TestClient(module.app) as client:
                        self.assertEqual(client.get("/health").json()["backend"], backend)
                        for payload in PAYLOADS:
                            response = client.post("/quote", json=payload)
                            self.assertEqual(response.status_code, 200)
                            self.check_response(response.json(), payload)
                        for invalid in ({"discount": -0.1}, {"discount": 0.6}, {"tax_rate": 0.4}, {"shipping_fee": -1}, {"extra": True}, []):
                            self.assertEqual(client.post("/quote", json=invalid).status_code, 422)
                        self.assertEqual(client.post("/quote", content='{"discount": NaN}',
                                                     headers={"Content-Type": "application/json"}).status_code, 422)

    def test_websocket_repeated_quotes_and_recovery(self):
        for name, backend in (("websocket_unoptimized", "python"),
                              ("websocket_optimized", "numpy"), ("websocket_optimized", "numba")):
            module = importlib.import_module(name)
            with self.subTest(module=name, backend=backend):
                with patch.dict(os.environ, PRICING_COUNT="1003", PRICING_BACKEND=backend):
                    with TestClient(module.app) as client:
                        with client.websocket_connect("/quote") as socket:
                            for invalid in ('{"discount": -0.1}', 'null', '[]', 'not json', '{"discount": NaN}'):
                                socket.send_text(invalid)
                                self.assertEqual(socket.receive_json()["error"], "invalid_pricing")
                            for payload in PAYLOADS:
                                socket.send_json(payload)
                                self.check_response(socket.receive_json(), payload)

    def test_rest_websocket_core_is_identical(self):
        # Keep the requested teaching property intact as files evolve.
        for level in ("optimized", "unoptimized"):
            cores = []
            for transport in ("rest", "websocket"):
                tree = ast.parse(Path(__file__).with_name(f"{transport}_{level}.py").read_text())
                cores.append([ast.dump(node) for node in tree.body
                              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                              and node.name != "quote"])
            self.assertEqual(*cores)

    def test_bad_backend_does_not_silently_fall_back(self):
        with patch.dict(os.environ, PRICING_COUNT="1003", PRICING_BACKEND="typo"):
            with self.assertRaisesRegex(ValueError, "PRICING_BACKEND"):
                with TestClient(fast.app):
                    pass

    @unittest.skipUnless(os.environ.get("PRICING_TEST_CUDA") == "1", "Set PRICING_TEST_CUDA=1 for real GPU checks")
    def test_cuda_full_arrays_tail_and_concurrent_requests(self):
        n = 131_079  # Exceeds the capped grid size; also has a non-block-aligned tail.
        engine = fast.CudaPricer(fast.make_orders(n))
        args = [(np.float32(i / 100), np.float32(0.08), np.float32(i)) for i in range(8)]
        engine(*args[0])  # Compile before concurrent launches.
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda values: engine(*values), args))
        for values, result in zip(args, actual):
            expected = fast.price_numpy(*fast.make_orders(n), *values)
            np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-3)
        for name in ("rest_optimized", "websocket_optimized"):
            module = importlib.import_module(name)
            with patch.dict(os.environ, PRICING_COUNT="1003", PRICING_BACKEND="cuda"):
                with TestClient(module.app) as client:
                    if name.startswith("rest"):
                        self.check_response(client.post("/quote", json=PAYLOADS[1]).json(), PAYLOADS[1])
                    else:
                        with client.websocket_connect("/quote") as socket:
                            for payload in PAYLOADS:
                                socket.send_json(payload)
                                self.check_response(socket.receive_json(), payload)


if __name__ == "__main__":
    unittest.main()
