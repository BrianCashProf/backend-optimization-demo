"""Batch order pricing, NumPy / Numba / CUDA. Run: python -m uvicorn rest_optimized:app --port 8002"""

import os
from contextlib import asynccontextmanager
from threading import Lock
from time import perf_counter

import numpy as np
from fastapi import FastAPI
from numba import njit
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool


class PricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    discount: float = Field(default=0.1, ge=0, le=0.5)
    tax_rate: float = Field(default=0.08, ge=0, le=0.3)
    shipping_fee: float = Field(default=5, ge=0, le=50)


def make_orders(n):
    rng = np.random.default_rng(7)
    return (
        rng.uniform(1, 100, n).astype(np.float32),
        rng.integers(1, 21, n).astype(np.float32),  # Whole quantities, stored as numeric arrays.
    )


def price_numpy(unit_prices, quantities, discount, tax_rate, shipping_fee):
    subtotal = unit_prices * quantities
    discounted = subtotal * (np.float32(1.0) - discount)
    return discounted * (np.float32(1.0) + tax_rate) + shipping_fee


@njit(nogil=True)
def price_numba(unit_prices, quantities, discount, tax_rate, shipping_fee):
    discount, tax_rate, shipping_fee = float(discount), float(tax_rate), float(shipping_fee)
    out = np.empty(len(unit_prices), dtype=np.float32)
    for i in range(len(unit_prices)):
        subtotal = float(unit_prices[i]) * float(quantities[i])
        discounted = subtotal * (1.0 - discount)
        out[i] = discounted * (1.0 + tax_rate) + shipping_fee
    return out


class CudaPricer:
    """Persistent device inputs; one locked output buffer per server process."""

    def __init__(self, orders):
        # Import only for an explicitly requested CUDA backend. Never silently fall back.
        global cuda
        from numba import cuda

        if not cuda.is_available():
            raise RuntimeError("CUDA unavailable; install requirements-cuda.txt and check the driver")

        @cuda.jit
        def kernel(unit_prices, quantities, discount, tax_rate, shipping_fee, out):
            start = cuda.grid(1)
            stride = cuda.gridsize(1)
            for i in range(start, unit_prices.size, stride):
                subtotal = unit_prices[i] * quantities[i]
                discounted = subtotal * (np.float32(1.0) - discount)
                out[i] = discounted * (np.float32(1.0) + tax_rate) + shipping_fee

        self.cuda = cuda
        self.kernel = kernel
        self.lock = Lock()
        self.n = len(orders[0])
        self.blocks = min(1024, (self.n + 127) // 128)
        with cuda.gpus[0]:
            self.inputs = tuple(cuda.to_device(a) for a in orders)
            self.output = cuda.device_array(self.n, dtype=np.float32)

    def __call__(self, discount, tax_rate, shipping_fee):
        # FastAPI may call from different worker threads. Select the context there.
        # Hold the lock through the copy so concurrent requests cannot mix results.
        with self.lock, self.cuda.gpus[0]:
            self.kernel[self.blocks, 128](*self.inputs, discount, tax_rate, shipping_fee, self.output)
            return self.output.copy_to_host()  # Includes GPU completion and D2H transfer.


@asynccontextmanager
async def lifespan(app):
    n = int(os.environ.get("PRICING_COUNT", "200000"))
    if not 1 <= n <= 5_000_000:
        raise ValueError("PRICING_COUNT must be between 1 and 5000000")
    backend = os.environ.get("PRICING_BACKEND", "numba")
    if backend not in {"numpy", "numba", "cuda"}:
        raise ValueError("PRICING_BACKEND must be numpy, numba, or cuda")
    app.state.orders = make_orders(n)
    app.state.backend = backend
    app.state.gpu = CudaPricer(app.state.orders) if backend == "cuda" else None
    # Warm the exact signature before accepting traffic; compilation is a startup cost.
    await run_in_threadpool(process, PricingRequest())
    yield


app = FastAPI(title="Batch order pricing", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, error):
    # Keep errors JSON-safe even if a client sends nonstandard NaN/Infinity literals.
    detail = [{key: issue[key] for key in ("loc", "msg", "type")} for issue in error.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


def process(pricing):
    started = perf_counter()
    parameters = tuple(np.float32(v) for v in (pricing.discount, pricing.tax_rate, pricing.shipping_fee))
    backend = app.state.backend
    if backend == "cuda":
        totals = app.state.gpu(*parameters)
    else:
        price = price_numba if backend == "numba" else price_numpy
        totals = price(*app.state.orders, *parameters)
    return {
        "count": len(totals),
        "total_amount": float(np.sum(totals, dtype=np.float64)),
        "average_total": float(np.mean(totals, dtype=np.float64)),
        "preview": totals[:8].tolist(),
        "backend": backend,
        "compute_ms": (perf_counter() - started) * 1000,
    }


@app.get("/health")
async def health():
    return {"status": "ready", "count": len(app.state.orders[0]), "backend": app.state.backend}


# Transport adapter: the WebSocket file changes only imports and this section.
@app.post("/quote")
async def quote(pricing: PricingRequest):
    return await run_in_threadpool(process, pricing)
