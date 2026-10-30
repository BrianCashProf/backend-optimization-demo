"""Batch order pricing, ordinary Python. Run: python -m uvicorn websocket_unoptimized:app --port 8003"""

import os
from contextlib import asynccontextmanager
from time import perf_counter

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
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


def price_python(unit_prices, quantities, discount, tax_rate, shipping_fee):
    discount, tax_rate, shipping_fee = float(discount), float(tax_rate), float(shipping_fee)
    out = np.empty(len(unit_prices), dtype=np.float32)
    for i in range(len(unit_prices)):
        subtotal = float(unit_prices[i]) * float(quantities[i])
        discounted = subtotal * (1.0 - discount)
        out[i] = discounted * (1.0 + tax_rate) + shipping_fee
    return out

@asynccontextmanager
async def lifespan(app):
    n = int(os.environ.get("PRICING_COUNT", "200000"))
    if not 1 <= n <= 5_000_000:
        raise ValueError("PRICING_COUNT must be between 1 and 5000000")
    app.state.orders = make_orders(n)
    yield


app = FastAPI(title="Batch order pricing", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, error):
    # Keep errors JSON-safe even if a client sends nonstandard NaN/Infinity literals.
    detail = [{key: issue[key] for key in ("loc", "msg", "type")} for issue in error.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


def process(pricing):
    started = perf_counter()
    totals = price_python(*app.state.orders, pricing.discount, pricing.tax_rate, pricing.shipping_fee)
    return {
        "count": len(totals),
        "total_amount": float(np.sum(totals, dtype=np.float64)),
        "average_total": float(np.mean(totals, dtype=np.float64)),
        "preview": totals[:8].tolist(),
        "backend": "python",
        "compute_ms": (perf_counter() - started) * 1000,
    }


@app.get("/health")
async def health():
    return {"status": "ready", "count": len(app.state.orders[0]), "backend": "python"}


# Transport adapter: same PricingRequest model, process(), orders, and response as REST.
@app.websocket("/quote")
async def quote(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                pricing = PricingRequest.model_validate(await websocket.receive_json())
            except (ValidationError, ValueError):
                await websocket.send_json({
                    "error": "invalid_pricing",
                    "detail": "Send finite discount (0..0.5), tax_rate (0..0.3), shipping_fee (0..50).",
                })
                continue
            result = await run_in_threadpool(process, pricing)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
