"""Standard-library-only order-pricing workload: compare identical work with 1 or N threads.

Use the SAME free-threaded executable with -X gil=1 and -X gil=0.
No NumPy/Numba imports: this measures Python execution rather than native kernels.
"""

import argparse
import json
import os
import random
import statistics
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from time import perf_counter


def price_chunk(job):
    orders, start, stop = job
    total = 0.0
    for i in range(start, stop):
        unit_price, quantity = orders[i]
        subtotal = unit_price * quantity
        discounted = subtotal * (1.0 - 0.1)
        total += discounted * (1.0 + 0.08) + 5.0
    return total


def benchmark(count=200_000, workers=4, repeat=3):
    rng = random.Random(7)
    # A separate stdlib-generated fixture; fixed and identical across the GIL runs.
    # Read-only input, per-task accumulators: no shared mutable result or locks.
    orders = tuple((rng.uniform(1, 100), rng.randint(1, 20)) for _ in range(count))
    jobs = [(orders, count * i // workers, count * (i + 1) // workers) for i in range(workers)]
    reference = [price_chunk(job) for job in jobs]
    samples = {"serial": [], "threads": []}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Force every lazy worker to start before timing any workload.
        ready = Barrier(workers + 1)
        startup = [pool.submit(ready.wait) for _ in range(workers)]
        ready.wait()
        for future in startup:
            future.result()
        for trial in range(repeat):
            order = ("serial", "threads") if trial % 2 == 0 else ("threads", "serial")
            for mode in order:
                started = perf_counter()
                result = ([price_chunk(job) for job in jobs] if mode == "serial"
                          else list(pool.map(price_chunk, jobs)))
                samples[mode].append(perf_counter() - started)
                if result != reference:
                    raise AssertionError("Threaded totals differ from serial totals")
    serial = statistics.median(samples["serial"])
    threaded = statistics.median(samples["threads"])
    return {
        "python": sys.version,
        "executable": sys.executable,
        "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "gil_enabled": sys._is_gil_enabled(),
        "cpu_count": os.cpu_count(),
        "count": count,
        "workers": workers,
        "repeat": repeat,
        "serial_seconds": serial,
        "threaded_seconds": threaded,
        "thread_speedup": serial / threaded,
        "samples_seconds": samples,
        "total_amount": sum(reference),
        "average_total": sum(reference) / count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--expect-gil", type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.count < 1 or args.workers < 1 or args.workers > args.count or args.repeat < 1:
        parser.error("count/repeat/workers must be positive and workers must not exceed count")
    if not hasattr(sys, "_is_gil_enabled"):
        parser.error("Use Python 3.13 or newer (3.14t recommended)")
    if args.expect_gil is not None:
        if not sysconfig.get_config_var("Py_GIL_DISABLED"):
            parser.error("Both comparison runs require a free-threaded build, e.g. python3.14t")
        if sys._is_gil_enabled() != bool(args.expect_gil):
            parser.error("Actual GIL state does not match --expect-gil")
    print(json.dumps(benchmark(args.count, args.workers, args.repeat)))


if __name__ == "__main__":
    main()
