"""Start one API at a time, run Locust, save HTML/CSV reports, and stop every child.

python demo_runner.py --users 1 4 --seconds 10 --backend numba
python demo_runner.py --users 1 4 --seconds 10 --backend cuda
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from html import escape
from pathlib import Path
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parent


def read_html_statistics(report_path):
    # Periodic CSV snapshots can omit the final completed requests. Use the same
    # frozen statistics as the HTML report so its numbers match the overview.
    document = report_path.read_text(encoding="utf-8")
    marker = re.search(r"window\.templateArgs\s*=\s*", document)
    if marker is None:
        raise RuntimeError(f"Locust HTML report has no embedded statistics: {report_path}")
    data, _ = json.JSONDecoder().raw_decode(document[marker.end():].lstrip())
    stats = next(row for row in data["requests_statistics"] if row["name"] == "/quote")
    percentiles = next(row for row in data["response_time_statistics"] if row["name"] == "/quote")
    return {
        "requests": int(stats["num_requests"]),
        "failures": int(stats["num_failures"]),
        "requests_per_second": float(stats["total_rps"]),
        "p50_ms": float(percentiles["0.5"]),
        "p95_ms": float(percentiles["0.95"]),
        "mean_ms": float(stats["avg_response_time"]),
    }


def write_report_index(folder, rows, seconds):
    report_rows = []
    for row in rows:
        report_rows.append(
            f"<tr><td>{escape(row['server'])}</td><td>{escape(row['backend'])}</td>"
            f"<td>{row['users']}</td><td>{row['requests']:,}</td><td>{row['failures']}</td>"
            f"<td>{row['requests_per_second']:.1f}</td><td>{row['p50_ms']:.1f}</td>"
            f"<td>{row['p95_ms']:.1f}</td>"
            f"<td><a href=\"{escape(row['html_report'], quote=True)}\">Open Locust report</a></td></tr>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Order pricing — Locust reports</title>
<style>
body {{ font: 16px/1.5 system-ui, sans-serif; color: #172033; background: #f6f8fb;
        margin: 0; padding: 40px 24px; }}
main {{ max-width: 1200px; margin: auto; }}
h1 {{ margin-bottom: 8px; }}
p {{ color: #475569; }}
.table {{ overflow-x: auto; background: white; border: 1px solid #dbe2ec; border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; white-space: nowrap; }}
th, td {{ padding: 14px; text-align: right; border-bottom: 1px solid #e2e8f0; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:last-child, td:last-child {{ text-align: left; }}
th {{ background: #edf2f8; }}
a {{ color: #1d4ed8; }}
</style>
</head>
<body><main>
<h1>Order pricing: Locust reports</h1>
<p>{rows[0]['count']:,} orders per quote · {seconds} seconds per run · one Uvicorn worker.</p>
<p>Compare Python and optimized implementations over REST and WebSockets.
Open a report for Locust's request statistics, latency percentiles, and charts.</p>
<div class="table"><table>
<thead><tr><th>API</th><th>Backend</th><th>Users</th><th>Quotes</th><th>Failures</th>
<th>Quotes/s</th><th>p50 ms</th><th>p95 ms</th><th>Report</th></tr></thead>
<tbody>{''.join(report_rows)}</tbody></table></div>
<p>This table uses the <code>/quote</code> requests. WebSocket handshakes appear separately
as <code>/connect</code> in the full reports. Timings include the client round trip.</p>
<p>These are saved results from a local load test. Short runs illustrate behavior;
repeat longer trials for capacity conclusions. API servers can be stopped while viewing reports.</p>
</main></body></html>
"""
    (folder / "index.html").write_text(document, encoding="utf-8")


@contextmanager
def server(module, count, backend, folder):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = dict(os.environ, PRICING_COUNT=str(count), PRICING_BACKEND=backend)
    log_path = folder / f"{module}.log"
    with log_path.open("w") as log:
        child = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", f"{module}:app", "--host", "127.0.0.1",
             "--port", str(port), "--workers", "1", "--no-access-log"],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            host = f"http://127.0.0.1:{port}"
            opener = build_opener(ProxyHandler({}))
            deadline = monotonic() + 120
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f"{module} stopped during startup. See {log_path}\n{log_path.read_text()}")
                try:
                    with opener.open(host + "/health", timeout=1) as response:
                        health = json.load(response)
                    expected = "python" if module.endswith("unoptimized") else backend
                    if health.get("count") != count or health.get("backend") != expected:
                        raise RuntimeError(f"Unexpected server configuration: {health}")
                    break
                except (URLError, TimeoutError):
                    if monotonic() >= deadline:
                        raise TimeoutError(f"{module} never became ready; see {log_path}")
                    sleep(0.1)
            yield host
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def run_suite(users=(1, 4), seconds=8, count=200_000, backend="numba", output=None):
    if not users or any(u < 1 for u in users) or seconds < 1 or not 1 <= count <= 5_000_000:
        raise ValueError("Require positive users/seconds and count between 1 and 5000000")
    if backend not in {"numpy", "numba", "cuda"}:
        raise ValueError("backend must be numpy, numba, or cuda")
    parent = Path(output).resolve() if output else ROOT / "locust-results"
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    rows = []
    cases = [("rest_unoptimized", "RestUser"), ("rest_optimized", "RestUser"),
             ("websocket_unoptimized", "WebSocketUser"), ("websocket_optimized", "WebSocketUser")]
    for module, user_class in cases:
        with server(module, count, backend, folder) as host:
            for concurrency in users:
                label = f"{module}-{concurrency}users"
                prefix = folder / label
                html_report = folder / f"{label}.html"
                print(f"Locust: {module}, {concurrency} user(s), {seconds}s", flush=True)
                command = [
                    sys.executable, "-m", "locust", "-f", str(ROOT / "locustfile.py"), user_class,
                    "--host", host, "--headless", "--users", str(concurrency),
                    "--spawn-rate", str(concurrency), "--run-time", f"{seconds}s",
                    "--reset-stats", "--stop-timeout", "30", "--only-summary",
                    "--exit-code-on-error", "1", "--csv", str(prefix),
                    "--html", str(html_report),
                ]
                with (folder / f"{label}.log").open("w") as log:
                    try:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       check=True, timeout=seconds + 60)
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                        raise RuntimeError(f"Locust failed. See {folder / (label + '.log')}") from error
                if not html_report.is_file() or html_report.stat().st_size == 0:
                    raise RuntimeError(f"Locust did not produce its HTML report: {html_report}")
                stats = read_html_statistics(html_report)
                if stats["requests"] == 0 or stats["failures"]:
                    raise RuntimeError(f"No successful samples or failures in {html_report}")
                rows.append({
                    "server": module, "backend": "python" if module.endswith("unoptimized") else backend,
                    "users": concurrency, "count": count,
                    **stats,
                    "html_report": html_report.name,
                })
    (folder / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_report_index(folder, rows, seconds)
    latest_rows = [dict(row, html_report=f"{folder.name}/{row['html_report']}") for row in rows]
    write_report_index(parent, latest_rows, seconds)
    print(f"Latest HTML reports: {parent / 'index.html'}", flush=True)
    print(f"Open the HTML reports: {folder / 'index.html'}", flush=True)
    print(f"Locust HTML reports, CSVs, and logs: {folder}", flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--seconds", type=int, default=8)
    parser.add_argument("--count", type=int, default=200_000)
    parser.add_argument("--backend", choices=["numpy", "numba", "cuda"], default="numba")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(run_suite(args.users, args.seconds, args.count, args.backend, args.output), indent=2))
