#!/usr/bin/env python
######################################################
# Developer : Methun Kamruzzaman
# Date      : August, 2024
# Purpose   : Web application entry point for P2MAT — serves the browser UI,
#             accepts SMILES input, runs property prediction, and exports results.
######################################################
"""
p2mat.py
--------
Entry point and web server for P2MAT — Material Property Prediction.

Serves a single-page browser UI (``templates/index.html`` + ``static/``) where
the user can:
1. Type one or more SMILES strings (one per line).
2. Tick any combination of property checkboxes populated from ``config.json``.
3. Click **Predict properties**; the job runs in a background thread and the
   page polls for progress.
4. Inspect the results table and download it as CSV.

All computation is delegated to :class:`utils.worker.PredictionJob`.

Usage::

    python p2mat.py                          # http://127.0.0.1:8050, opens browser
    python p2mat.py --host 0.0.0.0 --port 80 --no-browser
"""
from __future__ import annotations

import argparse
import io
import os
import threading
import webbrowser

from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

from utils import helper
from utils.worker import JobStore, PredictionJob

# Maps config.json property keys → human-readable checkbox labels.
_LABEL_MAP: dict[str, str] = {
    "MP": "Melting point (K)",
    "BP": "Boiling point (K)",
    "HC": "Heat capacity (J/K)",
    "dH": "Heat of hydrogenation",
    "h2_uptake": "H₂ uptake",
}

# Maps raw result column names → display headers.
_COL_LABELS: dict[str, str] = {
    "smiles": "SMILES",
    "MP_pred_K": "Melting point (K)",
    "BP_pred_K": "Boiling point (K)",
    "HC_pred_JK": "Heat capacity (J/K)",
    "dH_pred_JK": "Heat of hydrogenation",
    "Saturated_smiles": "Saturated SMILES",
    "H2_uptake": "H₂ uptake",
}

_BATCH_SIZE: int = 128   # PaDEL descriptor batch size forwarded to the job
_VERBOSE: int = 0

app = Flask(
    __name__,
    template_folder=helper.resource_path("templates"),
    static_folder=helper.resource_path("static"),
)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MB of SMILES is plenty

_jobs = JobStore()


def _enabled_props() -> dict[str, str]:
    """Return ``{key: label}`` for properties marked true in ``config.json``."""
    config_props = helper.load_config().get("properties", {})
    return {k: label for k, label in _LABEL_MAP.items() if config_props.get(k, False)}


# ---------------------------------------------------------------------------
# Pages and assets
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html", props=_enabled_props())


@app.get("/design/<path:filename>")
def design_asset(filename: str):
    """Serve logo/branding files from ``design/``."""
    return send_from_directory(helper.resource_path("design"), filename)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.post("/api/predict")
def predict():
    """Start a prediction job.

    Body: ``{"smiles": ["CCO", ...], "props": ["MP", "BP"]}``.
    Returns ``202`` with the job snapshot (including ``id``).
    """
    body = request.get_json(silent=True) or {}
    smiles = [s.strip() for s in body.get("smiles", []) if isinstance(s, str) and s.strip()]
    if not smiles:
        return jsonify(error="Enter at least one SMILES string."), 400

    enabled = _enabled_props()
    requested = [p for p in body.get("props", []) if p in enabled]
    if not requested:
        return jsonify(error="Select at least one property."), 400

    props = {key: key in requested for key in enabled}
    job = _jobs.submit(PredictionJob(smiles, props, batch=_BATCH_SIZE, verbose=_VERBOSE))
    return jsonify(job.snapshot()), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        abort(404)
    data = job.snapshot()
    data["headers"] = [_COL_LABELS.get(c, c) for c in data["columns"]]
    return jsonify(data)


@app.get("/api/jobs/<job_id>/results.csv")
def job_csv(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job.result is None or job.result.empty:
        abort(404)
    buf = io.StringIO()
    job.result.to_csv(buf, index=False)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=p2mat_predictions.csv"},
    )


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P2MAT web server")
    parser.add_argument("--host", default=os.environ.get("P2MAT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("P2MAT_PORT", 8050)))
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser tab on startup.")
    args = parser.parse_args()

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    print(f"P2MAT running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()

    # threaded=True so status polling is served while a job runs.
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
