######################################################
# Developer : Methun Kamruzzaman
# Date      : May 2, 2026
# Purpose   : Background prediction jobs for the web server — runs property
#             prediction off the request thread and exposes progress for polling.
######################################################
"""
utils/worker.py
---------------
Background prediction jobs for the P2MAT web server.

Each call to ``POST /api/predict`` creates a :class:`PredictionJob` that runs
:class:`~utils.mspp.MaterialStructuralPropertyPrediction` in a daemon thread.
The browser polls ``GET /api/jobs/<id>``, which returns
:meth:`PredictionJob.snapshot`.

Only one job computes at a time (PaDEL is a heavy Java process that already
uses every core); later jobs wait in ``queued`` state until the earlier one
finishes.

This module intentionally contains no web code.  All HTTP handling lives in
``p2mat.py``.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid

import pandas as pd

from utils.mspp import MaterialStructuralPropertyPrediction

logger = logging.getLogger("p2mat")

# Serialises PaDEL/model runs across all jobs.
_RUN_LOCK = threading.Lock()

# Finished jobs are forgotten after this many seconds.
_JOB_TTL_SECONDS: int = 60 * 60


class PredictionJob:
    """One prediction request, run in a background thread.

    State moves ``queued`` → ``running`` → ``done`` | ``error``.  All mutable
    fields are guarded by ``self._lock`` because the request threads read them
    while the job thread writes them.

    Example::

        job = PredictionJob(["CCO"], props={"MP": True})
        job.start()
        job.snapshot()   # {"state": "running", "progress": 0, ...}
    """

    def __init__(
        self,
        smiles: list[str],
        props: dict,
        batch: int = 128,
        verbose: int = 0,
    ) -> None:
        """
        Args:
            smiles:  List of SMILES strings to process (one per molecule).
            props:   Property-enable dict, e.g. ``{"MP": True, "BP": False}``.
            batch:   PaDEL batch size passed to
                     :class:`~utils.mspp.MaterialStructuralPropertyPrediction`.
            verbose: Verbosity level forwarded to the prediction class.
        """
        self.id: str = uuid.uuid4().hex
        self._smiles = smiles
        self._props = props
        self._batch = batch
        self._verbose = verbose
        self._lock = threading.Lock()

        self.state: str = "queued"
        self.progress: int = 0
        self.total: int = len(smiles)
        self.status: str = "Waiting for an earlier prediction to finish…"
        self.invalid_smiles: list[str] = []
        self.error: str | None = None
        self.result: pd.DataFrame | None = None
        self.finished_at: float | None = None

    # ------------------------------------------------------------------
    # Callbacks from the pipeline (job thread)
    # ------------------------------------------------------------------

    def _set(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, key, value)

    def _on_progress(self, n: int) -> None:
        self._set(progress=n)

    def _on_status(self, msg: str) -> None:
        self._set(status=msg.strip())

    def _on_valid_count(self, n: int) -> None:
        self._set(total=n)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Run the job in a daemon thread and return immediately."""
        threading.Thread(target=self._run, name=f"p2mat-job-{self.id[:8]}",
                         daemon=True).start()

    def _run(self) -> None:
        with _RUN_LOCK:
            self._set(state="running", status="Starting…")
            mspp = None
            try:
                mspp = MaterialStructuralPropertyPrediction(
                    props=self._props,
                    batch=self._batch,
                    verbose=self._verbose,
                    progress_callback=self._on_progress,
                    status_callback=self._on_status,
                    valid_count_callback=self._on_valid_count,
                )
                invalid = mspp.prediction_from_smile(self._smiles)
            except Exception as exc:
                logger.error("Prediction failed:\n%s", traceback.format_exc())
                # mspp._invalid_smiles is set before descriptors are computed,
                # so RDKit-invalid SMILES are still reported after a failure.
                self._set(
                    state="error",
                    error=str(exc) or exc.__class__.__name__,
                    invalid_smiles=list(mspp._invalid_smiles) if mspp else [],
                    finished_at=time.time(),
                )
                return

            self._set(
                state="done",
                status="Prediction complete.",
                invalid_smiles=list(invalid),
                result=mspp.get_data(),
                finished_at=time.time(),
            )

    # ------------------------------------------------------------------
    # Read access (request threads)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a JSON-serialisable view of the job's current state.

        ``rows`` is included only once the job is done; numeric NaN values
        become ``None``.
        """
        with self._lock:
            data = {
                "id": self.id,
                "state": self.state,
                "progress": self.progress,
                "total": self.total,
                "status": self.status,
                "invalid_smiles": list(self.invalid_smiles),
                "error": self.error,
                "columns": [],
                "rows": [],
            }
            df = self.result
        if df is not None and not df.empty:
            data["columns"] = list(df.columns)
            data["rows"] = df.astype(object).where(df.notna(), None).values.tolist()
        return data


class JobStore:
    """Thread-safe registry of :class:`PredictionJob` objects by id."""

    def __init__(self) -> None:
        self._jobs: dict[str, PredictionJob] = {}
        self._lock = threading.Lock()

    def submit(self, job: PredictionJob) -> PredictionJob:
        """Register *job*, start it, and drop expired finished jobs."""
        self._prune()
        with self._lock:
            self._jobs[job.id] = job
        job.start()
        return job

    def get(self, job_id: str) -> PredictionJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _prune(self) -> None:
        cutoff = time.time() - _JOB_TTL_SECONDS
        with self._lock:
            expired = [jid for jid, j in self._jobs.items()
                       if j.finished_at is not None and j.finished_at < cutoff]
            for jid in expired:
                del self._jobs[jid]
