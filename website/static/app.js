// P2MAT browser UI: submits SMILES to /api/predict, polls the job, renders results.
(function () {
  "use strict";

  const POLL_MS = 600;
  const STRING_COLS = new Set(["smiles", "Saturated_smiles"]);
  const EXAMPLES = [
    "C1=CC=C(C=C1)OCC#CC#CCOC2=CC=CC=C2",
    "C1=CC=C(C=C1)C#CC#CC2=CC=CC=C2",
    "C1CCC(C1)(C#CC#CC2(CCCC2)O)O",
    "C1=CC=C(C=C1)C#CC2=CC=C(C=C2)C#CC3=CC=CC=C3",
    "C#CC1=CC=CC=C1",
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
  ];

  const $ = (id) => document.getElementById(id);
  const form = $("predict-form");
  const smilesInput = $("smiles");
  const predictBtn = $("predict-btn");
  const checkboxes = Array.from(form.querySelectorAll('input[name="props"]'));

  // --- Input helpers ----------------------------------------------------

  function getSmiles() {
    return smilesInput.value.split("\n").map((s) => s.trim()).filter(Boolean);
  }

  function updateCount() {
    const n = getSmiles().length;
    $("smiles-count").textContent = n === 1 ? "1 molecule" : `${n} molecules`;
  }

  smilesInput.addEventListener("input", updateCount);

  $("load-example").addEventListener("click", () => {
    smilesInput.value = EXAMPLES.join("\n");
    updateCount();
    smilesInput.focus();
  });

  // At least one property must stay checked.
  checkboxes.forEach((cb) =>
    cb.addEventListener("change", () => {
      if (!checkboxes.some((c) => c.checked)) cb.checked = true;
    })
  );

  // --- Rendering --------------------------------------------------------

  function resetOutput() {
    $("messages").hidden = true;
    $("error-box").hidden = true;
    $("invalid-box").hidden = true;
    $("invalid-list").replaceChildren();
    $("results").hidden = true;
  }

  function showError(message) {
    $("error-box").textContent = message;
    $("error-box").hidden = false;
    $("messages").hidden = false;
  }

  function showInvalid(list) {
    if (!list || !list.length) return;
    $("invalid-list").replaceChildren(
      ...list.map((s) => {
        const li = document.createElement("li");
        li.textContent = s;
        return li;
      })
    );
    $("invalid-box").hidden = false;
    $("messages").hidden = false;
  }

  function showProgress(job) {
    $("progress").hidden = false;
    $("status-text").textContent = job.status || "Working…";
    const bar = $("bar-fill").parentElement;
    if (job.state === "queued" || !job.total) {
      bar.classList.add("indeterminate");
      $("progress-count").textContent = "";
    } else {
      bar.classList.remove("indeterminate");
      const pct = Math.min(100, (100 * job.progress) / job.total);
      $("bar-fill").style.width = `${pct}%`;
      $("progress-count").textContent = `${job.progress} / ${job.total}`;
    }
  }

  function formatCell(value, col) {
    if (value === null || value === undefined) return "";
    if (!STRING_COLS.has(col) && typeof value === "number") {
      return (Math.round(value * 1000) / 1000).toString();
    }
    return String(value);
  }

  function showResults(job) {
    if (!job.rows || !job.rows.length) return;

    const headRow = document.createElement("tr");
    job.columns.forEach((col, i) => {
      const th = document.createElement("th");
      th.textContent = job.headers[i];
      if (!STRING_COLS.has(col)) th.className = "num";
      headRow.appendChild(th);
    });

    const bodyRows = job.rows.map((row) => {
      const tr = document.createElement("tr");
      row.forEach((value, i) => {
        const col = job.columns[i];
        const td = document.createElement("td");
        td.className = STRING_COLS.has(col) ? "smiles" : "num";
        td.textContent = formatCell(value, col);
        tr.appendChild(td);
      });
      return tr;
    });

    const table = $("result-table");
    table.tHead.replaceChildren(headRow);
    table.tBodies[0].replaceChildren(...bodyRows);
    $("result-count").textContent =
      job.rows.length === 1 ? "· 1 molecule" : `· ${job.rows.length} molecules`;
    $("download-btn").href = `/api/jobs/${job.id}/results.csv`;
    $("results").hidden = false;
  }

  // --- Job lifecycle ----------------------------------------------------

  function finish(job) {
    predictBtn.disabled = false;
    $("progress").hidden = true;
    if (job.state === "error") showError(`Error: ${job.error}`);
    showInvalid(job.invalid_smiles);
    showResults(job);
    if (job.state === "done" && (!job.rows || !job.rows.length) && !(job.invalid_smiles || []).length) {
      showError("No predictions were produced.");
    }
  }

  async function poll(jobId) {
    try {
      const res = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const job = await res.json();
      if (job.state === "done" || job.state === "error") {
        finish(job);
        return;
      }
      showProgress(job);
    } catch (err) {
      // Transient network blip: keep polling, but tell the user.
      $("status-text").textContent = `Waiting for server… (${err.message})`;
    }
    setTimeout(() => poll(jobId), POLL_MS);
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    resetOutput();

    const smiles = getSmiles();
    if (!smiles.length) {
      showError("Enter at least one SMILES string.");
      smilesInput.focus();
      return;
    }
    const props = checkboxes.filter((c) => c.checked).map((c) => c.value);

    predictBtn.disabled = true;
    showProgress({ state: "queued", status: "Submitting…" });

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles, props }),
      });
      const job = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(job.error || `Server returned ${res.status}`);
      showProgress(job);
      poll(job.id);
    } catch (err) {
      predictBtn.disabled = false;
      $("progress").hidden = true;
      showError(`Error: ${err.message}`);
    }
  });

  updateCount();
})();
