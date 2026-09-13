const answerView = document.getElementById("answer-view");
const planView = document.getElementById("plan-view");
const reflectionView = document.getElementById("reflection-view");
const toolsView = document.getElementById("tools-view");
const evidenceView = document.getElementById("evidence-view");
const statusPill = document.getElementById("status-pill");
const runHistory = document.getElementById("run-history");
const evalHistory = document.getElementById("eval-history");
const runSummary = document.getElementById("run-summary");

function setStatus(label, mode = "idle") {
  statusPill.textContent = label;
  statusPill.className = `status-pill ${mode}`;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function renderResult(payload) {
  const result = payload.result || payload;
  const workflow = result.workflow_result || result;
  answerView.innerHTML = "";
  const pre = document.createElement("pre");
  pre.textContent = workflow.answer || result.message || workflow.message || pretty(result);
  answerView.appendChild(pre);
  renderRunSummary(workflow);
  planView.textContent = pretty(workflow.plan || {});
  reflectionView.textContent = pretty(workflow.reflection || {});
  renderEvidence(workflow.evidence || []);
  toolsView.textContent = pretty(workflow.tool_history || []);
}

function renderRunSummary(workflow) {
  const tools = workflow.tool_history || [];
  const evidence = workflow.evidence || [];
  const failedTools = tools.filter((item) => item.status !== "success");
  const totalMs = tools.reduce((sum, item) => sum + Number(item.elapsed_ms || 0), 0);
  const reflection = workflow.reflection || {};

  runSummary.innerHTML = `
    <div class="summary-chip">Tools: ${tools.length}</div>
    <div class="summary-chip">Failures: ${failedTools.length}</div>
    <div class="summary-chip">Evidence: ${evidence.length}</div>
    <div class="summary-chip">Tool time: ${totalMs} ms</div>
    <div class="summary-chip">Reflection: ${reflection.passed === false ? "补检/复核" : "通过/未知"}</div>
  `;

  if (failedTools.length) {
    const list = document.createElement("div");
    list.className = "failure-list";
    list.textContent = failedTools
      .map((item) => `${item.name}: ${item.error_type || item.status} (${item.result_preview || ""})`)
      .join("\n");
    runSummary.appendChild(list);
  }
}

function renderEvidence(items) {
  if (!items.length) {
    evidenceView.textContent = "No evidence yet.";
    return;
  }
  evidenceView.innerHTML = "";
  for (const item of items.slice(0, 8)) {
    const metadata = item.metadata || {};
    const block = document.createElement("div");
    block.className = "evidence-card";
    block.innerHTML = `
      <strong>${item.title || "unknown"}</strong>
      <span>${item.section || "section"} · score=${item.score ?? 0} · page=${metadata.page_start || "-"}</span>
      <span>${metadata.retrieval_reason || item.origin || ""}</span>
      <p>${metadata.citation_preview || item.content || ""}</p>
    `;
    evidenceView.appendChild(block);
  }
}

function renderHistoryMessage(target, message) {
  target.innerHTML = "";
  const item = document.createElement("p");
  item.className = "history-empty";
  item.textContent = message;
  target.appendChild(item);
}

function renderRunHistory(runs) {
  runHistory.innerHTML = "";
  if (!runs.length) {
    renderHistoryMessage(runHistory, "No workflow runs yet.");
    return;
  }

  for (const run of runs) {
    const item = document.createElement("button");
    item.className = "history-item";
    item.type = "button";
    item.dataset.runId = run.run_id;
    item.innerHTML = `
      <span class="history-title">${run.question || "(no question)"}</span>
      <span class="history-meta">${run.mode || "qa"} · ${run.tool_count} tools · ${run.evidence_count} evidence</span>
      <span class="history-meta">${run.created_at || ""}</span>
    `;
    item.addEventListener("click", () => loadRunDetail(run.run_id));
    runHistory.appendChild(item);
  }
}

function renderEvalHistory(evalRuns) {
  evalHistory.innerHTML = "";
  if (!evalRuns.length) {
    renderHistoryMessage(evalHistory, "No evaluation runs yet.");
    return;
  }

  for (const run of evalRuns) {
    const item = document.createElement("div");
    item.className = "history-item static";
    const averages = run.averages || {};
    item.innerHTML = `
      <span class="history-title">Eval ${run.eval_run_id}</span>
      <span class="history-meta">${run.question_count} questions · ${run.error_count} errors · ${run.item_count} rows</span>
      <span class="history-meta">recall@5=${averages.recall_at_5 ?? "-"} · citation=${averages.citation_accuracy ?? "-"}</span>
    `;
    evalHistory.appendChild(item);
  }
}

async function loadRunHistory() {
  try {
    const response = await fetch("/api/history/runs");
    const data = await response.json();
    if (!response.ok || data.ok === false) {
      throw new Error(data.error || "Failed to load run history");
    }
    if (!data.database_enabled) {
      renderHistoryMessage(runHistory, data.message || "MySQL persistence is disabled.");
      return;
    }
    renderRunHistory(data.runs || []);
  } catch (error) {
    renderHistoryMessage(runHistory, error.message);
  }
}

async function loadEvalHistory() {
  try {
    const response = await fetch("/api/history/evals");
    const data = await response.json();
    if (!response.ok || data.ok === false) {
      throw new Error(data.error || "Failed to load eval history");
    }
    if (!data.database_enabled) {
      renderHistoryMessage(evalHistory, data.message || "MySQL persistence is disabled.");
      return;
    }
    renderEvalHistory(data.eval_runs || []);
  } catch (error) {
    renderHistoryMessage(evalHistory, error.message);
  }
}

async function loadRunDetail(runId) {
  try {
    setStatus("Loading run...", "loading");
    const response = await fetch(`/api/history/runs/${encodeURIComponent(runId)}`);
    const data = await response.json();
    if (!response.ok || data.ok === false) {
      throw new Error(data.error || data.detail || "Failed to load run");
    }
    renderResult({ result: data.run });
    setStatus("Run loaded", "done");
  } catch (error) {
    answerView.textContent = error.message;
    setStatus("Error", "error");
  }
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || data.detail || "Request failed");
  }
  return data;
}

async function postSse(url, body, onEvent) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || data.detail || "Streaming request failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const messages = buffer.split("\n\n");
    buffer = messages.pop() || "";
    for (const message of messages) {
      const dataLine = message.split("\n").find((line) => line.startsWith("data: "));
      if (dataLine) onEvent(JSON.parse(dataLine.slice(6)));
    }
    if (done) break;
  }
}

function documentMetadata() {
  return {
    document_id: document.getElementById("document-id").value.trim(),
    title: document.getElementById("document-title").value.trim(),
    industry: document.getElementById("industry").value.trim(),
    company: document.getElementById("company").value.trim(),
    product_line: document.getElementById("product-line").value.trim(),
    document_type: document.getElementById("document-type").value.trim(),
  };
}

document.getElementById("ask-btn").addEventListener("click", async () => {
  const question = document.getElementById("question-input").value.trim();
  const threadId = document.getElementById("thread-id").value.trim();
  const userId = document.getElementById("user-id").value.trim();
  if (!question) {
    setStatus("Question required", "error");
    return;
  }
  try {
    setStatus("Running agent...", "loading");
    answerView.innerHTML = "<pre></pre>";
    const answer = answerView.querySelector("pre");
    let finalPayload = null;
    await postSse("/api/ask/stream", {
        question,
        thread_id: threadId,
        user_id: userId,
      }, (event) => {
        if (event.event === "node") {
          if (event.node === "summary" && event.status === "start") {
            answer.textContent = "";
          }
          setStatus(`${event.node}: ${event.status}`, event.status === "error" ? "error" : "loading");
        } else if (event.event === "token") {
          answer.textContent += event.content || "";
        } else if (event.event === "result") {
          finalPayload = { result: event.result };
        } else if (event.event === "error") {
          throw new Error(event.error || "Agent stream failed");
        }
    });
    if (finalPayload) renderResult(finalPayload);
    loadRunHistory();
    setStatus("Completed", "done");
  } catch (error) {
    answerView.textContent = error.message;
    setStatus("Error", "error");
  }
});

document.getElementById("brief-btn").addEventListener("click", async () => {
  try {
    setStatus("Generating research brief...", "loading");
    const payload = await postJson("/api/generate-brief", {});
    renderResult(payload);
    loadRunHistory();
    setStatus("Research brief done", "done");
  } catch (error) {
    answerView.textContent = error.message;
    setStatus("Error", "error");
  }
});

document.getElementById("graph-btn").addEventListener("click", async () => {
  try {
    setStatus("Loading graph...", "loading");
    const response = await fetch("/api/graph");
    const data = await response.json();
    if (!response.ok || data.ok === false) {
      throw new Error(data.error || "Failed to load graph");
    }
    answerView.innerHTML = `<pre>${data.mermaid}</pre>`;
    setStatus("Graph ready", "done");
  } catch (error) {
    answerView.textContent = error.message;
    setStatus("Error", "error");
  }
});

document.getElementById("upload-btn").addEventListener("click", async () => {
  const fileInput = document.getElementById("pdf-file");
  const file = fileInput.files[0];
  if (!file) {
    setStatus("Select a PDF first", "error");
    return;
  }

  const form = new FormData();
  form.append("file", file);
  for (const [key, value] of Object.entries(documentMetadata())) {
    form.append(key, value);
  }

  try {
    setStatus("Indexing research document...", "loading");
    const response = await fetch("/api/ingest-upload", {
      method: "POST",
      body: form,
    });
    const data = await response.json();
    if (!response.ok || data.ok === false) {
      throw new Error(data.error || data.detail || "Upload failed");
    }
    renderResult({
      result: {
        answer: "Research document indexed successfully.",
        plan: data.result,
        reflection: {},
        tool_history: [],
      },
    });
    setStatus("Upload complete", "done");
  } catch (error) {
    answerView.textContent = error.message;
    setStatus("Error", "error");
  }
});

document.getElementById("url-upload-btn").addEventListener("click", async () => {
  const url = document.getElementById("pdf-url").value.trim();
  if (!url) {
    setStatus("Enter a PDF URL first", "error");
    return;
  }

  try {
    setStatus("Downloading and indexing URL...", "loading");
    const data = await postJson("/api/ingest-url", {
      url,
      ...documentMetadata(),
    });
    renderResult({
      result: {
        answer: "Research document indexed successfully.",
        plan: data.result,
        reflection: {},
        tool_history: [],
      },
    });
    setStatus("URL indexed", "done");
  } catch (error) {
    answerView.textContent = error.message;
    setStatus("Error", "error");
  }
});

document.getElementById("refresh-runs-btn").addEventListener("click", loadRunHistory);
document.getElementById("refresh-evals-btn").addEventListener("click", loadEvalHistory);

loadRunHistory();
loadEvalHistory();
