const answerView = document.getElementById("answer-view");
const planView = document.getElementById("plan-view");
const reflectionView = document.getElementById("reflection-view");
const toolsView = document.getElementById("tools-view");
const statusPill = document.getElementById("status-pill");

function setStatus(label, mode = "idle") {
  statusPill.textContent = label;
  statusPill.className = `status-pill ${mode}`;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function renderResult(payload) {
  const result = payload.result || payload;
  answerView.innerHTML = "";
  const pre = document.createElement("pre");
  pre.textContent = result.answer || result.message || pretty(result);
  answerView.appendChild(pre);
  planView.textContent = pretty(result.plan || {});
  reflectionView.textContent = pretty(result.reflection || {});
  toolsView.textContent = pretty(result.tool_history || []);
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
    const payload = await postJson("/api/ask", {
      question,
      thread_id: threadId,
      user_id: userId,
    });
    renderResult(payload);
    setStatus("Completed", "done");
  } catch (error) {
    answerView.textContent = error.message;
    setStatus("Error", "error");
  }
});

document.getElementById("daily-btn").addEventListener("click", async () => {
  try {
    setStatus("Running daily workflow...", "loading");
    const payload = await postJson("/api/run-daily", {});
    renderResult(payload);
    setStatus("Daily workflow done", "done");
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
  form.append("paper_id", document.getElementById("paper-id").value.trim());
  form.append("title", document.getElementById("paper-title").value.trim());

  try {
    setStatus("Uploading PDF...", "loading");
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
        answer: "PDF indexed successfully.",
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
