/* =========================================================
   RAG Poisoning Visualizer — dashboard.js
   Vanilla JS state machine + fetch + SSE parser + DOM updates.

   Caching contract:
     - `ts` is the conversation token, generated ONLY on Run Experiment
       button press (lazy). Query/poison form changes don't bump ts.
     - faissResult bound to ts (single slot); rerankCache &
       genCache keyed by (ts, reranker).

   Database lifecycle (separate from the .pv-stage 3-state machine):
     none → [first build]   warming (blue shimmer)        → clean (default blue)
                             ↓ (lazy /api/init/stream)
     clean → [inject poison] injecting (red shimmer)      → poisoned (amber static)
     poisoned → [reranker reads quietly, Database static]
     poisoned → [remove]     rebuilding (green shimmer)   → clean (default blue)

   Database top-right has a 3-row timer block, each row independently
   hidden unless the client has observed that timing event.
   ========================================================= */

(function () {
  "use strict";

  // ============================================================
  // Server-pushed initial state
  // ============================================================
  const META = JSON.parse(
    document.getElementById("meta-data").textContent
  );
  window.__META__ = META;

  // ============================================================
  // Mutable client state
  // ============================================================
  const state = {
    currentRun: null,    // { ts, controller }
    faissResult: null,   // { ts, query, poison, top_k1_clean, top_k1_poisoned, metrics_k1 }
    rerankCache: new Map(),  // `${ts}|${reranker}` → { top_k2_clean, top_k2_poisoned, metrics_k2, elapsed }
    genCache: new Map(),     // `${ts}|${reranker}` → { clean, poisoned, elapsed }
    initialized: false,  // true after /api/init/stream done event
  };

  function cacheKey(ts, reranker) { return `${ts}|${reranker}`; }

  // ============================================================
  // DOM refs
  // ============================================================
  const $ = (id) => document.getElementById(id);
  const queryInput      = $("query");
  const poisonSelect    = $("poison");
  const poisonCylNum    = $("poison-cyl-num");
  const rerankerSelect  = $("reranker");
  const runButton       = $("run-button");
  const stage3Toggle    = $("stage3-toggle");
  const errorBanner     = $("error-banner");
  const skippedHint     = $("stage_generate-skipped-hint");
  const generatorCols   = $("stage_generate-cmpcols");
  const generatorClean  = $("stage_generate-clean");
  const generatorPoison = $("stage_generate-poisoned");

  // For Database cylinder number patching after init.
  // Two cylinders in Static-S sub-frame (background, base) — selected
  // by order of appearance in DOM. The Poison cylinder uses a separate
  // id (#poison-cyl-num) and is not in this list.
  const staticCylNums = (() => {
    const all = document.querySelectorAll(".kb-sub.static-sub .cyl-num");
    return { background: all[0], base: all[1] };
  })();

  const STAGE_IDS = [
    "stage_build",
    "stage_embed",
    "stage_search",
    "stage_rerank",
    "stage_generate",
  ];

  const HINT_OFF =
    "Generation skipped — turn the switch above on to view.";
  const HINT_AWAITING_RUN =
    "Press \"Run Experiment\" first — generator will run automatically.";

  // ============================================================
  // Helpers
  // ============================================================
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function poisonShortLabel(key) {
    const stripped = key.startsWith("P_") ? key.slice(2) : key;
    return stripped.slice(0, 6) + "…";
  }

  function newTs() {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
  }

  function showError(msg) {
    errorBanner.textContent = "Error: " + msg;
    errorBanner.classList.remove("hidden");
  }
  function hideError() { errorBanner.classList.add("hidden"); }

  function showSkippedHint() {
    skippedHint.style.display = "";
    generatorCols.hidden = true;
  }
  function showGeneratorCols() {
    skippedHint.style.display = "none";
    generatorCols.hidden = false;
  }

  // Update one of the Database timer rows. Reveals the row on first
  // write; subsequent writes just refresh the number.
  function updateTimerRow(rowBaseId, elapsed) {
    const row = $(rowBaseId);
    const value = $(`${rowBaseId}-value`);
    if (!row || !value) return;
    row.hidden = false;
    value.textContent = `${elapsed.toFixed(2)}s`;
  }

  // Database lifecycle class switching. Strips all lifecycle classes
  // first so we never end up with two simultaneously.
  function setDatabaseLifecycle(phase, newState) {
    const dbEl = document.querySelector('[data-stage-id="stage_build"]');
    if (!dbEl) return;
    dbEl.classList.remove(
      "stage-pending", "stage-running", "stage-complete",
      "stage-warming", "stage-injecting", "stage-poisoned",
      "stage-rebuilding",
    );
    if (newState === "running") {
      if (phase === "init")   dbEl.classList.add("stage-warming");
      if (phase === "inject") dbEl.classList.add("stage-injecting");
      if (phase === "rebuild") dbEl.classList.add("stage-rebuilding");
    } else if (newState === "complete") {
      // After inject: index is poisoned → amber static persists
      // through search-poisoned + the entire reranker stage.
      // After init / rebuild: clean → no class (default blue tint).
      if (phase === "inject") dbEl.classList.add("stage-poisoned");
    }
  }

  // ----- Stage frame state machine -----
  function setStageState(stageId, newState, elapsed, phase) {
    // Database (stage_build) uses its own lifecycle classes + timer
    // block, not the generic .pv-stage three-state.
    if (stageId === "stage_build") {
      setDatabaseLifecycle(phase, newState);
      if (newState === "complete" && typeof elapsed === "number") {
        if (phase === "inject")  updateTimerRow("db-timer-inject", elapsed);
        if (phase === "rebuild") updateTimerRow("db-timer-rebuild", elapsed);
      }
      return;
    }

    const el = document.querySelector(`[data-stage-id="${stageId}"]`);
    if (!el) return;
    const wasRunning = el.classList.contains("stage-running");
    el.classList.remove(
      "stage-pending", "stage-running", "stage-complete",
    );
    el.classList.add(`stage-${newState}`);

    // Three-state body line on stage_embed.
    if (stageId === "stage_embed") {
      const body = $("stage_embed-body");
      if (body) {
        body.classList.toggle("pending", newState === "pending");
        if (newState === "running") {
          body.innerHTML = "&#8635; encoding query…";
        } else if (newState === "complete") {
          body.innerHTML = "&#10003; query &rarr; 384-dim vector";
        } else {
          body.innerHTML = "pending &mdash; waits for query";
        }
      }
    }

    if (newState === "running" && !wasRunning) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    const corner = el.querySelector(".pv-stage-corner");
    if (!corner) return;
    if (newState === "running") {
      corner.innerHTML = "&#8635; running…";
    } else if (newState === "complete") {
      const secs = typeof elapsed === "number" ? elapsed.toFixed(2) : "—";
      corner.innerHTML = `&#10003; ${secs}s`;
    } else {
      corner.innerHTML = "";
    }
  }

  function resetAllStages() {
    // Database is NOT part of resetAllStages — its lifecycle is driven
    // separately by setDatabaseLifecycle, and a new Run begins with
    // Database already clean. We do clear the per-Run timer rows so
    // stale numbers from a prior Run don't linger.
    ["stage_embed", "stage_search", "stage_rerank", "stage_generate"]
      .forEach((id) => setStageState(id, "pending"));
    ["stage_search-grid", "stage_rerank-grid"].forEach((id) => {
      const grid = $(id);
      if (!grid) return;
      while (grid.children.length > 2) grid.removeChild(grid.lastChild);
    });
    showSkippedHint();
    skippedHint.textContent = HINT_OFF;
    generatorClean.textContent = "";
    generatorPoison.textContent = "";
    hideError();
  }

  function resetRerankAndGenerate() {
    const grid = $("stage_rerank-grid");
    if (grid) {
      while (grid.children.length > 2) grid.removeChild(grid.lastChild);
    }
    setStageState("stage_rerank", "pending");
    setStageState("stage_generate", "pending");
    generatorCols.hidden = true;
    skippedHint.style.display = "";
    skippedHint.textContent = stage3Toggle.checked
      ? HINT_AWAITING_RUN : HINT_OFF;
    generatorClean.textContent = "";
    generatorPoison.textContent = "";
    hideError();
  }

  // ----- Top-k row rendering -----
  function rowClass(r) {
    if (r.is_poison) return "poison";
    if (r.source === "msmarco") return "bg";
    return "base";
  }
  function rowBadge(r) {
    if (r.is_poison) return "&#9763; POISON";
    if (r.source === "msmarco") return "BG";
    return "BASE";
  }
  function rowHtml(r) {
    const cls = rowClass(r);
    return (
      `<div class="disp-item ${cls}">` +
      `<div class="di-line1"><b>${r.rank}.</b> ${escapeHtml(r.title)}</div>` +
      `<div class="di-line2">` +
      `<span class="di-badge">${rowBadge(r)}</span>` +
      `<small>score: ${r.score.toFixed(3)}</small>` +
      `</div>` +
      `</div>`
    );
  }
  function renderTopK(stageId, cleanList, poisonedList) {
    const grid = $(`${stageId}-grid`);
    if (!grid) return;
    while (grid.children.length > 2) grid.removeChild(grid.lastChild);
    const k = Math.max(cleanList.length, poisonedList.length);
    const html = [];
    for (let i = 0; i < k; i++) {
      html.push(cleanList[i] ? rowHtml(cleanList[i]) : "<div></div>");
      html.push(poisonedList[i] ? rowHtml(poisonedList[i]) : "<div></div>");
    }
    grid.insertAdjacentHTML("beforeend", html.join(""));
  }

  // ============================================================
  // Generic SSE-parsing loop. `onEvent(ev)` decides what to do.
  // ts validation is the caller's responsibility (init has no ts).
  // ============================================================
  async function streamSse(url, payload, signal, onEvent) {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload == null ? undefined : JSON.stringify(payload),
      signal,
    });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status} ${resp.statusText}`);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop();
      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        let event;
        try {
          event = JSON.parse(line.slice(6));
        } catch (e) {
          console.warn("Malformed SSE frame:", frame);
          continue;
        }
        onEvent(event);
      }
    }
  }

  // ============================================================
  // Lazy init: POST /api/init/stream on page load.
  // Database shimmers blue while building; Run button is disabled.
  // On done(elapsed != null) → record first-build time, fill cylinders.
  // On done(elapsed === null) → server already initialized (page
  // reload after init); don't show the first-build row.
  // ============================================================
  async function initPipeline() {
    runButton.disabled = true;
    setDatabaseLifecycle("init", "running");
    hideError();

    try {
      await streamSse("/api/init/stream", null, undefined, (ev) => {
        if (ev.type === "stage" && ev.state === "complete") {
          // Building done — leave shimmer on until done event so the
          // cylinder numbers populate before the user looks back at
          // the Database.
          setDatabaseLifecycle("init", "complete");
        } else if (ev.type === "done") {
          // Patch cylinder numbers from server-supplied meta.
          if (ev.meta) {
            if (staticCylNums.background && ev.meta.n_background != null) {
              staticCylNums.background.textContent = ev.meta.n_background;
            }
            if (staticCylNums.base && ev.meta.n_base != null) {
              staticCylNums.base.textContent = ev.meta.n_base;
            }
          }
          // first build row visibility: only show if THIS client
          // observed the actual build (elapsed != null).
          if (typeof ev.elapsed === "number") {
            updateTimerRow("db-timer-first-build", ev.elapsed);
          }
          setDatabaseLifecycle("init", "complete");
          state.initialized = true;
          runButton.disabled = false;
        } else if (ev.type === "error") {
          showError("Init failed: " + (ev.message || "unknown"));
          // Leave Run button disabled — server isn't ready.
        }
      });
    } catch (err) {
      showError("Init failed: " + err.message);
    }
  }

  // ============================================================
  // Run button → /api/run/stream (full pipeline)
  // ============================================================
  async function runExperiment() {
    if (state.currentRun) {
      state.currentRun.controller.abort();
      state.currentRun = null;
    }

    const query = queryInput.value.trim();
    if (!query) {
      showError("Query is empty.");
      return;
    }
    const poison = poisonSelect.value;
    const reranker = rerankerSelect.value;
    const ts = newTs();

    // Wipe Run-scoped caches; Database lifecycle is NOT touched here —
    // it goes "default → injecting → poisoned → rebuilding → default"
    // driven by the SSE events.
    state.faissResult = null;
    state.rerankCache.clear();
    state.genCache.clear();
    resetAllStages();
    setStageState("stage_embed", "running");

    const controller = new AbortController();
    state.currentRun = { ts, controller };
    runButton.disabled = true;

    try {
      await streamSse(
        "/api/run/stream",
        { query, poison, reranker, ts },
        controller.signal,
        (ev) => {
          if (ev.ts !== ts) return;  // stale event guard
          handleRunSseEvent(ev, { ts, query, poison, reranker });
        },
      );
    } catch (err) {
      if (err.name !== "AbortError") showError(err.message);
    } finally {
      if (state.currentRun?.ts === ts) state.currentRun = null;
      runButton.disabled = false;
    }
  }

  function handleRunSseEvent(ev, ctx) {
    if (ev.type === "stage") {
      if (ev.state === "running") {
        setStageState(ev.stage_id, "running", undefined, ev.phase);
      } else if (ev.state === "complete") {
        setStageState(ev.stage_id, "complete", ev.elapsed, ev.phase);
        if (ev.stage_id === "stage_search" && ev.data) {
          state.faissResult = {
            ts: ctx.ts,
            query: ctx.query,
            poison: ctx.poison,
            top_k1_clean: ev.data.clean,
            top_k1_poisoned: ev.data.poisoned,
          };
          renderTopK("stage_search", ev.data.clean, ev.data.poisoned);
        } else if (ev.stage_id === "stage_rerank" && ev.data) {
          state.rerankCache.set(cacheKey(ctx.ts, ctx.reranker), {
            top_k2_clean: ev.data.clean,
            top_k2_poisoned: ev.data.poisoned,
            elapsed: ev.elapsed,
          });
          renderTopK("stage_rerank", ev.data.clean, ev.data.poisoned);
        }
      }
    } else if (ev.type === "done") {
      if (state.faissResult && ev.metrics_k1) {
        state.faissResult.metrics_k1 = ev.metrics_k1;
      }
      const rerankEntry = state.rerankCache.get(
        cacheKey(ctx.ts, ctx.reranker)
      );
      if (rerankEntry && ev.metrics_k2) {
        rerankEntry.metrics_k2 = ev.metrics_k2;
      }
      if (stage3Toggle.checked) runGenerate();
    } else if (ev.type === "error") {
      showError(ev.message || "Unknown error");
    }
  }

  // ============================================================
  // Reranker change → /api/rerank/stream (partial) or cache hit
  // ============================================================
  async function applyReranker() {
    if (!state.faissResult) return;

    const ts = state.faissResult.ts;
    const reranker = rerankerSelect.value;
    const key = cacheKey(ts, reranker);

    if (state.rerankCache.has(key)) {
      const cached = state.rerankCache.get(key);
      renderTopK("stage_rerank", cached.top_k2_clean, cached.top_k2_poisoned);
      setStageState("stage_rerank", "complete", cached.elapsed);
      if (stage3Toggle.checked) runGenerate();
      else {
        setStageState("stage_generate", "pending");
        showSkippedHint();
      }
      return;
    }

    if (state.currentRun) {
      state.currentRun.controller.abort();
      state.currentRun = null;
    }
    const grid = $("stage_rerank-grid");
    if (grid) while (grid.children.length > 2) grid.removeChild(grid.lastChild);
    setStageState("stage_rerank", "running");
    setStageState("stage_generate", "pending");
    showSkippedHint();
    if (stage3Toggle.checked) skippedHint.textContent = HINT_AWAITING_RUN;
    hideError();

    const controller = new AbortController();
    state.currentRun = { ts, controller };
    runButton.disabled = true;

    try {
      await streamSse(
        "/api/rerank/stream",
        {
          query: state.faissResult.query,
          top_k1_clean: state.faissResult.top_k1_clean,
          top_k1_poisoned: state.faissResult.top_k1_poisoned,
          reranker,
          ts,
        },
        controller.signal,
        (ev) => {
          if (ev.ts !== ts) return;
          handleRerankSseEvent(ev, ts, reranker);
        },
      );
    } catch (err) {
      if (err.name !== "AbortError") showError(err.message);
    } finally {
      if (state.currentRun?.ts === ts) state.currentRun = null;
      runButton.disabled = false;
    }
  }

  function handleRerankSseEvent(ev, ts, reranker) {
    if (ev.type === "stage") {
      if (ev.state === "running") {
        setStageState(ev.stage_id, "running", undefined, ev.phase);
      } else if (ev.state === "complete") {
        setStageState(ev.stage_id, "complete", ev.elapsed, ev.phase);
        if (ev.stage_id === "stage_rerank" && ev.data) {
          state.rerankCache.set(cacheKey(ts, reranker), {
            top_k2_clean: ev.data.clean,
            top_k2_poisoned: ev.data.poisoned,
            elapsed: ev.elapsed,
          });
          renderTopK("stage_rerank", ev.data.clean, ev.data.poisoned);
        }
      }
    } else if (ev.type === "done") {
      const entry = state.rerankCache.get(cacheKey(ts, reranker));
      if (entry && ev.metrics_k2) entry.metrics_k2 = ev.metrics_k2;
      if (stage3Toggle.checked) runGenerate();
    } else if (ev.type === "error") {
      showError(ev.message || "Unknown error");
    }
  }

  // ============================================================
  // Stage 3 toggle / Run-done → /api/generate (cached by (ts, reranker))
  // ============================================================
  async function runGenerate() {
    if (!state.faissResult) return;
    const ts = state.faissResult.ts;
    const reranker = rerankerSelect.value;
    const key = cacheKey(ts, reranker);

    const rerankEntry = state.rerankCache.get(key);
    if (!rerankEntry) return;  // rerank not ready yet

    setStageState("stage_generate", "running");
    showGeneratorCols();
    generatorClean.textContent = "";
    generatorPoison.textContent = "";

    if (state.genCache.has(key)) {
      const cached = state.genCache.get(key);
      generatorClean.textContent = cached.clean;
      generatorPoison.textContent = cached.poisoned;
      setStageState("stage_generate", "complete", cached.elapsed);
      return;
    }

    try {
      const resp = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: state.faissResult.query,
          top_k2_clean: rerankEntry.top_k2_clean,
          top_k2_poisoned: rerankEntry.top_k2_poisoned,
          ts,
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status} ${resp.statusText}`);
      const data = await resp.json();
      if (!state.faissResult || data.ts !== state.faissResult.ts) return;
      state.genCache.set(key, data);
      generatorClean.textContent = data.clean;
      generatorPoison.textContent = data.poisoned;
      setStageState("stage_generate", "complete", data.elapsed);
    } catch (err) {
      showError(`Generator failed: ${err.message}`);
      setStageState("stage_generate", "pending");
    }
  }

  // ============================================================
  // Event wiring
  // ============================================================
  poisonSelect.addEventListener("change", () => {
    poisonCylNum.textContent = poisonShortLabel(poisonSelect.value);
  });

  rerankerSelect.addEventListener("change", () => {
    applyReranker();
  });

  runButton.addEventListener("click", runExperiment);

  stage3Toggle.addEventListener("change", () => {
    if (stage3Toggle.checked) {
      if (state.faissResult &&
          state.rerankCache.has(cacheKey(state.faissResult.ts,
                                          rerankerSelect.value))) {
        runGenerate();
      } else {
        skippedHint.textContent = HINT_AWAITING_RUN;
      }
    } else {
      skippedHint.textContent = HINT_OFF;
      showSkippedHint();
      setStageState("stage_generate", "pending");
    }
  });

  // ============================================================
  // Page-load: trigger lazy init (Database shimmers, Run locked).
  // ============================================================
  initPipeline();

  console.info(
    "dashboard.js MVP4+ loaded — lazy init, ts-scoped caches, " +
    "Database lifecycle (warming/injecting/poisoned/rebuilding). " +
    `${META.poison_sets.length} poison sets, ${META.rerankers.length} rerankers.`
  );
})();
