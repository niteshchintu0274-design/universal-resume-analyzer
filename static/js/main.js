(function () {
  "use strict";

  const state = {
    chatHistory: [],
    previewUrl: ""
  };

  document.addEventListener("DOMContentLoaded", () => {
    initIcons();
    initTheme();
    initLanguage();
    initRevealAnimations();
    initUpload();
    initProgressBars();
    initScoreChart();
    initPrivateToggle();
    initPreviewTools();
    initRewriteButtons();
    initCopyButtons();
    initJobMatch();
    initJobFilters();
    initChat();
  });

  function byId(id) {
    return document.getElementById(id);
  }

  function initIcons() {
    if (window.lucide) {
      window.lucide.createIcons();
    }
  }

  function initTheme() {
    const button = byId("themeToggle");
    if (!button) return;
    button.addEventListener("click", () => {
      document.documentElement.classList.toggle("dark");
      localStorage.setItem("resumeAnalyzerTheme", document.documentElement.classList.contains("dark") ? "dark" : "light");
      initIcons();
    });
  }

  function initLanguage() {
    const select = byId("languageToggle");
    if (!select) return;
    const saved = localStorage.getItem("resumeAnalyzerLanguage") || "en";
    select.value = saved;
    applyLanguage(saved);
    select.addEventListener("change", () => {
      localStorage.setItem("resumeAnalyzerLanguage", select.value);
      applyLanguage(select.value);
    });
  }

  function applyLanguage(language) {
    document.documentElement.lang = language;
    document.querySelectorAll("[data-en][data-hi]").forEach((element) => {
      element.textContent = language === "hi" ? element.dataset.hi : element.dataset.en;
    });
  }

  function initRevealAnimations() {
    document.querySelectorAll("main section, main article").forEach((element, index) => {
      element.classList.add("reveal-on-load");
      element.style.animationDelay = `${Math.min(index * 25, 180)}ms`;
    });
  }

  function initProgressBars() {
    document.querySelectorAll("[data-progress]").forEach((bar) => {
      const value = clamp(Number.parseFloat(bar.dataset.progress || "0"), 0, 100);
      requestAnimationFrame(() => {
        bar.style.width = `${value}%`;
      });
    });
  }

  function initUpload() {
    const form = byId("resumeForm");
    const dropZone = byId("dropZone");
    const fileInput = byId("resumeInput");
    const fileName = byId("fileName");
    const submitButton = byId("submitButton");
    const loadingOverlay = byId("loadingOverlay");
    const previewWrap = byId("pdfPreviewWrap");
    const preview = byId("pdfPreview");
    const clearPreviewButton = byId("clearPreviewButton");

    if (!form || !dropZone || !fileInput || !fileName) return;

    const preventDefaults = (event) => {
      event.preventDefault();
      event.stopPropagation();
    };

    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        preventDefaults(event);
        dropZone.classList.add("drop-zone-active");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        preventDefaults(event);
        dropZone.classList.remove("drop-zone-active");
      });
    });

    dropZone.addEventListener("drop", (event) => {
      const file = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files[0] : null;
      if (!file) return;
      if (!isPdf(file)) {
        showFileMessage(fileName, "Please choose a PDF file.", "error");
        return;
      }
      if (!assignDroppedFile(fileInput, file)) {
        showFileMessage(fileName, "Use the file picker for this browser.", "error");
        return;
      }
      handleSelectedFile(file, fileName, previewWrap, preview);
    });

    fileInput.addEventListener("change", () => {
      const file = fileInput.files ? fileInput.files[0] : null;
      if (!file) return;
      if (!isPdf(file)) {
        fileInput.value = "";
        showFileMessage(fileName, "Please choose a PDF file.", "error");
        return;
      }
      handleSelectedFile(file, fileName, previewWrap, preview);
    });

    if (clearPreviewButton) {
      clearPreviewButton.addEventListener("click", () => {
        fileInput.value = "";
        fileName.classList.add("hidden");
        hidePreview(previewWrap, preview);
      });
    }

    form.addEventListener("submit", () => {
      if (loadingOverlay) {
        loadingOverlay.classList.remove("hidden");
        loadingOverlay.classList.add("grid");
      }
      if (submitButton) {
        submitButton.disabled = true;
        submitButton.classList.add("opacity-80");
      }
    });
  }

  function handleSelectedFile(file, fileName, previewWrap, preview) {
    showFileMessage(fileName, file.name, "success");
    if (!previewWrap || !preview) return;
    hidePreview(previewWrap, preview);
    state.previewUrl = URL.createObjectURL(file);
    preview.setAttribute("data", state.previewUrl);
    previewWrap.classList.remove("hidden");
  }

  function hidePreview(previewWrap, preview) {
    if (state.previewUrl) {
      URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = "";
    }
    if (preview) preview.setAttribute("data", "");
    if (previewWrap) previewWrap.classList.add("hidden");
  }

  function isPdf(file) {
    return file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
  }

  function assignDroppedFile(fileInput, file) {
    if (window.DataTransfer) {
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      fileInput.files = dataTransfer.files;
      return true;
    }
    return false;
  }

  function showFileMessage(fileName, message, status) {
    fileName.textContent = message;
    fileName.classList.remove("hidden", "file-chip-success", "file-chip-error");
    fileName.classList.add(status === "error" ? "file-chip-error" : "file-chip-success");
  }

  function initScoreChart() {
    const canvas = byId("scoreChart");
    const scoreData = window.resumeAnalyzerScores;
    if (!canvas || !window.Chart || !scoreData) return;
    const entries = Object.values(scoreData);
    const labels = entries.map((item) => item.label.replace(" Score", ""));
    const values = entries.map((item) => Number(item.percentage || item.points || 0));
    new Chart(canvas, {
      type: "radar",
      data: {
        labels,
        datasets: [
          {
            label: "Resume score",
            data: values,
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.18)",
            pointBackgroundColor: "#0891b2"
          }
        ]
      },
      options: {
        responsive: true,
        scales: {
          r: {
            beginAtZero: true,
            max: 100,
            ticks: { display: false },
            grid: { color: "rgba(148, 163, 184, 0.28)" },
            angleLines: { color: "rgba(148, 163, 184, 0.28)" },
            pointLabels: { color: document.documentElement.classList.contains("dark") ? "#cbd5e1" : "#334155", font: { size: 11 } }
          }
        },
        plugins: { legend: { display: false } }
      }
    });
  }

  function initPrivateToggle() {
    const button = byId("togglePrivateInfo");
    if (!button) return;
    let revealed = false;
    button.addEventListener("click", () => {
      revealed = !revealed;
      document.querySelectorAll(".sensitive-value").forEach((element) => {
        element.textContent = revealed ? element.dataset.full || element.textContent : element.dataset.masked || element.textContent;
      });
      button.innerHTML = revealed
        ? '<i data-lucide="eye-off" class="h-3.5 w-3.5"></i>Hide'
        : '<i data-lucide="eye" class="h-3.5 w-3.5"></i>Reveal';
      initIcons();
    });
  }

  function initPreviewTools() {
    const preview = byId("resumePreview");
    const masked = byId("maskedPreviewLines");
    const raw = byId("rawPreviewLines");
    const maskButton = byId("togglePreviewMask");
    const zoomIn = byId("previewZoomIn");
    const zoomOut = byId("previewZoomOut");
    if (!preview) return;

    let fontSize = 12;
    let showingRaw = false;

    if (maskButton && masked && raw) {
      maskButton.addEventListener("click", () => {
        showingRaw = !showingRaw;
        masked.classList.toggle("hidden", showingRaw);
        raw.classList.toggle("hidden", !showingRaw);
        maskButton.classList.toggle("bg-brand-50", showingRaw);
      });
    }

    if (zoomIn) {
      zoomIn.addEventListener("click", () => {
        fontSize = clamp(fontSize + 1, 11, 16);
        preview.style.fontSize = `${fontSize}px`;
      });
    }

    if (zoomOut) {
      zoomOut.addEventListener("click", () => {
        fontSize = clamp(fontSize - 1, 11, 16);
        preview.style.fontSize = `${fontSize}px`;
      });
    }
  }

  function initRewriteButtons() {
    document.querySelectorAll(".rewrite-button").forEach((button) => {
      button.addEventListener("click", async () => {
        const article = button.closest("article");
        const output = article ? article.querySelector(".rewrite-output") : null;
        button.disabled = true;
        try {
          const data = await postJson(getEndpoint("rewriteUrl", "/api/rewrite-section"), {
            analysis_id: button.dataset.analysisId,
            section: button.dataset.section,
            before: button.dataset.before
          });
          if (output) {
            output.innerHTML = data.versions.map((version) => renderRewriteVersion(version)).join("");
            initCopyButtons(output);
            initIcons();
        }
      } catch (error) {
          if (output) output.innerHTML = `<div class="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">${escapeHtml(friendlyMessage(error.message, "rewrite"))}</div>`;
      } finally {
          button.disabled = false;
        }
      });
    });
  }

  function renderRewriteVersion(version) {
    return `
      <div class="rounded-md bg-brand-50 p-3 text-sm leading-6 text-slate-700 dark:bg-brand-950/30 dark:text-slate-200">
        <div class="flex items-start justify-between gap-3">
          <p>${escapeHtml(version)}</p>
          <button type="button" class="copy-button shrink-0 rounded-md p-1.5 text-brand-700 hover:bg-brand-100 dark:text-brand-200 dark:hover:bg-brand-900" title="Copy"><i data-lucide="copy" class="h-4 w-4"></i></button>
        </div>
      </div>
    `;
  }

  function initCopyButtons(root) {
    (root || document).querySelectorAll(".copy-button").forEach((button) => {
      if (button.dataset.bound === "1") return;
      button.dataset.bound = "1";
      button.addEventListener("click", async () => {
        const text = button.closest("div").querySelector("p")?.textContent || "";
        await navigator.clipboard.writeText(text);
        button.classList.add("bg-brand-100");
        setTimeout(() => button.classList.remove("bg-brand-100"), 700);
      });
    });
  }

  function initJobMatch() {
    const button = byId("jobMatchButton");
    const textarea = byId("jobMatchDescription");
    const output = byId("jobMatchOutput");
    if (!button || !textarea || !output) return;

    button.addEventListener("click", async () => {
      const jobDescription = textarea.value.trim();
      if (!jobDescription) {
        output.innerHTML = `<div class="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">Paste a job description first.</div>`;
        return;
      }
      button.disabled = true;
      output.innerHTML = `<div class="rounded-md border border-slate-200 p-3 text-sm text-slate-500">Analyzing job match...</div>`;
      try {
        const data = await postJson(getEndpoint("jobMatchUrl", "/api/job-match"), {
          analysis_id: button.dataset.analysisId,
          job_description: jobDescription
        });
        output.innerHTML = renderJobMatch(data.job_match || {});
      } catch (error) {
        output.innerHTML = `<div class="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">${escapeHtml(friendlyMessage(error.message, "jobMatch"))}</div>`;
      } finally {
        button.disabled = false;
      }
    });
  }

  function renderJobMatch(match) {
    return `
      <div class="rounded-md border border-slate-200 p-4 dark:border-slate-800">
        <p class="text-sm font-semibold text-slate-950 dark:text-white">Match Score: ${escapeHtml(match.score || 0)}%</p>
        <p class="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-300">${escapeHtml(match.reason || match.explanation || "")}</p>
        <div class="mt-3 space-y-2">
          <div class="flex flex-wrap items-center gap-2">
            <span class="text-xs font-semibold text-slate-500 dark:text-slate-400">Matched skills</span>
            ${(match.matched || []).slice(0, 12).map((item) => `<span class="rounded-md bg-brand-50 px-2.5 py-1.5 text-xs font-medium text-brand-700 dark:bg-brand-950/40 dark:text-brand-200">${escapeHtml(item)}</span>`).join("")}
          </div>
          <div class="flex flex-wrap items-center gap-2">
            <span class="text-xs font-semibold text-slate-500 dark:text-slate-400">Missing skills</span>
            ${(match.missing || []).slice(0, 10).map((item) => `<span class="rounded-md bg-red-50 px-2.5 py-1.5 text-xs font-medium text-red-700 dark:bg-red-950/40 dark:text-red-200">${escapeHtml(item)}</span>`).join("")}
          </div>
        </div>
        ${(match.suggestions || []).length ? `<ul class="mt-3 space-y-1 text-sm leading-6 text-slate-600 dark:text-slate-300">${match.suggestions.slice(0, 5).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}
      </div>
    `;
  }

  function initJobFilters() {
    const button = byId("refreshJobsButton");
    const form = byId("jobFilters");
    const list = byId("jobsList");
    if (!button || !form || !list) return;

    button.addEventListener("click", async () => {
      const params = new URLSearchParams();
      new FormData(form).forEach((value, key) => {
        if (value) params.append(key, value);
      });
      if (form.querySelector("[name='remote']")?.checked) params.set("remote", "1");
      if (form.querySelector("[name='freshers']")?.checked) params.set("freshers", "1");
      list.innerHTML = `<div class="rounded-md border border-slate-200 p-5 text-sm text-slate-500">Loading jobs...</div>`;
      try {
        const response = await fetch(`/api/jobs/${button.dataset.analysisId}?${params.toString()}`);
        const data = await response.json();
        if (!response.ok || data.ok === false) throw new Error(data.error || "Could not load jobs.");
        list.innerHTML = (data.jobs || []).map(renderJobCard).join("") || `<div class="rounded-md border border-dashed border-slate-200 p-5 text-sm text-slate-500">No jobs found.</div>`;
        initIcons();
      } catch (error) {
        list.innerHTML = `<div class="rounded-md border border-red-200 bg-red-50 p-5 text-sm text-red-700">${escapeHtml(friendlyMessage(error.message, "jobs"))}</div>`;
      }
    });
  }

  function renderJobCard(job) {
    return `
      <article class="job-card rounded-md border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-950">
        <div class="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div class="min-w-0">
            <div class="flex flex-wrap items-center gap-2">
              <h3 class="text-base font-semibold text-slate-950 dark:text-white">${escapeHtml(job.title || "")}</h3>
              ${job.work_mode ? `<span class="rounded-md bg-slate-100 px-2 py-1 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300">${escapeHtml(job.work_mode)}</span>` : ""}
              ${job.source ? `<span class="rounded-md bg-cyanx-50 px-2 py-1 text-xs text-cyanx-600 dark:bg-cyan-950/40 dark:text-cyan-200">${escapeHtml(job.source)}</span>` : ""}
            </div>
            <p class="mt-1 text-sm text-slate-500 dark:text-slate-400">${escapeHtml([job.company, job.location, job.salary].filter(Boolean).join(" | "))}</p>
            <p class="mt-3 line-clamp-2 text-sm leading-6 text-slate-600 dark:text-slate-300">${escapeHtml(job.description || "")}</p>
          </div>
          <div class="w-full shrink-0 md:w-48">
            <div class="flex items-center justify-between text-xs"><span class="text-slate-500">Match</span><span class="font-semibold text-slate-950 dark:text-white">${escapeHtml(job.match_percent || 0)}%</span></div>
            <div class="mt-2 h-2.5 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800"><div class="h-full rounded-full bg-brand-600" style="width: ${Number(job.match_percent || 0)}%;"></div></div>
            ${job.apply_link ? `<a href="${escapeAttribute(job.apply_link)}" target="_blank" rel="noopener" class="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-md bg-brand-600 px-3 py-2 text-xs font-semibold text-white hover:bg-brand-700">Apply <i data-lucide="external-link" class="h-3.5 w-3.5"></i></a>` : ""}
          </div>
        </div>
        <div class="mt-4 space-y-2">
          <div class="flex flex-wrap items-center gap-2">
            <span class="text-xs font-semibold text-slate-500 dark:text-slate-400">Matched skills</span>
            ${(job.matched_skills || []).slice(0, 8).map((skill) => `<span class="rounded-md bg-brand-50 px-2.5 py-1.5 text-xs font-medium text-brand-700 dark:bg-brand-950/40 dark:text-brand-200">${escapeHtml(skill)}</span>`).join("")}
          </div>
          <div class="flex flex-wrap items-center gap-2">
            <span class="text-xs font-semibold text-slate-500 dark:text-slate-400">Missing skills</span>
            ${(job.missing_skills || []).slice(0, 6).map((skill) => `<span class="rounded-md bg-red-50 px-2.5 py-1.5 text-xs font-medium text-red-700 dark:bg-red-950/40 dark:text-red-200">${escapeHtml(skill)}</span>`).join("")}
          </div>
        </div>
      </article>
    `;
  }

  function initChat() {
    const chatForm = byId("chatForm");
    const chatMessages = byId("chatMessages");
    const input = byId("careerQuestionInput");
    if (!chatForm || !chatMessages || !input) return;

    const elements = {
      chatForm,
      chatMessages,
      input,
      sendButton: byId("sendButton"),
      typingIndicator: byId("typingIndicator"),
      clearChatButton: byId("clearChatButton"),
      targetRole: byId("targetRole"),
      skillsInput: byId("skillsInput"),
      resumeContext: byId("resumeContext"),
      jobContext: byId("jobContext"),
      analysisId: byId("analysisId"),
      analysisPicker: byId("analysisPicker")
    };

    chatMessages.classList.add("app-scrollbar");

    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      await handleChatSubmit(elements);
    });

    input.addEventListener("input", () => resizeTextarea(input));
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
      }
    });

    document.querySelectorAll(".quick-prompt").forEach((button) => {
      button.addEventListener("click", () => {
        input.value = button.textContent.trim();
        resizeTextarea(input);
        input.focus();
      });
    });

    if (elements.clearChatButton) {
      elements.clearChatButton.addEventListener("click", () => clearChat(elements));
    }

    if (elements.analysisPicker && elements.analysisId) {
      elements.analysisPicker.addEventListener("change", async () => {
        elements.analysisId.value = elements.analysisPicker.value;
        await loadAnalysisContext(elements);
      });
    }
  }

  async function loadAnalysisContext(elements) {
    const analysisId = getValue(elements.analysisId);
    if (!analysisId) return;
    try {
      const response = await fetch(`/api/analysis/${analysisId}`);
      const data = await response.json();
      if (!response.ok || data.ok === false || !data.analysis) throw new Error(data.error || "Could not load analysis.");
      const analysis = data.analysis;
      if (elements.resumeContext) elements.resumeContext.value = analysis.resume_text || "";
      if (elements.jobContext) elements.jobContext.value = analysis.job_description || "";
      const result = analysis.claude_result || {};
      if (elements.targetRole) elements.targetRole.value = result.profile?.role || "";
      if (elements.skillsInput) elements.skillsInput.value = (result.extracted_skills || []).join(", ");
      state.chatHistory = [];
    } catch (error) {
      addChatMessage(elements.chatMessages, "assistant", friendlyMessage(error.message, "context"));
    }
  }

  async function handleChatSubmit(elements) {
    const message = elements.input.value.trim();
    if (!message) return;

    const historyBeforeMessage = [...state.chatHistory];
    addChatMessage(elements.chatMessages, "user", message);
    state.chatHistory.push({ role: "user", content: message });
    elements.input.value = "";
    resizeTextarea(elements.input);
    setChatBusy(elements, true);

    try {
      const data = await postJson(getEndpoint("careerChatUrl", "/api/career-chat"), {
        message,
        analysis_id: getValue(elements.analysisId),
        target_role: getValue(elements.targetRole),
        skills: getSkills(elements.skillsInput),
        resume_text: getValue(elements.resumeContext),
        job_description: getValue(elements.jobContext),
        chat_history: historyBeforeMessage
      });

      const reply = data.reply || "I could not generate a response.";
      addChatMessage(elements.chatMessages, "assistant", reply);
      state.chatHistory.push({ role: "assistant", content: reply });
    } catch (error) {
      addChatMessage(elements.chatMessages, "assistant", friendlyMessage(error.message, "chat"));
    } finally {
      setChatBusy(elements, false);
    }
  }

  function clearChat(elements) {
    state.chatHistory = [];
    elements.chatMessages.innerHTML = `
      <div class="flex justify-start">
        <div class="chat-bubble chat-bubble-assistant max-w-[86%] rounded-md px-4 py-3 text-sm leading-6 shadow-sm">
          Chat cleared. Ask a new resume question whenever you are ready.
        </div>
      </div>
    `;
  }

  function addChatMessage(container, role, content) {
    const isUser = role === "user";
    const wrapper = document.createElement("div");
    wrapper.className = `flex ${isUser ? "justify-end" : "justify-start"}`;
    wrapper.innerHTML = `
      <div class="chat-bubble ${isUser ? "chat-bubble-user" : "chat-bubble-assistant"} max-w-[86%] rounded-md px-4 py-3 text-sm leading-6 shadow-sm sm:max-w-[72%]">
        <div>${formatMessage(content)}</div>
        <span class="mt-1 block text-right text-[11px] ${isUser ? "text-emerald-800/60" : "text-slate-400"}">${currentTimeLabel()}</span>
      </div>
    `;
    container.appendChild(wrapper);
    scrollToLatest(container);
  }

  function setChatBusy(elements, isBusy) {
    if (elements.sendButton) elements.sendButton.disabled = isBusy;
    if (elements.typingIndicator) elements.typingIndicator.classList.toggle("hidden", !isBusy);
    scrollToLatest(elements.chatMessages);
  }

  async function postJson(url, payload) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) {
      throw new Error(data.error || "Request failed.");
    }
    return data;
  }

  function getEndpoint(name, fallback) {
    return document.body.dataset[name] || fallback;
  }

  function getSkills(input) {
    return getValue(input).split(",").map((skill) => skill.trim()).filter(Boolean);
  }

  function getValue(element) {
    return element && typeof element.value === "string" ? element.value.trim() : "";
  }

  function resizeTextarea(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 128)}px`;
  }

  function currentTimeLabel() {
    return new Date().toLocaleString("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit"
    });
  }

  function scrollToLatest(container) {
    if (container) container.scrollTop = container.scrollHeight;
  }

  function formatMessage(text) {
    return escapeHtml(text)
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br>");
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function escapeAttribute(value) {
    return escapeHtml(value).replaceAll("`", "&#096;");
  }

  function friendlyMessage(message, context) {
    const text = String(message || "").toLowerCase();
    if (text.includes("api key") || text.includes("traceback") || text.includes("httpx") || text.includes("request failed")) {
      if (context === "jobs") return "Job matches are in demo mode right now. Try adjusting the filters.";
      if (context === "chat") return "I could not answer that just now. Please try again.";
      if (context === "rewrite") return "Could not regenerate this section right now. Please try again.";
      return "This request could not be completed right now. Please try again.";
    }
    return message || "This request could not be completed right now. Please try again.";
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }
})();
