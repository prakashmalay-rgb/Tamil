/**
 * Application Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  // State
  let currentModel = "tamil"; // 'tamil' | 'gpt' | 'arena'
  let isGenerating = false;
  let conversations = JSON.parse(localStorage.getItem("tamil_chat_history") || "[]");
  let activeChatId = null;
  let sessionMessages = [];

  // DOM Elements
  const chatViewport = document.getElementById("chatViewport");
  const welcomeHero = document.getElementById("welcomeHero");
  const chatInput = document.getElementById("chatInput");
  const btnSend = document.getElementById("btnSend");
  const modelTabs = document.querySelectorAll(".model-tab");
  const btnNewChat = document.getElementById("btnNewChat");
  const historyList = document.getElementById("historyList");
  const quickPrompts = document.querySelectorAll(".prompt-chip");
  const settingsBtn = document.getElementById("settingsBtn");
  const settingsModal = document.getElementById("settingsModal");
  const closeSettingsBtn = document.getElementById("closeSettingsBtn");
  const saveSettingsBtn = document.getElementById("saveSettingsBtn");
  const sidebarToggle = document.getElementById("sidebarToggle");
  const appSidebar = document.getElementById("appSidebar");

  // Load Settings into modal inputs
  function loadSettings() {
    document.getElementById("kaggleUrlInput").value = ApiService.getKaggleUrl();
    document.getElementById("openAiKeyInput").value = ApiService.getOpenAiKey();
    const params = ApiService.getSamplingParams();
    document.getElementById("tempInput").value = params.temperature;
    document.getElementById("tempValue").innerText = params.temperature;
    document.getElementById("maxTokensInput").value = params.max_tokens;
    document.getElementById("topPInput").value = params.top_p;
    document.getElementById("topPValue").innerText = params.top_p;
  }

  // Model switching
  modelTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      modelTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      currentModel = tab.dataset.model;
      updateViewportLayout();
    });
  });

  function updateViewportLayout() {
    if (currentModel === "arena") {
      chatViewport.classList.add("arena-mode");
    } else {
      chatViewport.classList.remove("arena-mode");
    }
  }

  // Auto-resize textarea
  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + "px";
    btnSend.disabled = chatInput.value.trim().length === 0 || isGenerating;
  });

  // Enter to send (Shift+Enter for newline)
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!btnSend.disabled) sendMessage();
    }
  });

  btnSend.addEventListener("click", sendMessage);

  // Quick prompts
  quickPrompts.forEach(chip => {
    chip.addEventListener("click", () => {
      const promptText = chip.querySelector(".prompt-chip-text").innerText;
      chatInput.value = promptText;
      chatInput.style.height = "auto";
      chatInput.style.height = chatInput.scrollHeight + "px";
      btnSend.disabled = false;
      sendMessage();
    });
  });

  // Send Message Logic
  async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || isGenerating) return;

    if (welcomeHero) welcomeHero.style.display = "none";

    isGenerating = true;
    btnSend.disabled = true;
    chatInput.value = "";
    chatInput.style.height = "auto";

    // 1. Render User Message
    appendUserMessage(text);

    // 2. Dispatch to Model(s)
    if (currentModel === "arena") {
      await handleArenaExecution(text);
    } else if (currentModel === "gpt") {
      await handleSingleModelExecution(text, "gpt");
    } else {
      await handleSingleModelExecution(text, "tamil");
    }

    isGenerating = false;
    btnSend.disabled = false;
    chatInput.focus();
  }

  function appendUserMessage(text) {
    sessionMessages.push({ role: "user", content: text });
    const row = document.createElement("div");
    row.className = "message-row user-row";
    row.innerHTML = `<div class="user-bubble">${SafeMarkdown.escapeHtml(text)}</div>`;
    chatViewport.appendChild(row);
    chatViewport.scrollTop = chatViewport.scrollHeight;
  }

  async function handleSingleModelExecution(prompt, modelType) {
    const isTamil = modelType === "tamil";
    const badgeClass = isTamil ? "badge-tamil" : "badge-gpt";
    const modelTitle = isTamil ? "Tamil Qwen 3.6" : "OpenAI GPT-4o";

    const row = document.createElement("div");
    row.className = "message-row assistant-row";
    row.innerHTML = `
      <div class="assistant-card">
        <div class="assistant-header">
          <div class="model-badge-row">
            <span class="model-badge ${badgeClass}">${modelTitle}</span>
            <span class="metrics-badge metrics-time">Generating...</span>
          </div>
          <button class="btn-action btn-copy" style="display:none;">Copy</button>
        </div>
        <div class="assistant-body"><span class="typing-pulse"></span></div>
      </div>
    `;
    chatViewport.appendChild(row);
    chatViewport.scrollTop = chatViewport.scrollHeight;

    const bodyEl = row.querySelector(".assistant-body");
    const metricsEl = row.querySelector(".metrics-time");
    const copyBtn = row.querySelector(".btn-copy");

    const onToken = (accumulated) => {
      bodyEl.innerHTML = SafeMarkdown.render(accumulated) + '<span class="typing-pulse"></span>';
      chatViewport.scrollTop = chatViewport.scrollHeight;
    };

    let result;
    if (isTamil) {
      result = await ApiService.generateTamilLlm(prompt, onToken, sessionMessages);
    } else {
      result = await ApiService.generateOpenAi(prompt, onToken);
    }

    if (result && result.text) {
      sessionMessages.push({ role: "assistant", content: result.text });
    }

    bodyEl.innerHTML = SafeMarkdown.render(result.text);
    metricsEl.innerText = `${result.durationMs}ms • ~${result.tokens} tokens`;
    copyBtn.style.display = "inline-flex";
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(result.text);
      copyBtn.innerText = "Copied!";
      setTimeout(() => (copyBtn.innerText = "Copy"), 1500);
    };
  }

  async function handleArenaExecution(prompt) {
    const row = document.createElement("div");
    row.className = "message-row arena-row";
    row.innerHTML = `
      <div class="arena-viewport">
        <!-- Tamil Qwen 3.6 Column -->
        <div class="arena-card-column">
          <div class="arena-column-header">
            <span class="model-badge badge-tamil">Tamil Qwen 3.6 (Kaggle GPU)</span>
            <span class="metrics-badge tamil-metrics">Pending...</span>
          </div>
          <div class="arena-content-area tamil-body"><span class="typing-pulse"></span></div>
        </div>
        <!-- GPT-4o Column -->
        <div class="arena-card-column">
          <div class="arena-column-header">
            <span class="model-badge badge-gpt">OpenAI GPT-4o</span>
            <span class="metrics-badge gpt-metrics">Pending...</span>
          </div>
          <div class="arena-content-area gpt-body"><span class="typing-pulse"></span></div>
        </div>
      </div>
    `;
    chatViewport.appendChild(row);
    chatViewport.scrollTop = chatViewport.scrollHeight;

    const tamilBody = row.querySelector(".tamil-body");
    const gptBody = row.querySelector(".gpt-body");
    const tamilMetrics = row.querySelector(".tamil-metrics");
    const gptMetrics = row.querySelector(".gpt-metrics");

    // Execute concurrently
    const tamilPromise = ApiService.generateTamilLlm(prompt, (tokens) => {
      tamilBody.innerHTML = SafeMarkdown.render(tokens) + '<span class="typing-pulse"></span>';
    }).then(res => {
      tamilBody.innerHTML = SafeMarkdown.render(res.text);
      tamilMetrics.innerText = `${res.durationMs}ms • ${res.tokens} tok`;
      return res;
    });

    const gptPromise = ApiService.generateOpenAi(prompt, (tokens) => {
      gptBody.innerHTML = SafeMarkdown.render(tokens) + '<span class="typing-pulse"></span>';
    }).then(res => {
      gptBody.innerHTML = SafeMarkdown.render(res.text);
      gptMetrics.innerText = `${res.durationMs}ms • ${res.tokens} tok`;
      return res;
    });

    await Promise.allSettled([tamilPromise, gptPromise]);
  }

  // Settings Modal Controls
  settingsBtn.addEventListener("click", () => {
    loadSettings();
    settingsModal.classList.add("open");
  });

  closeSettingsBtn.addEventListener("click", () => {
    settingsModal.classList.remove("open");
  });

  saveSettingsBtn.addEventListener("click", () => {
    localStorage.setItem("kaggle_api_url", document.getElementById("kaggleUrlInput").value.trim());
    localStorage.setItem("openai_api_key", document.getElementById("openAiKeyInput").value.trim());
    localStorage.setItem("sampling_temperature", document.getElementById("tempInput").value);
    localStorage.setItem("sampling_max_tokens", document.getElementById("maxTokensInput").value);
    localStorage.setItem("sampling_top_p", document.getElementById("topPInput").value);
    settingsModal.classList.remove("open");
  });

  // Slider visual updates
  document.getElementById("tempInput").addEventListener("input", (e) => {
    document.getElementById("tempValue").innerText = e.target.value;
  });
  document.getElementById("topPInput").addEventListener("input", (e) => {
    document.getElementById("topPValue").innerText = e.target.value;
  });

  // New chat button
  btnNewChat.addEventListener("click", () => {
    chatViewport.innerHTML = "";
    sessionMessages = [];
    if (welcomeHero) {
      chatViewport.appendChild(welcomeHero);
      welcomeHero.style.display = "block";
    }
  });

  // Sidebar toggle for mobile/compact screens
  if (sidebarToggle && appSidebar) {
    sidebarToggle.addEventListener("click", () => {
      appSidebar.classList.toggle("open");
    });
  }
});
