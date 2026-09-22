/**
 * API Service for Tamil LLM Qwen 3.6 (Kaggle Backend) and OpenAI GPT
 */

const ApiService = {
  getKaggleUrl() {
    return localStorage.getItem("kaggle_api_url") || "http://localhost:8000";
  },

  getOpenAiKey() {
    return localStorage.getItem("openai_api_key") || "";
  },

  getSamplingParams() {
    return {
      temperature: parseFloat(localStorage.getItem("sampling_temperature") || "0.7"),
      max_tokens: parseInt(localStorage.getItem("sampling_max_tokens") || "100", 10),
      top_p: parseFloat(localStorage.getItem("sampling_top_p") || "0.9")
    };
  },

  /**
   * Send prompt to Tamil Qwen 3.6 LLM Backend
   */
  async generateTamilLlm(prompt, onToken) {
    const startTime = performance.now();
    const kaggleUrl = this.getKaggleUrl();
    const params = this.getSamplingParams();

    // Immediate user feedback so they know Kaggle GPU is computing
    if (onToken) {
      onToken("*⚡ Qwen 3.6 is thinking on Kaggle Tesla T4 GPU...*");
    }

    try {
      const response = await fetch(`${kaggleUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt,
          temperature: params.temperature,
          max_tokens: params.max_tokens,
          top_p: params.top_p
        })
      });

      if (!response.ok) {
        throw new Error(`Server responded with HTTP ${response.status}`);
      }

      const data = await response.json();
      const outputText = data.response || data.text || JSON.stringify(data);
      const durationMs = Math.round(performance.now() - startTime);

      // Smooth streaming output into UI
      if (onToken) {
        await this.simulateStream(outputText, onToken);
      }

      return {
        text: outputText,
        durationMs,
        tokens: Math.max(1, Math.round(outputText.length / 3.5)),
        model: "Tamil Qwen 3.6"
      };
    } catch (err) {
      console.warn("Kaggle bridge error:", err);
      const fallbackText = `வணக்கம்! உங்கள் வினவல்: "${prompt}"\n\n(Qwen 3.6 GPU notice: ${err.message})`;
      const durationMs = Math.round(performance.now() - startTime);
      if (onToken) await this.simulateStream(fallbackText, onToken);
      return {
        text: fallbackText,
        durationMs,
        tokens: Math.round(fallbackText.length / 3.5),
        model: "Tamil Qwen 3.6 (Standby)"
      };
    }
  },

  /**
   * Send prompt to OpenAI GPT-4o
   */
  async generateOpenAi(prompt, onToken) {
    const startTime = performance.now();
    const apiKey = this.getOpenAiKey();
    const params = this.getSamplingParams();

    if (!apiKey) {
      const msg = "OpenAI API Key is not configured. Click Settings (gear icon in sidebar) to add your OpenAI key for GPT comparison.";
      if (onToken) onToken(msg);
      return {
        text: msg,
        durationMs: 0,
        tokens: 0,
        model: "GPT-4o (No API Key)"
      };
    }

    try {
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: "gpt-4o",
          messages: [
            { role: "system", content: "You are a helpful, expert AI assistant fluent in Tamil, Tanglish, and English." },
            { role: "user", content: prompt }
          ],
          temperature: params.temperature,
          max_tokens: params.max_tokens,
          stream: true
        })
      });

      if (!response.ok) {
        throw new Error(`OpenAI API returned status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let fullText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");
        for (const line of lines) {
          if (line.startsWith("data: ") && line !== "data: [DONE]") {
            try {
              const parsed = JSON.parse(line.substring(6));
              const token = parsed.choices[0]?.delta?.content || "";
              fullText += token;
              if (onToken) onToken(fullText);
            } catch (e) {}
          }
        }
      }

      const durationMs = Math.round(performance.now() - startTime);
      return {
        text: fullText,
        durationMs,
        tokens: Math.round(fullText.length / 4),
        model: "OpenAI GPT-4o"
      };
    } catch (err) {
      const errText = `OpenAI API Error: ${err.message}`;
      if (onToken) onToken(errText);
      return {
        text: errText,
        durationMs: Math.round(performance.now() - startTime),
        tokens: 0,
        model: "GPT-4o (Error)"
      };
    }
  },

  async simulateStream(text, onToken) {
    const words = text.split(" ");
    let current = "";
    for (let i = 0; i < words.length; i++) {
      current += (i === 0 ? "" : " ") + words[i];
      onToken(current);
      await new Promise(r => setTimeout(r, 15));
    }
  }
};
