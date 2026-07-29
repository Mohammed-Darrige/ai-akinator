type Message = {
  role: "system" | "user" | "assistant";
  content: string;
};

type Provider = {
  key: string;
  baseUrl: string;
  model: string;
};

function providers(): Provider[] {
  const candidates = [
    {
      key: process.env.AKINATOR_CEREBRAS_API_KEY || process.env.CEREBRAS_API_KEY || "",
      baseUrl: "https://api.cerebras.ai/v1",
      model: process.env.AKINATOR_CEREBRAS_MODEL || "llama-3.3-70b",
    },
    {
      key: process.env.AKINATOR_GROQ_API_KEY || process.env.GROQ_API_KEY || "",
      baseUrl: "https://api.groq.com/openai/v1",
      model: process.env.AKINATOR_GROQ_MODEL || "llama-3.3-70b-versatile",
    },
    {
      key: process.env.AKINATOR_SAMBANOVA_API_KEY || process.env.SAMBANOVA_API_KEY || "",
      baseUrl: "https://api.sambanova.ai/v1",
      model: process.env.AKINATOR_SAMBANOVA_MODEL || "Meta-Llama-3.3-70B-Instruct",
    },
  ];
  return candidates.filter((provider) => provider.key);
}

function extractJson(value: string) {
  const start = value.indexOf("{");
  const end = value.lastIndexOf("}");
  if (start < 0 || end <= start) throw new Error("Provider returned invalid JSON.");
  return JSON.parse(value.slice(start, end + 1)) as Record<string, unknown>;
}

export function hasOpenWorldProvider() {
  return providers().length > 0;
}

export async function llmCallJson(
  messages: Message[],
  options: { temperature?: number; maxTokens?: number } = {},
) {
  let lastError: unknown;
  for (const provider of providers()) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8_000);
    try {
      const response = await fetch(`${provider.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${provider.key}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: provider.model,
          messages,
          temperature: options.temperature ?? 0.1,
          max_tokens: options.maxTokens ?? 700,
          response_format: { type: "json_object" },
        }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Provider failed with HTTP ${response.status}.`);
      const payload = await response.json() as {
        choices?: Array<{ message?: { content?: string } }>;
      };
      return extractJson(payload.choices?.[0]?.message?.content || "");
    } catch (error) {
      lastError = error;
    } finally {
      clearTimeout(timeout);
    }
  }
  throw lastError instanceof Error ? lastError : new Error("No open-world provider is configured.");
}
