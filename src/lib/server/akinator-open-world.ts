import { QUESTIONS } from "./akinator-kb";

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
  // Groq listed first: verified working. Cerebras/SambaNova are kept as
  // fallbacks in case their quotas are restored later — a failed request
  // (402/429/404) throws and the loop below falls through to the next one.
  const candidates = [
    {
      key: process.env.AKINATOR_GROQ_API_KEY || process.env.GROQ_API_KEY || "",
      baseUrl: "https://api.groq.com/openai/v1",
      model: process.env.AKINATOR_GROQ_MODEL || "openai/gpt-oss-120b",
    },
    {
      key: process.env.AKINATOR_CEREBRAS_API_KEY || process.env.CEREBRAS_API_KEY || "",
      baseUrl: "https://api.cerebras.ai/v1",
      model: process.env.AKINATOR_CEREBRAS_MODEL || "gpt-oss-120b",
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
  if (value.length > 32_000) throw new Error("Provider response exceeded the allowed size.");
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
    const timeout = setTimeout(() => controller.abort(), 15_000);
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
          // gpt-oss-family models spend part of this budget on internal
          // reasoning tokens before emitting the final JSON — 700 was
          // truncating mid-object and throwing in extractJson().
          max_tokens: options.maxTokens ?? 1500,
          response_format: { type: "json_object" },
        }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Provider failed with HTTP ${response.status}.`);
      const reader = response.body?.getReader();
      if (!reader) throw new Error("Provider returned an empty response.");
      const chunks: Uint8Array[] = [];
      let bytes = 0;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        bytes += value.byteLength;
        if (bytes > 128_000) {
          await reader.cancel();
          throw new Error("Provider response exceeded the allowed size.");
        }
        chunks.push(value);
      }
      const rawPayload = Buffer.concat(chunks).toString("utf8");
      const payload = JSON.parse(rawPayload) as {
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

export type VerifiedAnimal = {
  outcome: "VALID_NEW_ANIMAL" | "OUT_OF_SCOPE" | "AMBIGUOUS_REVIEW_REQUIRED";
  canonical_name?: { en: string; tr: string; ar: string };
  traits?: Record<string, boolean>;
};

export function parseAnimalVerification(raw: unknown): VerifiedAnimal | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const value = raw as Record<string, unknown>;
  if (value.outcome === "OUT_OF_SCOPE" || value.outcome === "AMBIGUOUS_REVIEW_REQUIRED") return { outcome: value.outcome };
  if (value.outcome !== "VALID_NEW_ANIMAL") return null;
  const names = value.canonical_name;
  const facts = value.traits;
  if (!names || typeof names !== "object" || Array.isArray(names) || !facts || typeof facts !== "object" || Array.isArray(facts)) return null;
  const name = names as Record<string, unknown>;
  if (!["en", "tr", "ar"].every(lang => typeof name[lang] === "string" && (name[lang] as string).trim().length > 0 && (name[lang] as string).length <= 80 && !/[\u0000-\u001f]/.test(name[lang] as string))) return null;
  if (Object.keys(facts).length > 150) return null;
  const inputTraits = facts as Record<string, unknown>;
  const traits = Object.fromEntries(QUESTIONS.filter(q => typeof inputTraits[q.id] === "boolean").map(q => [q.id, inputTraits[q.id] as boolean]));
  if (!Object.keys(traits).length) return null;
  return { outcome: "VALID_NEW_ANIMAL", canonical_name: { en: (name.en as string).trim(), tr: (name.tr as string).trim(), ar: (name.ar as string).trim() }, traits };
}
