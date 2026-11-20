// src/api.js
import { API_BASE_URL, FRAME_COUNT, LANGUAGES, USE_MOCK } from "./config";

export class ApiError extends Error {
  constructor(message, kind = "unknown") {
    super(message);
    this.name = "ApiError";
    this.kind = kind; // "network" | "timeout" | "http" | "invalid" | "input"
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Shared POST helper: timeout, network errors, bad status, bad JSON
async function postJson(path, body, timeoutMs = 15000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    let response;
    try {
      response = await fetch(API_BASE_URL + path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (err) {
      if (err.name === "AbortError") {
        throw new ApiError(
          "The backend took too long to respond. Please try again.",
          "timeout"
        );
      }
      throw new ApiError(
        `Cannot reach the backend at ${API_BASE_URL}. Is it running?`,
        "network"
      );
    }

    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }

    if (!response.ok) {
      const detail =
        data && typeof data.detail === "string" ? " " + data.detail : "";
      throw new ApiError(`Backend error (${response.status}).${detail}`, "http");
    }
    if (data === null) {
      throw new ApiError("The backend returned an invalid response.", "invalid");
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

// POST /api/predict
// Returns { sign, confidence, mock }
export async function predictSign(frames) {
  if (!Array.isArray(frames) || frames.length !== FRAME_COUNT) {
    throw new ApiError(`Expected exactly ${FRAME_COUNT} frames.`, "input");
  }
  if (frames.some((f) => typeof f !== "string" || f.startsWith("data:"))) {
    throw new ApiError(
      "Frames must be Base64 strings without the data: prefix.",
      "input"
    );
  }

  if (USE_MOCK) {
    await sleep(700);
    return { sign: "WATER", confidence: 0.94, mock: true };
  }

  const data = await postJson("/api/predict", { frames });
  const okSign = typeof data.sign === "string" && data.sign.trim() !== "";
  const okConf =
    typeof data.confidence === "number" &&
    Number.isFinite(data.confidence) &&
    data.confidence >= 0 &&
    data.confidence <= 1;
  if (!okSign || !okConf) {
    throw new ApiError(
      "The backend returned an unexpected prediction format.",
      "invalid"
    );
  }
  return { sign: data.sign.trim(), confidence: data.confidence, mock: false };
}

// POST /api/translate
// Returns { sentence, mock }
export async function translateSigns(signs, language) {
  if (!Array.isArray(signs) || signs.length === 0) {
    throw new ApiError("Add at least one sign before translating.", "input");
  }
  if (!LANGUAGES.some((l) => l.code === language)) {
    throw new ApiError("Unsupported language.", "input");
  }

  if (USE_MOCK) {
    await sleep(800);
    return { sentence: "I would like some water.", mock: true };
  }

  const data = await postJson("/api/translate", { signs, language }, 20000);
  if (typeof data.sentence !== "string" || data.sentence.trim() === "") {
    throw new ApiError(
      "The backend returned an unexpected translation format.",
      "invalid"
    );
  }
  return { sentence: data.sentence.trim(), mock: false };
}