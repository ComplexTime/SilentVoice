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

function checkFrames(frames) {
  if (!Array.isArray(frames) || frames.length !== FRAME_COUNT) {
    throw new ApiError(`Expected exactly ${FRAME_COUNT} frames.`, "input");
  }
  if (frames.some((f) => typeof f !== "string" || f.startsWith("data:"))) {
    throw new ApiError(
      "Frames must be Base64 strings without the data: prefix.",
      "input"
    );
  }
}

// Shared by word and alphabet prediction (same request, same response shape)
async function requestPrediction(path, frames, mockResult) {
  checkFrames(frames);

  if (USE_MOCK) {
    await sleep(700);
    return mockResult();
  }

  const data = await postJson(path, { frames });
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

// POST /api/predict  (word mode)
export function predictSign(frames) {
  return requestPrediction("/api/predict", frames, () => ({
    sign: "WATER",
    confidence: 0.94,
    mock: true,
  }));
}

// POST /api/predict/alphabet  (alphabet mode)
const MOCK_LETTERS = ["H", "E", "L", "L", "O"];
let mockLetterIndex = 0;

export function predictLetter(frames) {
  return requestPrediction("/api/predict/alphabet", frames, () => {
    const sign = MOCK_LETTERS[mockLetterIndex % MOCK_LETTERS.length];
    mockLetterIndex += 1;
    return { sign, confidence: 0.91, mock: true };
  });
}

// POST /api/translate
// Returns { sentence, mock }
const MOCK_SENTENCES = {
  en: "I would like some water.",
  he: "אני רוצה קצת מים.",
  ru: "Я хотел бы немного воды.",
};

export async function translateSigns(signs, language) {
  if (!Array.isArray(signs) || signs.length === 0) {
    throw new ApiError("Add at least one sign before translating.", "input");
  }
  if (!LANGUAGES.some((l) => l.code === language)) {
    throw new ApiError("Unsupported language.", "input");
  }

  if (USE_MOCK) {
    await sleep(800);
    return {
      sentence: MOCK_SENTENCES[language] || MOCK_SENTENCES.en,
      mock: true,
    };
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
