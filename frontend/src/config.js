// src/config.js
// ------------------------------------------------------------
// Change USE_MOCK to false when the backend is ready.
// ------------------------------------------------------------
export const USE_MOCK = false;

export const API_BASE_URL = "http://localhost:8000";

// Camera / recording (must match the Vision developer's input)
export const FRAME_COUNT = 15;
export const FRAME_INTERVAL_MS = 100;
export const CAPTURE_WIDTH = 640;
export const CAPTURE_HEIGHT = 480;
export const JPEG_QUALITY = 0.8;

// A prediction below this confidence needs no auto-add (user confirms)
export const MIN_CONFIDENCE = 0.8;

// Output languages offered in the UI (sent to POST /api/translate)
export const LANGUAGES = [
  { code: "en", label: "English", speechLang: "en-US" },
  { code: "he", label: "עברית (Hebrew)", speechLang: "he-IL" },
  { code: "ru", label: "Русский (Russian)", speechLang: "ru-RU" },
];