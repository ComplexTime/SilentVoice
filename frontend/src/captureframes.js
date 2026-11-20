// src/captureFrames.js
import {
  CAPTURE_HEIGHT,
  CAPTURE_WIDTH,
  FRAME_COUNT,
  FRAME_INTERVAL_MS,
  JPEG_QUALITY,
} from "./config";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Captures FRAME_COUNT frames from a <video> element, in order.
// Returns an array of Base64 JPEG strings WITHOUT the "data:image/jpeg;base64," prefix.
// The canvas is created off-screen (never added to the page), so it is hidden.
// Frames come from the raw video, so they are NOT mirrored.
export async function captureFrames(video, onProgress) {
  if (!video || video.readyState < 2) {
    throw new Error("The camera is not ready yet.");
  }

  const canvas = document.createElement("canvas");
  canvas.width = CAPTURE_WIDTH;
  canvas.height = CAPTURE_HEIGHT;
  const ctx = canvas.getContext("2d");

  const frames = [];
  for (let i = 0; i < FRAME_COUNT; i++) {
    const startedAt = performance.now();

    if (video.readyState < 2) {
      throw new Error("The camera stopped during recording.");
    }

    ctx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
    const dataUrl = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    frames.push(dataUrl.split(",")[1]);

    if (onProgress) onProgress(i + 1);

    if (i < FRAME_COUNT - 1) {
      const elapsed = performance.now() - startedAt;
      await sleep(Math.max(0, FRAME_INTERVAL_MS - elapsed));
    }
  }
  return frames;
}