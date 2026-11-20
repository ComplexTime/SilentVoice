// src/components/SpeechButton.jsx
import { useCallback, useEffect, useState } from "react";
import { LANGUAGES } from "../config";

// Picks the best installed voice for a language code like "he-IL"
function pickVoice(voices, speechLang) {
  const lang = speechLang.toLowerCase();
  const base = lang.split("-")[0];
  const exact = voices.filter((v) => v.lang.toLowerCase() === lang);
  const loose = voices.filter((v) => v.lang.toLowerCase().split(/[-_]/)[0] === base);
  const pool = exact.length ? exact : loose;
  if (!pool.length) return null;
  return pool.find((v) => v.localService) || pool[0];
}

export default function SpeechButton({ text, language }) {
  const [voices, setVoices] = useState([]);
  const [speaking, setSpeaking] = useState(false);
  const [message, setMessage] = useState("");

  const supported =
    typeof window !== "undefined" &&
    "speechSynthesis" in window &&
    "SpeechSynthesisUtterance" in window;

  const speechLang =
    LANGUAGES.find((l) => l.code === language)?.speechLang || "en-US";

  // Voices load asynchronously in most browsers
  useEffect(() => {
    if (!supported) return;
    const load = () => setVoices(window.speechSynthesis.getVoices());
    load();
    window.speechSynthesis.addEventListener("voiceschanged", load);
    return () =>
      window.speechSynthesis.removeEventListener("voiceschanged", load);
  }, [supported]);

  // Stop speaking when the sentence changes or the component unmounts
  useEffect(() => {
    return () => {
      if (supported) window.speechSynthesis.cancel();
    };
  }, [supported, text, language]);

  // Reset UI state when the sentence changes
  useEffect(() => {
    setSpeaking(false);
    setMessage("");
  }, [text, language]);

  const stop = useCallback(() => {
    if (supported) window.speechSynthesis.cancel();
    setSpeaking(false);
  }, [supported]);

  const speak = useCallback(() => {
    if (!supported || !text) return;
    setMessage("");

    const voice = pickVoice(window.speechSynthesis.getVoices(), speechLang);
    if (!voice) {
      const name = LANGUAGES.find((l) => l.code === language)?.label || language;
      setMessage(
        `No ${name} voice is installed on this device. Install one in Windows Settings > Time & language > Speech, then reload the page.`
      );
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = voice;
    utterance.lang = voice.lang;
    utterance.rate = 0.95;
    utterance.onstart = () => setSpeaking(true);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = (e) => {
      setSpeaking(false);
      if (e.error !== "canceled" && e.error !== "interrupted") {
        setMessage("Speech failed. Please try again.");
      }
    };
    window.speechSynthesis.speak(utterance);
  }, [supported, text, speechLang, language]);

  if (!supported) {
    return (
      <p className="alert alert-error" role="alert">
        This browser does not support speech. Use Chrome or Edge.
      </p>
    );
  }

  const voiceAvailable = voices.length === 0 || !!pickVoice(voices, speechLang);

  return (
    <div className="speech">
      <button
        type="button"
        className={"btn " + (speaking ? "btn-danger" : "btn-primary")}
        onClick={speaking ? stop : speak}
        disabled={!text}
        aria-label={speaking ? "Stop speaking" : "Speak the sentence aloud"}
      >
        {speaking ? "Stop" : "Speak"}
      </button>

      {!voiceAvailable && !message && (
        <p className="alert alert-warning" role="status">
          No voice found for this language on this device. Speaking may not work.
        </p>
      )}
      {message && (
        <p className="alert alert-error" role="alert">
          {message}
        </p>
      )}
    </div>
  );
}