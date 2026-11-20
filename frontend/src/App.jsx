// src/App.jsx
import { useCallback, useRef, useState } from "react";
import "./App.css";
import {
  ALPHABET_MIN_CONFIDENCE,
  FRAME_COUNT,
  LANGUAGES,
  MIN_CONFIDENCE,
  USE_MOCK,
} from "./config";
import Camera from "./components/Camera";
import SignDisplay from "./components/SignDisplay";
import SentenceBuilder from "./components/SentenceBuilder";
import Translator from "./components/Translator";
import ModeSelector from "./components/ModeSelector";
import LetterBuffer from "./components/LetterBuffer";
import { captureFrames } from "./captureFrames";
import { predictLetter, predictSign, translateSigns } from "./api";

function Logo() {
  return (
    <svg className="brand-logo" viewBox="0 0 48 48" role="img" aria-label="SilentVoice logo">
      <rect width="48" height="48" rx="12" fill="#111C2E" />
      <path d="M10 24c4-8 8-8 12 0s8 8 12 0 3-5 4-6" fill="none" stroke="#2DD4BF" strokeWidth="3.5" strokeLinecap="round" />
      <path d="M10 32c4-6 8-6 12 0s8 6 12 0 3-4 4-5" fill="none" stroke="#A78BFA" strokeWidth="3.5" strokeLinecap="round" />
    </svg>
  );
}

function App() {
  const cameraRef = useRef(null);
  const busyRef = useRef(false); // blocks overlapping record/predict runs
  const nextId = useRef(1);
  const translateRequestRef = useRef(0); // lets us ignore outdated translations

  const [cameraStatus, setCameraStatus] = useState("off");
  const [recording, setRecording] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");

  const [prediction, setPrediction] = useState(null);
  const [signs, setSigns] = useState([]);
  const [mode, setMode] = useState("word"); // "word" | "alphabet"
  const [letters, setLetters] = useState([]); // [{ id, ch, mock }]

  const [language, setLanguage] = useState(LANGUAGES[0].code);
  const [sentenceResult, setSentenceResult] = useState(null);
  const [translating, setTranslating] = useState(false);
  const [translateError, setTranslateError] = useState("");

  // Clears the sentence and cancels any translation still in flight
  const invalidateSentence = useCallback(() => {
    translateRequestRef.current += 1;
    setSentenceResult(null);
    setTranslateError("");
    setTranslating(false);
  }, []);

  const handleRecord = useCallback(async () => {
    if (busyRef.current) return;
    busyRef.current = true;

    setError("");
    setPrediction(null);
    setProgress(0);
    setRecording(true);

    try {
      const video = cameraRef.current?.getVideoElement();
      const frames = await captureFrames(video, setProgress);
      setRecording(false);

      setPredicting(true);
      const predict = mode === "alphabet" ? predictLetter : predictSign;
      const result = await predict(frames);
      setPrediction(result);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setRecording(false);
      setPredicting(false);
      busyRef.current = false;
    }
  }, [mode]);

const handleAdd = useCallback(() => {
    if (!prediction || prediction.sign.toUpperCase() === "UNKNOWN") return;
    const id = nextId.current++;

    if (mode === "alphabet") {
      setLetters((prev) => [
        ...prev,
        { id, ch: prediction.sign.toUpperCase(), mock: prediction.mock },
      ]);
      setPrediction(null);
      setProgress(0);
      return;
    }

    setSigns((prev) => [
      ...prev,
      {
        id,
        sign: prediction.sign,
        confidence: prediction.confidence,
        mock: prediction.mock,
      },
    ]);
    setPrediction(null);
    setProgress(0);
    invalidateSentence();
  }, [prediction, mode, invalidateSentence]);

  const handleModeChange = useCallback((next) => {
    if (busyRef.current) return; // never switch while recording/predicting
    setMode(next);
    setPrediction(null); // clear stale prediction
    setProgress(0);
    setError("");
  }, []);

  const handleBackspace = useCallback(() => {
    setLetters((prev) => prev.slice(0, -1));
  }, []);

  const handleSpace = useCallback(() => {
    setLetters((prev) => {
      if (prev.length === 0 || prev[prev.length - 1].ch === " ") return prev;
      return [...prev, { id: nextId.current++, ch: " ", mock: false }];
    });
  }, []);

  const handleRemoveLetter = useCallback((id) => {
    setLetters((prev) => prev.filter((l) => l.id !== id));
  }, []);

  const handleClearLetters = useCallback(() => {
    setLetters([]);
  }, []);

  const handleAddWord = useCallback(() => {
    const words = letters
      .map((l) => l.ch)
      .join("")
      .split(/\s+/)
      .filter(Boolean);
    if (words.length === 0) return;
    const anyMock = letters.some((l) => l.mock);
    const entries = words.map((w) => ({
      id: nextId.current++,
      sign: w,
      confidence: null,
      mock: anyMock,
    }));
    setSigns((prev) => [...prev, ...entries]);
    setLetters([]);
    invalidateSentence();
  }, [letters, invalidateSentence]);

  const handleDiscard = useCallback(() => {
    setPrediction(null);
    setProgress(0);
  }, []);

  const handleRemove = useCallback(
    (id) => {
      setSigns((prev) => prev.filter((s) => s.id !== id));
      invalidateSentence();
    },
    [invalidateSentence]
  );

  const handleClear = useCallback(() => {
    setSigns([]);
    invalidateSentence();
  }, [invalidateSentence]);

  const handleLanguageChange = useCallback(
    (code) => {
      setLanguage(code);
      invalidateSentence();
    },
    [invalidateSentence]
  );

  const handleTranslate = useCallback(async () => {
    if (translating || signs.length === 0) return;

    const requestId = ++translateRequestRef.current;
    setTranslating(true);
    setTranslateError("");
    setSentenceResult(null);

    try {
      const result = await translateSigns(
        signs.map((s) => s.sign),
        language
      );
      if (requestId !== translateRequestRef.current) return; // outdated
      setSentenceResult({ ...result, language });
    } catch (err) {
      if (requestId !== translateRequestRef.current) return;
      setTranslateError(err.message || "Translation failed.");
    } finally {
      if (requestId === translateRequestRef.current) setTranslating(false);
    }
  }, [translating, signs, language]);

  const cameraOn = cameraStatus === "on";
  const busy = recording || predicting;

  let buttonText = mode === "alphabet" ? "Record Letter" : "Record Sign";
  if (recording) buttonText = "Recording...";
  if (predicting) buttonText = "Recognizing...";

  let progressText = "Start the camera first";
  if (cameraOn) progressText = "Ready to record";
  if (recording || progress > 0) progressText = `Frame ${progress} / ${FRAME_COUNT}`;
  if (predicting) progressText = "Sending frames for recognition...";

  return (
    <div className="app">
      <a className="skip-link" href="#main">Skip to main content</a>

      <header className="header">
        <div className="brand">
          <Logo />
          <div>
            <h1 className="brand-name">Silent<span>Voice</span></h1>
            <p className="brand-slogan">Give every movement a voice.</p>
          </div>
        </div>

        {USE_MOCK ? (
          <div className="mode-badge mock" role="status">
            <span className="dot" aria-hidden="true" />
            MOCK MODE: demo data, not real recognition
          </div>
        ) : (
          <div className="mode-badge live" role="status">
            <span className="dot" aria-hidden="true" />
            LIVE: connected to backend
          </div>
        )}
      </header>

      <main id="main" className="layout">
        <div className="column">
          <section className="card" aria-labelledby="camera-title">
            <h2 className="card-title" id="camera-title">
              <span className="step" aria-hidden="true">1</span>
              Camera
            </h2>
<ModeSelector
              value={mode}
              onChange={handleModeChange}
              disabled={busy}
            />
            <Camera ref={cameraRef} onStatusChange={setCameraStatus} />

            <div className="record-area">
              <button
                type="button"
                className="btn btn-primary btn-record"
                onClick={handleRecord}
                disabled={!cameraOn || busy}
              >
                {predicting && <span className="spinner" aria-hidden="true" />}
                {buttonText}
              </button>

              <div className="progress-wrap" aria-live="polite">
                <div
                  className="progress"
                  role="progressbar"
                  aria-label="Recording progress"
                  aria-valuemin={0}
                  aria-valuemax={FRAME_COUNT}
                  aria-valuenow={progress}
                >
                  <div
                    className="progress-bar"
                    style={{ width: (progress / FRAME_COUNT) * 100 + "%" }}
                  />
                </div>
                <span className="progress-label">{progressText}</span>
              </div>
            </div>

            {error && (
              <p className="alert alert-error" role="alert" style={{ marginTop: 12 }}>
                {error}
              </p>
            )}
          </section>
        </div>

        <div className="column">
          <section className="card" aria-labelledby="signs-title">
            <h2 className="card-title" id="signs-title">
              <span className="step" aria-hidden="true">2</span>
              Recognized signs
            </h2>

   <SignDisplay
              prediction={prediction}
              loading={predicting}
              onAdd={handleAdd}
              onDiscard={handleDiscard}
              minConfidence={
                mode === "alphabet" ? ALPHABET_MIN_CONFIDENCE : MIN_CONFIDENCE
              }
              label={mode === "alphabet" ? "Recognized letter" : "Recognized sign"}
              addLabel={mode === "alphabet" ? "Add letter" : "Add to sentence"}
            />

            {mode === "alphabet" && (
              <LetterBuffer
                letters={letters}
                onBackspace={handleBackspace}
                onSpace={handleSpace}
                onClear={handleClearLetters}
                onRemove={handleRemoveLetter}
                onAddWord={handleAddWord}
                disabled={busy}
              />
            )}

            <SentenceBuilder
              signs={signs}
              onRemove={handleRemove}
              onClear={handleClear}
              disabled={busy}
            />
          </section>

          <section className="card" aria-labelledby="translate-title">
            <h2 className="card-title" id="translate-title">
              <span className="step ai" aria-hidden="true">3</span>
              AI sentence
            </h2>

            <Translator
              signs={signs}
              language={language}
              onLanguageChange={handleLanguageChange}
              result={sentenceResult}
              loading={translating}
              error={translateError}
              onTranslate={handleTranslate}
            />
          </section>
        </div>
      </main>

      <footer className="footer">SilentVoice &middot; Hackathon project</footer>
    </div>
  );
}

export default App;