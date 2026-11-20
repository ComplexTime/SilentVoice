// src/App.jsx
import { useCallback, useRef, useState } from "react";
import "./App.css";
import { FRAME_COUNT, LANGUAGES, USE_MOCK } from "./config";
import Camera from "./components/Camera";
import SignDisplay from "./components/SignDisplay";
import SentenceBuilder from "./components/SentenceBuilder";
import Translator from "./components/Translator";
import { captureFrames } from "./captureFrames";
import { predictSign, translateSigns } from "./api";

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
      const result = await predictSign(frames);
      setPrediction(result);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setRecording(false);
      setPredicting(false);
      busyRef.current = false;
    }
  }, []);

  const handleAdd = useCallback(() => {
    if (!prediction || prediction.sign.toUpperCase() === "UNKNOWN") return;
    const id = nextId.current++;
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
  }, [prediction, invalidateSentence]);

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

  let buttonText = "Record Sign";
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
            />

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