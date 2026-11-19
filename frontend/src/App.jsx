// src/App.jsx

import "./App.css";
import { USE_MOCK } from "./config";

function Logo() {
  return (
    <svg
      className="brand-logo"
      viewBox="0 0 48 48"
      role="img"
      aria-label="SilentVoice logo"
    >
      <rect width="48" height="48" rx="12" fill="#111C2E" />
      <path
        d="M10 24c4-8 8-8 12 0s8 8 12 0 3-5 4-6"
        fill="none"
        stroke="#2DD4BF"
        strokeWidth="3.5"
        strokeLinecap="round"
      />
      <path
        d="M10 32c4-6 8-6 12 0s8 6 12 0 3-4 4-5"
        fill="none"
        stroke="#A78BFA"
        strokeWidth="3.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

function App() {
  return (
    <div className="app">
      <a className="skip-link" href="#main">
        Skip to main content
      </a>

      <header className="header">
        <div className="brand">
          <Logo />
          <div>
            <h1 className="brand-name">
              Silent<span>Voice</span>
            </h1>
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
            <div className="placeholder">Camera preview goes here</div>
          </section>
        </div>

        <div className="column">
          <section className="card" aria-labelledby="signs-title">
            <h2 className="card-title" id="signs-title">
              <span className="step" aria-hidden="true">2</span>
              Recognized signs
            </h2>
            <div className="placeholder">Recognized sign and history go here</div>
          </section>

          <section className="card" aria-labelledby="translate-title">
            <h2 className="card-title" id="translate-title">
              <span className="step ai" aria-hidden="true">3</span>
              AI sentence
            </h2>
            <div className="placeholder">Translation and Speak button go here</div>
          </section>
        </div>
      </main>

      <footer className="footer">
        SilentVoice &middot; Hackathon project
      </footer>
    </div>
  );
}

export default App;
