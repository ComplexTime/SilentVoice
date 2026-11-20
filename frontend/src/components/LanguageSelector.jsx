// src/components/LanguageSelector.jsx
import { LANGUAGES } from "../config";

export default function LanguageSelector({ value, onChange, disabled }) {
  return (
    <div className="lang-select">
      <label htmlFor="language-select" className="lang-label">
        Output language
      </label>
      <select
        id="language-select"
        className="lang-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
      >
        {LANGUAGES.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.label}
          </option>
        ))}
      </select>
    </div>
  );
}