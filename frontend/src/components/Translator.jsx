// src/components/Translator.jsx
import SpeechButton from "./SpeechButton";
import LanguageSelector from "./LanguageSelector";

export default function Translator({
  signs,
  language,
  onLanguageChange,
  result, // { sentence, mock, language } or null
  loading,
  error,
  onTranslate,
}) {
  const hasSigns = signs.length > 0;
  const isRtl = result?.language === "he";

  return (
    <div className="translator">
      <div className="translator-controls">
        <LanguageSelector
          value={language}
          onChange={onLanguageChange}
          disabled={loading}
        />
        <button
          type="button"
          className="btn btn-ai"
          onClick={onTranslate}
          disabled={!hasSigns || loading}
        >
          {loading && <span className="spinner" aria-hidden="true" />}
          {loading ? "Translating..." : "Translate"}
        </button>
      </div>

      {!hasSigns && (
        <p className="builder-empty">
          Add at least one sign to the word history to translate it.
        </p>
      )}

      {error && (
        <p className="alert alert-error" role="alert">
          {error}
        </p>
      )}

      <div className="sentence-box" aria-live="polite">
        {loading ? (
          <p className="sentence-loading">
            <span className="spinner" aria-hidden="true" />
            The AI is building your sentence...
          </p>
        ) : result ? (
          <>
            {result.mock && (
              <p className="mock-tag">
                MOCK RESULT: demo data, not a real AI translation
              </p>
            )}
                        <p
              className="sentence-text"
              lang={result.language}
              dir={isRtl ? "rtl" : "ltr"}
            >
              {result.sentence}
            </p>
            <SpeechButton text={result.sentence} language={result.language} />
            </>
        ) : (
          <p className="sentence-placeholder">
            Your AI-generated sentence will appear here.
          </p>
        )}
      </div>
    </div>
  );
}