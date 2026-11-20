// src/components/SignDisplay.jsx
import { MIN_CONFIDENCE } from "../config";

export default function SignDisplay({
  prediction,
  loading,
  onAdd,
  onDiscard,
  minConfidence = MIN_CONFIDENCE,
  label = "Recognized sign",
  addLabel = "Add to sentence",
}) {
  if (loading) {
    return (
      <div className="sign-empty" role="status">
        <span className="spinner" aria-hidden="true" />
        Recognizing...
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="placeholder">
        Record and the result will appear here.
      </div>
    );
  }

  const percent = Math.round(prediction.confidence * 100);
  const isUnknown = prediction.sign.toUpperCase() === "UNKNOWN";
  const isLow = prediction.confidence < minConfidence;
  const level = isUnknown || isLow ? "low" : "high";

  return (
    <div className="sign-display">
      {prediction.mock && (
        <p className="mock-tag">MOCK RESULT: demo data, not real recognition</p>
      )}

      <p className="sign-label">{label}</p>
      <p className="sign-word" aria-live="polite">
        {prediction.sign}
      </p>

      <div className="confidence">
        <div className="confidence-row">
          <span>Confidence</span>
          <strong>{percent}%</strong>
        </div>
        <div
          className="meter"
          role="meter"
          aria-label="Prediction confidence"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
        >
          <div
            className={"meter-fill " + level}
            style={{ width: percent + "%" }}
          />
        </div>
      </div>

      {isUnknown && (
        <p className="alert alert-error" role="alert">
          Not recognized. Please record again.
        </p>
      )}
      {!isUnknown && isLow && (
        <p className="alert alert-warning" role="alert">
          Low confidence (below {Math.round(minConfidence * 100)}%). Check that
          this is what you meant.
        </p>
      )}

      <div className="sign-actions">
        {!isUnknown && (
          <button
            type="button"
            className={"btn " + (isLow ? "btn-ghost" : "btn-primary")}
            onClick={onAdd}
          >
            {isLow ? "Add anyway" : addLabel}
          </button>
        )}
        <button type="button" className="btn btn-ghost" onClick={onDiscard}>
          Discard
        </button>
      </div>
    </div>
  );
}