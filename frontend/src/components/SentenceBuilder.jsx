// src/components/SentenceBuilder.jsx
export default function SentenceBuilder({ signs, onRemove, onClear, disabled }) {
  return (
    <div className="builder">
      <div className="builder-head">
        <h3 className="builder-title" id="history-title">
          Word history ({signs.length})
        </h3>
        <button
          type="button"
          className="btn btn-danger btn-small"
          onClick={onClear}
          disabled={signs.length === 0 || disabled}
        >
          Clear sentence
        </button>
      </div>

      {signs.length === 0 ? (
        <p className="builder-empty">No words yet. Add recognized signs here.</p>
      ) : (
        <ol className="chips" aria-labelledby="history-title">
          {signs.map((item, index) => (
            <li key={item.id} className="chip">
              <span className="chip-num" aria-hidden="true">
                {index + 1}
              </span>
              <span className="chip-word">{item.sign}</span>
              <button
                type="button"
                className="chip-remove"
                onClick={() => onRemove(item.id)}
                disabled={disabled}
                aria-label={`Remove ${item.sign}, word ${index + 1}`}
              >
                <span aria-hidden="true">&times;</span>
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}