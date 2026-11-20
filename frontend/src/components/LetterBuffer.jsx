// src/components/LetterBuffer.jsx
export default function LetterBuffer({
  letters,
  onBackspace,
  onSpace,
  onClear,
  onRemove,
  onAddWord,
  disabled,
}) {
  const empty = letters.length === 0;
  const text = letters.map((l) => l.ch).join("");
  const hasWord = text.trim() !== "";

  return (
    <div className="builder">
      <div className="builder-head">
        <h3 className="builder-title" id="letters-title">
          Letter buffer ({letters.length})
        </h3>
      </div>

      {empty ? (
        <p className="builder-empty">
          No letters yet. Confirm recognized letters to spell a word.
        </p>
      ) : (
        <>
          <p className="letter-word" aria-live="polite">
            {text.replace(/ /g, "\u00A0")}
          </p>
          <ol className="chips" aria-labelledby="letters-title">
            {letters.map((item, index) => {
              const name = item.ch === " " ? "space" : item.ch;
              return (
                <li key={item.id} className="chip">
                  <span className="chip-word">
                    {item.ch === " " ? "\u2423" : item.ch}
                  </span>
                  <button
                    type="button"
                    className="chip-remove"
                    onClick={() => onRemove(item.id)}
                    disabled={disabled}
                    aria-label={`Remove ${name}, position ${index + 1}`}
                  >
                    <span aria-hidden="true">&times;</span>
                  </button>
                </li>
              );
            })}
          </ol>
        </>
      )}

      <div className="builder-actions">
        <button
          type="button"
          className="btn btn-ghost btn-small"
          onClick={onBackspace}
          disabled={empty || disabled}
        >
          Backspace
        </button>
        <button
          type="button"
          className="btn btn-ghost btn-small"
          onClick={onSpace}
          disabled={empty || disabled}
        >
          Space
        </button>
        <button
          type="button"
          className="btn btn-danger btn-small"
          onClick={onClear}
          disabled={empty || disabled}
        >
          Clear
        </button>
        <button
          type="button"
          className="btn btn-primary btn-small"
          onClick={onAddWord}
          disabled={!hasWord || disabled}
        >
          Add word to sentence
        </button>
      </div>
    </div>
  );
}