// src/components/ModeSelector.jsx
const MODES = [
  { id: "word", label: "Word mode" },
  { id: "alphabet", label: "Alphabet mode" },
];

export default function ModeSelector({ value, onChange, disabled }) {
  return (
    <div className="mode-select" role="group" aria-label="Recognition mode">
      {MODES.map((m) => (
        <button
          key={m.id}
          type="button"
          className={"mode-btn" + (value === m.id ? " active" : "")}
          aria-pressed={value === m.id}
          disabled={disabled}
          onClick={() => onChange(m.id)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}