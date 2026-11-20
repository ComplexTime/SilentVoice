
export default function AlphabetGuide() {
  return (
    <section className="alphabet-guide">
      <h2>ASL Alphabet Reference</h2>

      <p>Learn the hand positions and practice fingerspelling.</p>

      <a href="/asl-alphabet.svg" target="_blank" rel="noreferrer">
        <img
          src="/asl-alphabet.svg"
          alt="American Sign Language alphabet hand positions"
        />
      </a>

      <p className="alphabet-note">
        Supported: 24 static letters.
        J and Z require movement and are not currently supported.
      </p>
    </section>
  );
}