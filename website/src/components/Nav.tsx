interface Props {
  sections: { id: string; label: string }[];
  generatedAt: string;
}

export default function Nav({ sections, generatedAt }: Props) {
  return (
    <nav className="toc" aria-label="Section navigation">
      <h4>Contents</h4>
      <ol>
        {sections.map((s) => (
          <li key={s.id}>
            <a href={`#${s.id}`}>{s.label}</a>
          </li>
        ))}
      </ol>
      <div style={{ marginTop: "1rem", fontSize: "0.78rem" }}>
        Snapshot:
        <br />
        <code>{generatedAt.slice(0, 10)}</code>
      </div>
    </nav>
  );
}
