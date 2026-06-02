import ArAvDiagram from "./components/ArAvDiagram";

export default function App() {
  return (
    <main
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "2rem",
        boxSizing: "border-box",
      }}
    >
      <div style={{ width: "100%", maxWidth: 720 }}>
        <ArAvDiagram />
      </div>
    </main>
  );
}
