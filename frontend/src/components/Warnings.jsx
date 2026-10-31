// components/Warnings.jsx - owner: Abheetha
// Shows the warnings list from POST /api/molecules/validate. Renders nothing when there are none.
export default function Warnings({ warnings }) {
  if (!warnings || warnings.length === 0) return null;
  return (
    <div
      role="status"
      data-testid="validation-warnings"
      style={{ border: "1px solid #b54708", background: "#fffaeb", color: "#7a2e0e", padding: "10px 12px", borderRadius: 6, margin: "12px 0" }}
    >
      <strong>Warnings</strong>
      <ul style={{ margin: "6px 0 0 18px" }}>
        {warnings.map((w, i) => (
          <li key={i} data-testid={`validation-warning-${i}`}>
            {w}
          </li>
        ))}
      </ul>
    </div>
  );
}
