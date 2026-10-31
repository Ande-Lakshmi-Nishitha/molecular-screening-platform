// The one place a uniform error is rendered. Pass any ApiError (or Error).
const describe = (d) =>
  typeof d === "string"
    ? d
    : d.message ||
      (d.service ? `${d.service}: ${d.type}` : d.field ? `${d.field}: ${d.type}` : JSON.stringify(d));

export default function ErrorBanner({ error }) {
  if (!error) return null;
  return (
    <div role="alert" data-testid="error-banner" style={{ border: "1px solid #b42318", background: "#fef3f2", color: "#7a271a", padding: "10px 12px", borderRadius: 6, margin: "12px 0" }}>
      <strong data-testid="error-message">{error.message}</strong>
      {error.code && error.code !== "LOGIN_FAILED" && (
        <div data-testid="error-code" style={{ fontSize: 12, marginTop: 4 }}>Code: {error.code}</div>
      )}
      {error.requestId && (
        <div data-testid="error-request-id" style={{ fontSize: 12, marginTop: 2 }}>Request ID: {error.requestId}</div>
      )}
      {error.details?.length > 0 && (
        <ul style={{ margin: "6px 0 0 18px" }}>
          {error.details.map((d, i) => (
            <li key={i} data-testid="error-detail">{describe(d)}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
