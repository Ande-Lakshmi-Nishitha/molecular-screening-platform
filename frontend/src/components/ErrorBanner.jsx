// The one place a uniform error is rendered. Pass any ApiError (or Error).
export default function ErrorBanner({ error }) {
  if (!error) return null;
  return (
    <div role="alert" style={{ border: "1px solid #b42318", background: "#fef3f2", color: "#7a271a", padding: "10px 12px", borderRadius: 6, margin: "12px 0" }}>
      <strong>{error.message}</strong>
      {error.code && error.code !== "LOGIN_FAILED" && (
        <div style={{ fontSize: 12, marginTop: 4 }}>Code: {error.code}</div>
      )}
      {error.requestId && (
        <div style={{ fontSize: 12, marginTop: 2 }}>Request ID: {error.requestId}</div>
      )}
      {error.details?.length > 0 && (
        <ul style={{ margin: "6px 0 0 18px" }}>
          {error.details.map((d, i) => (
            <li key={i}>{typeof d === "string" ? d : d.message || JSON.stringify(d)}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
