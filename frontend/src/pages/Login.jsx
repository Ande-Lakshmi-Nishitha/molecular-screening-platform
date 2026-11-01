import { useState } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import ErrorBanner from "../components/ErrorBanner";

export default function Login() {
  const { login, isAuthenticated, notice } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  if (isAuthenticated) return <Navigate to="/submit" replace />;

  async function onSubmit(e) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await login(email, password);
      navigate(location.state?.from?.pathname || "/submit", { replace: true });
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ maxWidth: 360, margin: "60px auto" }}>
      <h1 data-testid="login-heading">Sign in</h1>
      {notice && <p role="status" data-testid="login-notice">{notice}</p>}
      {error && (
        <div data-testid="login-error">
          <ErrorBanner error={error} />
        </div>
      )}
      <form onSubmit={onSubmit} style={{ display: "grid", gap: 12 }} data-testid="login-form">
        <label>Email<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required style={{ width: "100%" }} data-testid="login-email" /></label>
        <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required style={{ width: "100%" }} data-testid="login-password" /></label>
        <button type="submit" disabled={busy} data-testid="login-submit">{busy ? "Signing in…" : "Sign in"}</button>
      </form>
      <p>No account? <Link to="/register" data-testid="login-register-link">Register</Link></p>
    </div>
  );
}
