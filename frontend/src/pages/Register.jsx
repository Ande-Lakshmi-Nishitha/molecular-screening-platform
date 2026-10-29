import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import ErrorBanner from "../components/ErrorBanner";

export default function Register() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e) {
    e.preventDefault();
    setError(null);
    if (password !== confirm) {
      setError({ message: "Passwords do not match." });
      return;
    }
    setBusy(true);
    try {
      await api.register(email, password);
      navigate("/login", { replace: true });
    } catch (err) {
      setError(err); // uniform error JSON rendered by ErrorBanner
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ maxWidth: 360, margin: "60px auto" }}>
      <h1>Create account</h1>
      <ErrorBanner error={error} />
      <form onSubmit={onSubmit} style={{ display: "grid", gap: 12 }}>
        <label>Email<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required style={{ width: "100%" }} /></label>
        <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required style={{ width: "100%" }} /></label>
        <label>Confirm password<input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} required style={{ width: "100%" }} /></label>
        <button type="submit" disabled={busy}>{busy ? "Creating…" : "Register"}</button>
      </form>
      <p>Already registered? <Link to="/login">Sign in</Link></p>
    </div>
  );
}
