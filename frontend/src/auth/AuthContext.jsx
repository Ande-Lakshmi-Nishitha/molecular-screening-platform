import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, setUnauthorizedHandler, tokenStore } from "../api/client";

const AuthContext = createContext(null);
export const useAuth = () => useContext(AuthContext);

export function AuthProvider({ children }) {
  const navigate = useNavigate();
  const [token, setToken] = useState(tokenStore.get());
  const [notice, setNotice] = useState("");

  const logout = useCallback(() => {
    tokenStore.clear();
    setToken(null);
    navigate("/login", { replace: true });
  }, [navigate]);

  // Expired session: clear token, tell the user, go to login.
  useEffect(() => {
    setUnauthorizedHandler(() => {
      tokenStore.clear();
      setToken(null);
      setNotice("Your session has expired. Please sign in again.");
      navigate("/login", { replace: true });
    });
  }, [navigate]);

  const login = useCallback(async (email, password) => {
    try {
      const data = await api.login(email, password);
      const t = data.token || data.access_token;
      if (!t) throw new Error("no token");
      tokenStore.set(t);
      setToken(t);
      setNotice("");
    } catch (err) {
      // Deliberately generic: never reveal whether the email or password was wrong.
      if (err.code === "NETWORK_ERROR") throw err;
      const generic = new Error("Invalid email or password.");
      generic.code = "LOGIN_FAILED";
      throw generic;
    }
  }, []);

  const value = useMemo(
    () => ({ token, isAuthenticated: !!token, login, logout, notice, clearNotice: () => setNotice("") }),
    [token, login, logout, notice]
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
