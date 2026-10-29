import { ApiError, normalizeError, networkError } from "./errors";

const BASE_URL = import.meta.env?.VITE_API_URL || "/api";
const TOKEN_KEY = "msp_token";

export const tokenStore = {
  get: () => sessionStorage.getItem(TOKEN_KEY),
  set: (t) => sessionStorage.setItem(TOKEN_KEY, t),
  clear: () => sessionStorage.removeItem(TOKEN_KEY),
};

// AuthContext registers this so an expired session can redirect to login.
let onUnauthorized = () => {};
export const setUnauthorizedHandler = (fn) => {
  onUnauthorized = fn;
};

// The ONLY function in the app that calls fetch.
export async function request(path, { method = "GET", body, auth = true } = {}) {
  const headers = { "Content-Type": "application/json" };
  const token = tokenStore.get();
  if (auth && token) headers.Authorization = `Bearer ${token}`;

  let res;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw networkError();
  }

  let data = null;
  try {
    data = await res.json();
  } catch {
    /* empty or non-JSON body */
  }

  if (!res.ok) {
    // 401 on a protected call = expired/invalid session. Login/register use auth:false.
    if (res.status === 401 && auth) onUnauthorized();
    throw normalizeError(res.status, data);
  }
  return data;
}

export const api = {
  login: (email, password) => request("/auth/login", { method: "POST", body: { email, password }, auth: false }),
  register: (email, password) => request("/auth/register", { method: "POST", body: { email, password }, auth: false }),
};

export { ApiError };
