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
// A FormData body (file upload) is sent as is; the browser sets the multipart Content-Type.
export async function request(path, { method = "GET", body, auth = true } = {}) {
  const isForm = typeof FormData !== "undefined" && body instanceof FormData;
  const headers = isForm ? {} : { "Content-Type": "application/json" };
  const token = tokenStore.get();
  if (auth && token) headers.Authorization = `Bearer ${token}`;

  let res;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      method,
      headers,
      body: body ? (isForm ? body : JSON.stringify(body)) : undefined,
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
    throw normalizeError(res.status, data, res.headers.get("X-Request-ID"));
  }
  return data;
}

export const api = {
  login: (email, password) => request("/auth/login", { method: "POST", body: { email, password }, auth: false }),
  register: (email, password) => request("/auth/register", { method: "POST", body: { email, password }, auth: false }),
  // Status and partial results of one of the caller's own jobs.
  getJob: (id) => request(`/jobs/${encodeURIComponent(id)}`),
  // Cancel a job; the response is the updated job, same shape as getJob.
  cancelJob: (id) => request(`/jobs/${encodeURIComponent(id)}`, { method: "DELETE" }),
  // Supported targets: { targets: [{ id, name, chembl_id, unit }] }.
  getTargets: () => request("/targets"),
  // Standardise a molecule and report per-tier eligibility (MSP-20).
  validateSmiles: (smiles, targetId) =>
    request("/molecules/validate", { method: "POST", body: targetId ? { smiles, target_id: targetId } : { smiles } }),
  validateFile: (file, targetId) => {
    const form = new FormData();
    form.append("file", file);
    if (targetId) form.append("target_id", targetId);
    return request("/molecules/validate", { method: "POST", body: form });
  },
  // Create a job and queue its tiers (MSP-29); 202 with { job_id, molecule, tasks }.
  submitMolecule: (payload) => request("/molecules/submit", { method: "POST", body: payload }),
  // PubChem or ChEMBL name search (MSP-21); { source, query, results, cached }.
  searchMolecules: (query, source) =>
    request(`/molecules/search?query=${encodeURIComponent(query)}&source=${encodeURIComponent(source)}`),
   getModels: () => request("/models"),
};

export { ApiError };
