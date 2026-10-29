// Single place that understands the backend's uniform error JSON.
// ASSUMED shape: { "error": { "code": "...", "message": "...", "details": [...] } }
// If the SRS defines it differently, change ONLY normalizeError below.

export class ApiError extends Error {
  constructor({ status, code, message, details }) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details || [];
  }
}

export function normalizeError(status, body) {
  const e = (body && body.error) || body || {};
  return new ApiError({
    status,
    code: e.code || "UNKNOWN_ERROR",
    message: e.message || "Something went wrong. Please try again.",
    details: Array.isArray(e.details) ? e.details : [],
  });
}

export const networkError = () =>
  new ApiError({
    status: 0,
    code: "NETWORK_ERROR",
    message: "Cannot reach the server. Check your connection and try again.",
  });
