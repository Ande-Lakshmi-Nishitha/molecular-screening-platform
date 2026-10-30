// Single place that understands the backend's uniform error JSON.
// Shape: { "error": { "code", "message", "details", "request_id" } }.
// details may be a list (API skeleton) or an object (auth/jobs routes); both are accepted.
// If the SRS defines it differently, change ONLY normalizeError below.

export class ApiError extends Error {
  constructor({ status, code, message, details, requestId }) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details || [];
    this.requestId = requestId || null;
  }
}

// headerRequestId is the X-Request-ID response header; it matches the server logs.
export function normalizeError(status, body, headerRequestId) {
  const e = (body && body.error) || body || {};
  return new ApiError({
    status,
    code: e.code || "UNKNOWN_ERROR",
    message: e.message || "Something went wrong. Please try again.",
    details: Array.isArray(e.details) ? e.details : [],
    requestId: headerRequestId || e.request_id,
  });
}

export const networkError = () =>
  new ApiError({
    status: 0,
    code: "NETWORK_ERROR",
    message: "Cannot reach the server. Check your connection and try again.",
  });
