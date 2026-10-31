# Triage summary (MSP-31)

Stored in `jobs.summary` and returned as `summary` by `GET /api/jobs/<id>` once every task is final
(done, failed, skipped or cancelled). It is `null` while any task is still queued or running.

```
{
  "rules_version": "0.2",
  "rules_status": "provisional",
  "generated_at": "2026-11-03T10:00:00+00:00",
  "dimensions": {
    "developability":       {"status": "green|amber|red|unavailable", "reasons": ["..."], "inputs": {"logs": -3.1, "unit": "logS"}},
    "safety_signals":       {"status": "...", "reasons": ["NR-AR signal 0.62"], "inputs": {"max_probability": 0.62, "assays_raised": ["NR-AR"]}},
    "target_activity":      {"status": "...", "reasons": ["..."], "inputs": {"mean": 7.4, "target": "EGFR"}},
    "electronic_stability": {"status": "...", "reasons": ["HOMO-LUMO gap 4.50 eV", "stub backend: illustrative only"], "inputs": {"gap_ev": 4.5, "is_stub": true}},
    "confidence":           {"status": "...", "reasons": ["..."], "inputs": {"failed_tiers": []}}
  },
  "tiers": {"ml": {"status": "done", "reason": null}, "dl": {"status": "skipped", "reason": "NO_TARGET_SELECTED"}},
  "flags": [{"code": "POTENT_POORLY_SOLUBLE", "message": "..."}],
  "limitation": "<text from core/labels.py>"
}
```

- A dimension with `"status": "unavailable"` has a `"reason"` code instead of useful `reasons`/`inputs`:
  `NO_TARGET_SELECTED`, `NO_FRAGMENT`, `UNSUPPORTED_ATOM` (skipped tiers), `COMPUTE_ERROR`, `TIMEOUT`,
  `MODEL_UNAVAILABLE` (failed tiers), `CANCELLED`, `NOT_FINISHED`, `NOT_CONVERGED`, `STUB_BACKEND`, `NO_DATA`.
- Red beats amber beats green inside one dimension. There is no overall colour (BR-005).
- Show `limitation` and `rules_status` with the summary; the thresholds are provisional until TBD-C9 closes.
- Flags: `POTENT_POORLY_SOLUBLE`, `POTENT_TOX21_SIGNAL`, `GOOD_PROFILE_OUT_OF_DOMAIN`.
