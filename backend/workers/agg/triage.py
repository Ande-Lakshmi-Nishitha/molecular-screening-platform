"""Green/amber/red for five dimensions from triage_rules.yaml (FR-033, FR-035, BR-005).

Owner: Ayushi. Pure functions: no database, no Celery. Rules are explicit and configurable, there is
no opaque score. Within a dimension the worst colour wins. A tier that did not finish shows
"unavailable" with its reason; unconverged QML is excluded.
The caller passes a list of task dicts: {tier, status, skip_reason, error, payload}.
"""
from datetime import datetime, timezone

from backend.core import labels
from backend.workers.agg import flags

GREEN, AMBER, RED, UNAVAILABLE = "green", "amber", "red", "unavailable"
RANK = {GREEN: 0, AMBER: 1, RED: 2}
LIMITATION = getattr(labels, "TRIAGE_LIMITATION", None)  # wording is owned by core/labels.py
REQUIRED_RULES = {
    "developability": ("logs_green_min", "logs_amber_min"),
    "safety_signals": ("probability_amber_min", "probability_red_min"),
    "target_activity": ("potency_green_min", "potency_amber_min"),
    "electronic_stability": ("gap_ev_green_min", "gap_ev_amber_min", "exclude_stub_backend"),
    "confidence": ("ml_interval_width_amber", "dl_sd_amber", "dl_sd_red", "dl_baseline_gap_amber",
                   "failed_tiers_amber", "failed_tiers_red"),
}


def validate_rules(rules):
    """Raise ValueError naming every missing rule (a half-empty rules file must fail loudly)."""
    missing = []
    for section, keys in REQUIRED_RULES.items():
        block = rules.get(section) if isinstance(rules, dict) else None
        missing += [f"{section}.{key}" for key in keys if not isinstance(block, dict) or key not in block]
    if missing:
        raise ValueError("triage_rules.yaml is missing: " + ", ".join(missing))


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _dig(data, *keys):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def _payload(task):
    """(payload, None) for a finished task with a payload, otherwise (None, reason)."""
    if task is None:
        return None, "NO_TASK"
    status = task.get("status")
    if status == "done":
        payload = task.get("payload")
        return (payload, None) if isinstance(payload, dict) else (None, "NO_DATA")
    if status == "skipped":
        return None, task.get("skip_reason") or "SKIPPED"
    if status == "failed":
        return None, task.get("error") or "FAILED"
    if status == "cancelled":
        return None, "CANCELLED"
    return None, "NOT_FINISHED"


def _unavailable(reason):
    return {"status": UNAVAILABLE, "reason": reason, "reasons": [], "inputs": {}}


def _result(status, reasons, inputs):
    return {"status": status, "reasons": reasons, "inputs": inputs}


def _grade(value, green_min, amber_min):
    if value >= green_min:
        return GREEN
    return AMBER if value >= amber_min else RED


def developability(tasks, rules):
    """ML tier: predicted logS."""
    payload, reason = _payload(tasks.get("ml"))
    if payload is None:
        return _unavailable(reason)
    logs = _num(_dig(payload, "solubility", "logs"))
    if logs is None:
        return _unavailable("NO_DATA")
    cfg = rules["developability"]
    status = _grade(logs, cfg["logs_green_min"], cfg["logs_amber_min"])
    return _result(status, [f"predicted logS {logs:.2f}"], {"logs": logs, "unit": _dig(payload, "solubility", "unit")})


def safety_signals(tasks, rules):
    """ML tier: highest Tox21 assay probability."""
    payload, reason = _payload(tasks.get("ml"))
    if payload is None:
        return _unavailable(reason)
    signals = _dig(payload, "tox21", "signals")
    probs = {}
    if isinstance(signals, dict):
        for name, value in signals.items():
            number = _num(value.get("probability")) if isinstance(value, dict) else None
            if number is not None:
                probs[name] = number
    if not probs:
        return _unavailable("NO_DATA")
    cfg = rules["safety_signals"]
    top = max(probs.values())
    raised = sorted(name for name, value in probs.items() if value >= cfg["probability_amber_min"])
    status = RED if top >= cfg["probability_red_min"] else AMBER if top >= cfg["probability_amber_min"] else GREEN
    reasons = [f"{name} signal {probs[name]:.2f}" for name in raised] or ["no assay signal raised"]
    return _result(status, reasons, {"max_probability": top, "assays_raised": raised})


def target_activity(tasks, rules):
    """DL tier: ensemble mean potency for the selected target."""
    payload, reason = _payload(tasks.get("dl"))
    if payload is None:
        return _unavailable(reason)
    mean = _num(_dig(payload, "activity", "mean"))
    if mean is None:
        return _unavailable("NO_DATA")
    cfg = rules["target_activity"]
    status = _grade(mean, cfg["potency_green_min"], cfg["potency_amber_min"])
    return _result(status, [f"predicted potency {mean:.2f}"],
                   {"mean": mean, "target": _dig(payload, "target", "id")})


def electronic_stability(tasks, rules):
    """QML tier: HOMO-LUMO gap. Unconverged runs are excluded."""
    payload, reason = _payload(tasks.get("qml"))
    if payload is None:
        return _unavailable(reason)
    cfg = rules["electronic_stability"]
    if _dig(payload, "convergence", "converged") is False:
        return _unavailable("NOT_CONVERGED")
    stub = bool(payload.get("is_stub"))
    if stub and cfg["exclude_stub_backend"]:
        return _unavailable("STUB_BACKEND")
    gap = _num(_dig(payload, "electronic_stability", "value"))
    if gap is None:
        return _unavailable("NO_DATA")
    status = _grade(gap, cfg["gap_ev_green_min"], cfg["gap_ev_amber_min"])
    reasons = [f"HOMO-LUMO gap {gap:.2f} eV"] + (["stub backend: illustrative only"] if stub else [])
    return _result(status, reasons, {"gap_ev": gap, "is_stub": stub})


def confidence(tasks, rules):
    """How much to trust the other four: uncertainty, tier disagreement, failures, domain."""
    cfg = rules["confidence"]
    levels, reasons, seen = [], [], False
    failed = sorted(t["tier"] for t in tasks.values() if t.get("status") == "failed")
    if len(failed) >= cfg["failed_tiers_red"]:
        levels.append(RED)
    elif len(failed) >= cfg["failed_tiers_amber"]:
        levels.append(AMBER)
    if failed:
        reasons.append("tier failed: " + ", ".join(failed))

    ml, _ = _payload(tasks.get("ml"))
    if ml is not None:
        seen = True
        low, high = _num(_dig(ml, "solubility", "interval", "low")), _num(_dig(ml, "solubility", "interval", "high"))
        if low is not None and high is not None and high - low > cfg["ml_interval_width_amber"]:
            levels.append(AMBER)
            reasons.append(f"wide solubility interval ({high - low:.1f} log units)")
        if _dig(ml, "domain", "in_domain") is False:
            levels.append(AMBER)
            reasons.append("outside the applicability domain")
    dl, _ = _payload(tasks.get("dl"))
    if dl is not None:
        seen = True
        sd = _num(_dig(dl, "activity", "sd"))
        if sd is not None and sd >= cfg["dl_sd_red"]:
            levels.append(RED)
            reasons.append(f"ensemble disagreement SD {sd:.2f}")
        elif sd is not None and sd >= cfg["dl_sd_amber"]:
            levels.append(AMBER)
            reasons.append(f"ensemble disagreement SD {sd:.2f}")
        mean, base = _num(_dig(dl, "activity", "mean")), _num(_dig(dl, "baseline", "value"))
        if mean is not None and base is not None and abs(mean - base) >= cfg["dl_baseline_gap_amber"]:
            levels.append(AMBER)
            reasons.append(f"ensemble and baseline differ by {abs(mean - base):.2f}")
    qml, _ = _payload(tasks.get("qml"))
    seen = seen or qml is not None
    if not seen and not failed:
        return _unavailable("NO_RESULTS")
    status = max(levels, key=RANK.get) if levels else GREEN
    return _result(status, reasons or ["no uncertainty warning"], {"failed_tiers": failed})


def _tier_entry(task):
    return {"status": task.get("status"), "reason": task.get("skip_reason") or task.get("error")}


def build_triage(tasks, rules, now=None):
    """Build the triage summary for one job from its final tasks."""
    validate_rules(rules)
    by_tier = {task["tier"]: task for task in tasks}
    dimensions = {
        "developability": developability(by_tier, rules),
        "safety_signals": safety_signals(by_tier, rules),
        "target_activity": target_activity(by_tier, rules),
        "electronic_stability": electronic_stability(by_tier, rules),
        "confidence": confidence(by_tier, rules),
    }
    return {
        "rules_version": rules.get("version"),
        "rules_status": rules.get("status"),
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "dimensions": dimensions,
        "tiers": {tier: _tier_entry(task) for tier, task in by_tier.items()},
        "flags": flags.cross_signal_flags(dimensions, by_tier),
        "limitation": LIMITATION,
    }
