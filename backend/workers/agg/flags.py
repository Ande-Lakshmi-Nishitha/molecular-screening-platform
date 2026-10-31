"""Cross-signal flags (FR-034).

Owner: Ayushi. Pure functions: they read the five dimension colours and the ML payload.
The fourth flag in FR-034 (labile fragment overlaps a structural alert) needs the fragment path,
which is descoped in v1.0, so it is not produced.
"""


def _flag(code, message):
    return {"code": code, "message": message}


def _in_domain_false(task):
    payload = task.get("payload") if task and task.get("status") == "done" else None
    domain = payload.get("domain") if isinstance(payload, dict) else None
    return isinstance(domain, dict) and domain.get("in_domain") is False


def cross_signal_flags(dimensions, tasks):
    """Return the flags whose conditions hold, as a list of {code, message}."""
    def colour(name):
        return dimensions.get(name, {}).get("status")

    found = []
    potent = colour("target_activity") == "green"
    if potent and colour("developability") == "red":
        found.append(_flag("POTENT_POORLY_SOLUBLE",
                           "Predicted potent against the target but predicted poorly soluble."))
    if potent and colour("safety_signals") in ("amber", "red"):
        found.append(_flag("POTENT_TOX21_SIGNAL",
                           "Predicted potent against the target but a Tox21 signal is raised."))
    good = all(colour(name) == "green" for name in ("developability", "safety_signals", "target_activity"))
    if good and _in_domain_false(tasks.get("ml")):
        found.append(_flag("GOOD_PROFILE_OUT_OF_DOMAIN",
                           "Good predicted profile, but the predictions are outside the applicability domain."))
    return found
