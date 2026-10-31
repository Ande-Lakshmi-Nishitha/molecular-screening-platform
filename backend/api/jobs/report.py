"""PDF and JSON report export (MSP-38; FR-039, BR-009).

Owner: Abheetha.
GET /api/jobs/<id>/report?format=pdf|json, for the job's owner only: 403 for someone else's job,
404 if unknown (same rules as GET /api/jobs/<id>, because both use build_status).

The JSON and the PDF are rendered from one dict built by assemble_report(), so they cannot disagree.
The report carries the disclaimer from backend.core.labels (never its own wording), the rules
version, and the model version and params hash of every tier that produced a result.
The PDF needs reportlab; it is imported only when a PDF is asked for, so JSON works without it.
Register report_bp with one line in api/app.py register_blueprints().
"""
import io
import uuid
from datetime import datetime, timezone
from xml.sax.saxutils import escape

from flask import Blueprint, Response, g, jsonify, request

from backend.api.auth.sessions import error_response, login_required
from backend.api.jobs.status import JobAccessError, build_status
from backend.core import audit
from backend.core.labels import DISCLAIMER
from backend.db.models import Molecule
from backend.db.repositories import jobs
from backend.db.session import session_scope

report_bp = Blueprint("job_report", __name__)

REPORT_VERSION = "1"
FORMATS = ("json", "pdf")


def assemble_report(status, molecule, now=None):
    """Build the report dict from a build_status() body and the molecule's fields. No database access."""
    tiers, models = [], {}
    for task in status.get("tasks", []):
        result = task.get("result") or {}
        if result.get("model_version"):
            models[task["tier"]] = result["model_version"]
        tiers.append({
            "tier": task["tier"],
            "status": task["status"],
            "skip_reason": task.get("skip_reason"),
            "error": task.get("error"),
            "model_version": result.get("model_version"),
            "params_hash": result.get("params_hash"),
            "result": result.get("payload"),
        })
    return {
        "report_version": REPORT_VERSION,
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "disclaimer": DISCLAIMER,
        "job": {
            "id": status["job_id"],
            "status": status["status"],
            "target_id": status.get("target_id"),
            "created_at": status.get("created_at"),
            "finished_at": status.get("finished_at"),
            "cancelled_at": status.get("cancelled_at"),
        },
        "molecule": molecule,
        "versions": {"rules_version": status.get("rules_version"), "models": models},
        "summary": status.get("summary"),
        "tiers": tiers,
    }


def build_report(session, job_id, user_id):
    """Load the caller's job and assemble its report. Raises JobAccessError (403 or 404)."""
    status = build_status(session, job_id, user_id)
    job = jobs.get_job_owned(session, job_id, user_id)
    row = session.get(Molecule, job.molecule_id)
    molecule = {
        "canonical_smiles": row.canonical_smiles,
        "inchikey": row.inchikey,
        "formula": row.formula,
        "heavy_atoms": row.heavy_atoms,
    }
    return assemble_report(status, molecule)


def _flatten(value, prefix=""):
    """Yield (dotted path, text) rows for a nested result payload."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        if all(not isinstance(item, (dict, list, tuple)) for item in value):
            yield prefix, ", ".join("" if item is None else str(item) for item in value)
        else:
            for index, item in enumerate(value):
                yield from _flatten(item, f"{prefix}[{index}]")
    else:
        yield prefix, "" if value is None else str(value)


def render_pdf(report):
    """Render the report dict as PDF bytes (needs reportlab)."""
    # pylint: disable=import-outside-toplevel
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    cell = ParagraphStyle("cell", parent=styles["BodyText"], fontSize=8, leading=10)

    def para(text, style=cell):
        return Paragraph(escape(str(text)), style)

    def table(rows):
        grid = Table([[para(key), para(value)] for key, value in rows], colWidths=[50 * mm, 120 * mm])
        grid.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return grid

    job, molecule, versions = report["job"], report["molecule"], report["versions"]
    story = [
        para("Molecular Screening Platform: job report", styles["Title"]),
        para(report["disclaimer"], styles["BodyText"]),
        Spacer(1, 6 * mm),
        para("Job", styles["Heading2"]),
        table([("Job ID", job["id"]), ("Status", job["status"]), ("Target", job["target_id"] or "none"),
               ("Created", job["created_at"]), ("Finished", job["finished_at"] or ""),
               ("Report generated", report["generated_at"])]),
        para("Molecule", styles["Heading2"]),
        table([("Canonical SMILES", molecule["canonical_smiles"]), ("InChIKey", molecule["inchikey"]),
               ("Formula", molecule["formula"]), ("Heavy atoms", molecule["heavy_atoms"])]),
        para("Versions", styles["Heading2"]),
        table([("Report format", report["report_version"]), ("Rules", versions["rules_version"] or "")]
              + [(f"Model {tier.upper()}", version) for tier, version in versions["models"].items()]),
    ]
    if report.get("summary"):
        story += [para("Job summary", styles["Heading2"]), table(list(_flatten(report["summary"])))]
    for tier in report["tiers"]:
        story.append(para(f"Tier {tier['tier'].upper()}: {tier['status']}", styles["Heading2"]))
        rows = []
        if tier["skip_reason"]:
            rows.append(("Skip reason", tier["skip_reason"]))
        if tier["error"]:
            rows.append(("Error", tier["error"]))
        if tier["model_version"]:
            rows.append(("Model version", tier["model_version"]))
        if tier["params_hash"]:
            rows.append(("Parameters hash", tier["params_hash"]))
        rows += list(_flatten(tier["result"])) if tier["result"] else []
        if rows:
            story.append(table(rows))
    story += [Spacer(1, 6 * mm), para(report["disclaimer"], styles["BodyText"])]

    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, title="MSP job report", leftMargin=15 * mm, rightMargin=15 * mm).build(story)
    return buffer.getvalue()


def _parse(job_id):
    try:
        return uuid.UUID(job_id)
    except ValueError:
        return None


@report_bp.get("/api/jobs/<job_id>/report")
@login_required
def job_report(job_id):
    """Download the caller's job report as JSON (default) or PDF."""
    fmt = (request.args.get("format") or "json").strip().lower()
    if fmt not in FORMATS:
        return error_response("VALIDATION_ERROR", "format must be pdf or json.", 400, {"field": "format"})
    parsed = _parse(job_id)
    if parsed is None:
        return error_response("JOB_NOT_FOUND", "Job not found.", 404)
    user_id = uuid.UUID(g.claims["sub"])
    try:
        with session_scope() as session:
            report = build_report(session, parsed, user_id)
            audit.record(session, "report_exported", user_id, {"job_id": str(parsed), "format": fmt})
    except JobAccessError as err:
        return error_response(err.code, err.message, err.status)

    filename = f"msp-report-{parsed}.{fmt}"
    disposition = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if fmt == "pdf":
        return Response(render_pdf(report), mimetype="application/pdf", headers=disposition)
    response = jsonify(report)
    response.headers.update(disposition)
    return response
