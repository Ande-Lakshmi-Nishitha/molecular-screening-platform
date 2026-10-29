"""Log redaction (SEC-007): strip molecular structures from log output.

Values of the keys smiles, structure, molfile and inchikey are replaced with
[REDACTED] in the log message, its arguments and any exception text.
"""
import logging
import re

SENSITIVE_KEYS = ("smiles", "structure", "molfile", "inchikey")
REDACTED = "[REDACTED]"

_PATTERN = re.compile(
    r"""(["']?(?:%s)["']?\s*[:=]\s*)("[^"]*"|'[^']*'|[^\s,}&]+)""" % "|".join(SENSITIVE_KEYS),
    re.IGNORECASE,
)


def redact(text):
    """Return text with the value of every sensitive key replaced."""
    return _PATTERN.sub(lambda m: m.group(1) + REDACTED, text)


class RedactionFilter(logging.Filter):
    """Logging filter that redacts a record in place."""

    def filter(self, record):
        self.scrub(record)
        return True

    @staticmethod
    def scrub(record):
        """Redact message, args and exception text of a log record."""
        record.msg = redact(record.getMessage())
        record.args = ()
        if record.exc_info:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact(record.exc_text)


def install_log_filter():
    """Redact every record at creation time, whichever logger or handler is used."""
    current = logging.getLogRecordFactory()
    if getattr(current, "msp_redacting", False):
        return

    def factory(*args, **kwargs):
        record = current(*args, **kwargs)
        RedactionFilter.scrub(record)
        return record

    factory.msp_redacting = True
    logging.setLogRecordFactory(factory)
