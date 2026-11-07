"""Structured (JSON-line) logging."""
import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": datetime.now(timezone.utc).isoformat(), "level": record.levelname,
                   "logger": record.name, "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str = "flfraud", level: str = "INFO") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
        log.setLevel(level)
        log.propagate = False
    return log
