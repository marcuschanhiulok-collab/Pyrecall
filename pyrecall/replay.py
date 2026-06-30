from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

from .utils import get_logger, safe_model_name

def _save(self) -> None:
    meta = {
        "total_seen": self._total_seen,
        "max_size": self._max_size,
        "seen_hashes": list(self._seen_hashes),
    }

    lines = [json.dumps(meta)]
    lines.extend(
        json.dumps(
            {
                "text": entry["text"],
                "category": entry.get("category"),
            }
        )
        for entry in self._buffer
    )

    content = "\n".join(lines) + "\n"

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=self._path.parent,
        prefix=".buffer_tmp_",
    )

    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(tmp_path, self._path)

        if os.name != "nt":
            dir_fd = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise
