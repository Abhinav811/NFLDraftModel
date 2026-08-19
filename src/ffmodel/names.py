from __future__ import annotations

import re
import unicodedata

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.I)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str | None) -> str:
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", "").replace(".", " ")
    text = _SUFFIX.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
