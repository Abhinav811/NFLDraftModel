from __future__ import annotations

import json
import ssl
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import USER_AGENT


def _ssl_unverified() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def fetch_bytes(url: str, dest: Path | None = None, timeout: int = 120) -> bytes:
    """Download bytes via urllib, then curl. Mac Python often lacks certs."""
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    req = urllib.request.Request(url, headers=headers)
    data: bytes | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except (urllib.error.URLError, ssl.SSLError):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_unverified()) as resp:
                data = resp.read()
        except Exception:
            data = None
    if data is None:
        result = subprocess.run(
            ["curl", "-sL", "--max-time", str(timeout), "-A", USER_AGENT, url],
            check=True,
            capture_output=True,
        )
        data = result.stdout
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return data


def fetch_json(url: str, timeout: int = 60) -> Any:
    raw = fetch_bytes(url, timeout=timeout)
    return json.loads(raw.decode("utf-8"))


def fetch_text(url: str, timeout: int = 60) -> str:
    return fetch_bytes(url, timeout=timeout).decode("utf-8", errors="replace")
