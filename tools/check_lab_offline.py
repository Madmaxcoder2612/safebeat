"""Fail if the lab could contact another server.

Scans lab/index.html, lab/lab.js and lab/style.css for web addresses, and checks
that the page still carries its Content-Security-Policy. The only web address
allowed is the SVG namespace string, which is an identifier and not a request.

Run from the repository root:  python tools/check_lab_offline.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent / "lab"
ALLOWED = {"http://www.w3.org/2000/svg"}
FILES = ["index.html", "lab.js", "style.css"]


def main() -> int:
    problems: list[str] = []
    for name in FILES:
        text = (LAB / name).read_text(encoding="utf-8")
        for url in re.findall(r"(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}[^\s\"')]*", text):
            if url not in ALLOWED:
                problems.append(f"{name}: outside address {url}")
        if re.search(r"@import|fetch\(|XMLHttpRequest|WebSocket|sendBeacon|localStorage|sessionStorage|document\.cookie", text):
            problems.append(f"{name}: network or storage API found")
    html = (LAB / "index.html").read_text(encoding="utf-8")
    if "Content-Security-Policy" not in html or "connect-src 'none'" not in html:
        problems.append("index.html: Content-Security-Policy with connect-src 'none' is missing")
    for tag in re.findall(r"<(?:script|link|img|iframe|source)[^>]+>", html):
        m = re.search(r'(?:src|href)="([^"]+)"', tag)
        if m and re.match(r"(?:https?:)?//", m.group(1)):
            problems.append(f"index.html: external resource {m.group(1)}")
    if problems:
        print("\n".join(problems))
        return 1
    print("Lab is offline: no outside addresses, no network or storage APIs, CSP present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
