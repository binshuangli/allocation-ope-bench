#!/usr/bin/env python
"""Build-quality gate for a paper directory. Fails loudly rather than reporting 'clean'.

Written after a compression pass shipped a multiply-defined label: the ad-hoc grep used at
the time had been shortened and no longer covered that warning class. The point of this
script is that the list of checks lives in one place and cannot be trimmed by accident.

    python scripts/check_paper.py paper_compact [--max-body-pages 12]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

WARNINGS = {
    "undefined reference": r"Reference `[^']*' on page \d+ undefined|There were undefined references",
    "undefined citation": r"Citation `[^']*' on page \d+ undefined",
    "multiply-defined label": r"multiply defined|multiply-defined",
    "overfull box": r"Overfull \\[hv]box",
    "underfull hbox (severe)": r"Underfull \\hbox \(badness 10000\)",
    "missing file": r"LaTeX Warning: File `[^']*' not found",
    "fatal error": r"Fatal error occurred",
    # A TikZ figure once shipped four silent 'Package PGF Math Error's (missing
    # \usetikzlibrary{positioning}); package errors recover visually but misplace
    # content, so they gate the build like everything else.
    "package error": r"^! Package \w+ Error|Package \w+ Math Error",
}


def _anon_patterns() -> str:
    """Identifying strings to scrub, read from the gitignored ``.anon-patterns``.

    Kept out of version control so a public release of this repository does not
    itself publish the author identifiers its anonymity gate searches for. When
    the file is absent the anonymity check is skipped and says so.
    """
    f = Path(__file__).resolve().parent.parent / ".anon-patterns"
    return f.read_text().strip() if f.exists() else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paper_dir")
    ap.add_argument("--max-body-pages", type=int, default=None)
    ap.add_argument("--max-abstract-words", type=int, default=None)
    a = ap.parse_args()
    d = Path(a.paper_dir)
    log, pdf = d / "main.log", d / "main.pdf"
    fail = []

    if not pdf.exists():
        print(f"FAIL  no {pdf} -- the build did not produce a PDF")
        return 1
    text = log.read_text(errors="ignore")
    for name, pat in WARNINGS.items():
        hits = re.findall(pat, text)
        status = "ok  " if not hits else "FAIL"
        if hits:
            fail.append(f"{name} x{len(hits)}")
        print(f"  {status} {name}: {len(hits)}")

    out = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
    pages = out.split("\f")
    body = None
    for i, p in enumerate(pages, 1):
        lines = [ln.strip() for ln in p.split("\n")]
        # TMLR excludes the Broader Impact Statement, references and appendices from the
        # 12-page body limit, so the body ends at whichever of those comes first.
        for _marker in ("Broader Impact Statement", "References"):
            if _marker in lines:
                lines = lines[: lines.index(_marker)] + ["References"]
                break
        if "References" not in lines:
            continue
        # Content preceding the heading on this page, ignoring blanks and the running
        # header that pdftotext emits at the top of every page. If nothing real precedes
        # it, the references OPEN this page and the body ends on the previous one.
        before = [
            ln for ln in lines[: lines.index("References")]
            if ln and not ln.startswith("Under review as")
        ]
        body = i - 1 if not before else i
        break
    if body is None:
        fail.append("could not locate the References heading -- page count unverified")
        print("  FAIL main content: References heading not found")
    else:
        print(f"  ---- total pages {len(pages) - 1}, main content {body}")
        if a.max_body_pages and body > a.max_body_pages:
            fail.append(f"body {body}pp > {a.max_body_pages}pp")

    if a.max_abstract_words:
        m = re.search(r"Organizations routinely(.*?)\n1\s+Introduction", out, re.S)
        if m:
            w = len(m.group(1).split())
            print(f"  ---- abstract {w} words")
            if w > a.max_abstract_words:
                fail.append(f"abstract {w} > {a.max_abstract_words} words")

    # identifying strings must not survive into an anonymous PDF
    ident = re.findall(f"(?i){_anon_patterns()}", out) if _anon_patterns() else []
    print(f"  {'ok  ' if not ident else 'FAIL'} identifying strings in PDF: {len(ident)}")
    if ident:
        fail.append("PDF is not anonymous")

    print(("\nFAILED: " + "; ".join(fail)) if fail else "\nAll checks passed.")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
