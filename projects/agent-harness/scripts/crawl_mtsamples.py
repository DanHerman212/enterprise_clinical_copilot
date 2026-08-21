"""crawl_mtsamples.py — download the MTSamples discharge-summary corpus.

One-shot (or re-runnable) crawler that pulls all 108 discharge-summary samples
from MTSamples and stores each as clean text.

MTSamples is behind a Cloudflare challenge that blocks plain requests, so this
uses Playwright (a real browser) which passes it. Output is stored under the
gitignored `projects/agent-harness/data/mtsamples/` directory — raw note text is
NEVER committed (same posture as MIMIC note text).

Provenance is recorded in a manifest. This is a dev/test dataset; the public
demo uses the hybrid/derived version.

Usage:
    .venv/bin/python scripts/crawl_mtsamples.py [--limit N] [--out DIR]
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

HARNESS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = HARNESS_ROOT / "data" / "mtsamples"

BROWSE_URL = "https://www.mtsamples.com/site/pages/browse.asp?type=89-Discharge+Summary"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# The note body sits between the first clinical section header and the
# "About This Sample" footer. Anchors are matched at LINE START (multiline) so
# nav boilerplate like "Summary - Medical Reports" or "History of Present
# Illness" in a menu is never mistaken for the clinical body. The real
# discharge summary always opens with one of these headers.
BODY_START_RE = re.compile(
    r"^(ADMISSION DIAGNOSIS|ADMITTING DIAGNOSIS|DISCHARGE DIAGNOSIS|"
    r"FINAL DIAGNOSIS|PRESENT ILLNESS|CHIEF COMPLAINT|SUMMARY OF ADMISSION)",
    re.IGNORECASE | re.MULTILINE,
)
BODY_END = "About This Sample"


def _clean_text(raw: str) -> str:
    """Strip the MTSamples nav boilerplate, keep the note body + title/desc."""
    lines = [ln.rstrip() for ln in raw.splitlines()]
    text = "\n".join(lines)

    # Title + description come first; the body starts at the first clinical
    # section header. Keep everything from the body start to "About This Sample".
    m = BODY_START_RE.search(text)
    start = m.start() if m else 0
    end = text.find(BODY_END)
    body = text[start:end] if end > start else text[start:]
    # Collapse 3+ blank lines to 1 (MTSamples uses wide spacing).
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="max samples (0=all 108)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        # headless=False avoids Cloudflare flagging headless-shell on
        # navigation; --no-sandbox keeps CI/dev environments working.
        browser = pw.chromium.launch(headless=False, args=["--no-sandbox"])
        ctx = browser.new_context(user_agent=UA)
        page = ctx.new_page()

        print("collecting sample URLs…")
        page.goto(BROWSE_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(3000)  # let the list render
        urls = page.eval_on_selector_all(
            'a[href*="sample.asp"]', "els => els.map(a => a.href)"
        )
        # Keep only discharge-summary URLs, deduped by the NUMERIC sample id.
        # (The page lists each sample under several encodings — navbar dropdown,
        # main table, related links — so deduping by full name over-counts.)
        by_id: dict[str, str] = {}
        for u in urls:
            if "Type=89-Discharge" not in u:
                continue
            m = re.search(r"Sample=(\d+)", u)
            if m and m.group(1) not in by_id:
                by_id[m.group(1)] = u
        samples = [by_id[k] for k in sorted(by_id, key=int)]
        print(f"found {len(samples)} unique discharge-summary samples")

        if args.limit:
            samples = samples[: args.limit]

        manifest = {
            "source": "https://www.mtsamples.com (Discharge Summary category)",
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "count": len(samples),
            "files": [],
        }
        for i, url in enumerate(samples, 1):
            sid = re.search(r"Sample=(\d+)", url).group(1)
            safe = sid
            # Fresh context per sample: a single slow navigation passes
            # Cloudflare, but a long-lived context doing many navigations gets
            # flagged and then EVERY page shows the interstitial. Fresh context
            # = each sample starts clean (like the verified single-load test).
            ctx = browser.new_context(user_agent=UA, viewport={"width": 1280, "height": 900})
            page = ctx.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded")
                # Cloudflare may show a "Performing security verification"
                # interstitial on navigation. Wait it out before reading the body.
                raw = ""
                for _ in range(60):  # up to ~60s
                    raw = page.evaluate("() => document.body.innerText")
                    if "security verification" not in raw.lower():
                        break
                    page.wait_for_timeout(1000)
                body = _clean_text(raw)
                if len(body) < 100:
                    print(f"  [{i}/{len(samples)}] WARN short body ({len(body)}) "
                          f"for {safe}")
                fname = f"{safe}.txt"
                (out_dir / fname).write_text(body, encoding="utf-8")
                manifest["files"].append({"id": sid, "file": fname,
                                          "chars": len(body)})
                print(f"  [{i}/{len(samples)}] {safe} ({len(body)} chars)")
            except Exception as exc:
                print(f"  [{i}/{len(samples)}] ERROR {safe}: {exc}")
            finally:
                ctx.close()
            # Brief pause between samples (politeness + anti-bot pacing).
            time.sleep(1.0 + (i % 3) * 0.5)

        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nwrote {len(manifest['files'])} samples to {out_dir}")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
