"""Optional visual QA: pip install playwright; playwright install chromium.

Checks offline rendering, JS errors, all charts, hover interaction and mobile overflow.
Use --browser-executable to reuse an existing Chromium installation.
"""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-executable")
    args = parser.parse_args()
    errors, remote_requests = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=args.browser_executable)
        context = browser.new_context(viewport={"width": 1440, "height": 1100}, offline=True)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda req: remote_requests.append(req.url) if req.url.startswith("http") else None,
        )
        page.goto((ROOT / "reports/demo/report.html").as_uri())
        page.wait_for_function(
            "document.querySelectorAll('.plotly-graph-div.js-plotly-plot').length === 6"
        )
        assert page.locator(".kpi").count() == 4
        assert page.locator(".panel").count() == 6
        assert page.locator(".scatterlayer .trace").count() >= 4
        page.locator(".barlayer .point path").first.hover(force=True)
        page.wait_for_function("document.querySelector('.hoverlayer .hovertext') !== null")
        page.mouse.move(5, 5)
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=str(ROOT / "docs/report-preview.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_function("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(ROOT / "reports/local/mobile-preview.png"), full_page=True)
        assert not errors, errors
        assert not remote_requests, remote_requests
        print(
            "PASS: 6 charts, hover, offline assets, no JS errors, mobile without horizontal overflow."
        )
        context.close()
        browser.close()


if __name__ == "__main__":
    main()
