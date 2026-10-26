"""Capture the Flask-app screenshots for the report with a headless browser.

Run on YOUR machine against YOUR running app and real model, so the images show your results:

  pip install playwright && playwright install chromium
  python app.py                       # in another terminal (http://127.0.0.1:5000)
  python scripts/capture_screenshots.py --base http://127.0.0.1:5000 \
         --real path/to/real.jpg --ai path/to/ai.jpg

Saves to artifacts/screenshots/: upload page, result page(s), 413 / 415 / 400 error pages,
history page and /health. Browser UI screenshots do not show an address bar, so add your own
terminal / browser-window shots (with username and date visible) where the assignment asks for context.
"""
import argparse
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    ap.add_argument("--real", help="path to a known real image")
    ap.add_argument("--ai", help="path to a known AI image")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "screenshots"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    (tmp / "doc.pdf").write_bytes(b"%PDF-1.4 not an image")
    (tmp / "photo.heic").write_bytes(b"not really heic")
    (tmp / "corrupt.png").write_bytes(b"\x89PNG\r\n\x1a\n" + os.urandom(200))
    (tmp / "huge.jpg").write_bytes(os.urandom(6 * 1024 * 1024))

    def submit(page, file_path, shot):
        page.goto(args.base + "/")
        page.set_input_files("#file", str(file_path))
        page.click("#go")
        page.wait_for_load_state("networkidle", timeout=180000)
        page.screenshot(path=str(out / shot), full_page=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(args.base + "/")
        page.screenshot(path=str(out / "fig_4_5_upload_page.png"), full_page=True)
        for name, shot in [("huge.jpg", "fig_4_2_error_413.png"), ("doc.pdf", "fig_4_3_error_415_pdf.png"),
                           ("photo.heic", "fig_4_4_error_415_heic.png"),
                           ("corrupt.png", "fig_4_x_error_400_corrupt.png")]:
            submit(page, tmp / name, shot)
        if args.real:
            submit(page, args.real, "fig_4_6_result_real.png")
        if args.ai:
            submit(page, args.ai, "fig_4_6_result_ai.png")
        page.goto(args.base + "/history")
        page.screenshot(path=str(out / "fig_5_1_history.png"), full_page=True)
        page.goto(args.base + "/health")
        page.screenshot(path=str(out / "fig_4_7_health.png"))
        browser.close()
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    print("Screenshots saved to", out)


if __name__ == "__main__":
    main()
