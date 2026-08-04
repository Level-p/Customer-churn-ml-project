"""Re-capture the two dashboard screenshots used as figures in the paper.

The screenshots in ``docs/figures/`` are full-page captures 2,880 px wide. Placed
in a printed figure they reduce body text to roughly four points. These are taken
at a narrow viewport instead, so the layout reflows compactly and the same text
survives the reduction to page width at a readable size.

Requires the development server to be running:

    cd bdss_project && python manage.py runserver 8811 --noreload
    python -m docs.capture_paper_screenshots
"""

from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = os.environ.get("BDSS_BASE_URL", "http://127.0.0.1:8811")
USER = os.environ.get("BDSS_USER", "admin")
PASSWORD = os.environ.get("BDSS_PASSWORD", "retention2026")
OUT = Path(__file__).resolve().parent / "paper_figures"

VIEWPORT = {"width": 1120, "height": 1000}
SCALE = 3  # 3,360 px of image for 1,120 CSS px: sharp at page width

FULL = OUT / "_batch-full.png"

# Regions of the full-page capture, in image pixels. The batch summary is the
# header, the four counters and the two batch-level charts; the queue figure is
# its heading row and the first two customers, which between them carry the
# column headings, a contested driver and the demographic guard. Two rows rather
# than three because the figure spans the page in an eight-page paper, and a
# third row costs more space than the extra recommendation is worth.
SUMMARY_BOX = (40, 260, 3320, 2010)
QUEUE_BOX = (40, 3260, 3320, 4760)


def crop(source: Path, target: Path, box: tuple[int, int, int, int]) -> None:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(source) as image:
        image.crop(box).save(target, optimize=True)
    print(f"  wrote {target.name}  {box[2] - box[0]}x{box[3] - box[1]}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as play:
        browser = play.chromium.launch()
        page = browser.new_context(
            viewport=VIEWPORT, device_scale_factor=SCALE
        ).new_page()

        page.goto(f"{BASE}/accounts/login/", wait_until="networkidle")
        page.fill("input[name=username]", USER)
        page.fill("input[name=password]", PASSWORD)
        page.click("button[type=submit], input[type=submit]")
        page.wait_for_load_state("networkidle")

        page.goto(f"{BASE}/history/", wait_until="networkidle")
        link = page.locator("a[href^='/batch/']").first
        href = link.get_attribute("href")
        batch = href.strip("/").split("/")[-1]
        print(f"  batch {batch}")

        page.goto(f"{BASE}/batch/{batch}/", wait_until="networkidle")
        page.wait_for_timeout(1200)
        page.screenshot(path=str(FULL), full_page=True)
        browser.close()

    crop(FULL, OUT / "fig07-dashboard-summary.png", SUMMARY_BOX)
    crop(FULL, OUT / "fig08-dashboard-queue.png", QUEUE_BOX)
    FULL.unlink()
    print("  captured")


if __name__ == "__main__":
    main()
