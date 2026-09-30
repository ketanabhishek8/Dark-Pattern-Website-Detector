"""Renders the report figures with headless Chrome: the architecture diagram and a dashboard screenshot.

Needs Google Chrome and Pillow, and the dashboard running (python app/server.py) for the screenshot.
Usage: python report/render_figures.py
"""
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

HERE = Path(__file__).parent
FIG = HERE.parent / "figures"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def screenshot(url, out, width, height, scale=2):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-prefers-reduced-motion",
                    f"--force-device-scale-factor={scale}", f"--window-size={width},{height}",
                    "--blink-settings=preferredColorScheme=1", "--virtual-time-budget=30000",
                    f"--screenshot={out}", url], check=True, capture_output=True)
    return Image.open(out).convert("RGB")


def trim(img, pad=16):
    """Crops away the white margin around the content."""
    box = ImageChops.difference(img, Image.new("RGB", img.size, (255, 255, 255))).getbbox()
    return img.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(img.width, box[2] + pad), min(img.height, box[3] + pad)))


with tempfile.TemporaryDirectory() as tmp:
    arch = screenshot((HERE / "architecture.html").as_uri(), f"{tmp}/arch.png", 1240, 420)
    trim(arch).save(FIG / "architecture.png")

    # Dashboard after scanning the sample shop: from the page strip down to the end of the bar chart
    dash = screenshot("http://localhost:8000/?demo", f"{tmp}/dash.png", 1280, 2000)
    top, bottom = 690, 1480  # CSS pixels at 1280 px wide
    dash.crop((90 * 2, top * 2, 1190 * 2, bottom * 2)).save(FIG / "dashboard_screenshot.png")

print("Saved figures/architecture.png and figures/dashboard_screenshot.png")
