"""Plot the one same-checkpoint reload call per frontend from pinned live JSON."""

import json
from pathlib import Path


here = Path(__file__).resolve().parent
values = {}
for frontend in ("python", "rust"):
    report = json.loads((here / "raw" / f"pr1573-{frontend}-orpheus-live.json").read_text())
    calls = [
        row for row in report["rows"]
        if row["path"] == "/update_weights_from_disk" and row["status"] == 200
    ]
    assert len(calls) == 1
    values[frontend] = float(calls[0]["elapsed_s"])

out = [
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 430" role="img" '
    'aria-label="Same-checkpoint Orpheus reload wall time on one H200">',
    '<rect width="680" height="430" fill="white"/>',
    '<text x="65" y="37" font-family="Arial,sans-serif" font-size="21" fill="#17212c">'
    'Same-checkpoint Orpheus reload</text>',
    '<text x="65" y="59" font-family="Arial,sans-serif" font-size="13" fill="#526274">'
    'One H200, one rank, one call per frontend</text>',
]
base_y, plot_h = 344, 245
for tick in range(0, 6):
    y = base_y - tick / 5 * plot_h
    out.append(f'<line x1="65" x2="645" y1="{y:.1f}" y2="{y:.1f}" stroke="#dce3ea"/>')
    out.append(
        f'<text x="53" y="{y + 5:.1f}" text-anchor="end" font-family="Arial,sans-serif" '
        f'font-size="12" fill="#526274">{tick}</text>'
    )
out.append('<text x="19" y="222" transform="rotate(-90 19 222)" '
           'font-family="Arial,sans-serif" font-size="13" fill="#34465a">Seconds</text>')
for frontend, x, color in (("python", 194, "#2f6db3"), ("rust", 425, "#d07831")):
    value = values[frontend]
    height = value / 5 * plot_h
    y = base_y - height
    out.append(f'<rect x="{x}" y="{y:.1f}" width="76" height="{height:.1f}" fill="{color}" rx="2"/>')
    out.append(
        f'<text x="{x + 38}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial,sans-serif" '
        f'font-size="13" fill="#17212c">{value:.3f}</text>'
    )
    out.append(
        f'<text x="{x + 38}" y="369" text-anchor="middle" font-family="Arial,sans-serif" '
        f'font-size="13" fill="#34465a">{frontend.title()}</text>'
    )
out.extend([
    '<text x="65" y="409" font-family="Arial,sans-serif" font-size="11" fill="#69798a">'
    'Functional probe at 5b42f8e906; same weights before/after, no repeated timing or throughput.</text>',
    '</svg>',
])
(here / "same-checkpoint-reload-latency.svg").write_text("\n".join(out) + "\n")
