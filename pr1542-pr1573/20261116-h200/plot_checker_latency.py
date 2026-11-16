"""Plot the six paused weight-checker calls from the pinned live JSON files."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
FRONTENDS = ("python", "rust")
COLORS = {"python": "#2f6db3", "rust": "#d07831"}
LABELS = ("first checksum", "matching compare", "after reload")
rows = {}
for frontend in FRONTENDS:
    report = json.loads((HERE / "raw" / f"pr1573-{frontend}-orpheus-live.json").read_text())
    rows[frontend] = [
        float(row["elapsed_s"])
        for row in report["rows"]
        if row["path"] == "/weight_checker" and row["status"] == 200
    ]
    assert len(rows[frontend]) == 3

out = [
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 470" role="img" '
    'aria-label="Paused weight checker latency on one H200 for Python and Rust frontends">',
    '<rect width="900" height="470" fill="white"/>',
    '<text x="76" y="36" font-family="Arial,sans-serif" font-size="22" fill="#17212c">'
    'Paused weight checker: one H200 live run</text>',
    '<text x="76" y="58" font-family="Arial,sans-serif" font-size="13" fill="#526274">'
    'One worker rank, real Orpheus FT weights; one call per operation and frontend</text>',
]

base_y, plot_h = 379, 272
for tick in (0, 2, 4, 6, 8):
    y = base_y - tick / 9 * plot_h
    out.append(f'<line x1="76" x2="860" y1="{y:.1f}" y2="{y:.1f}" stroke="#dce3ea"/>')
    out.append(
        f'<text x="65" y="{y + 5:.1f}" text-anchor="end" font-family="Arial,sans-serif" '
        f'font-size="12" fill="#526274">{tick}</text>'
    )
out.append('<text x="20" y="244" transform="rotate(-90 20 244)" '
           'font-family="Arial,sans-serif" font-size="13" fill="#34465a">Seconds</text>')

for group, label in enumerate(LABELS):
    center = 190 + group * 267
    for offset, frontend in ((-37, "python"), (12, "rust")):
        value = rows[frontend][group]
        height = value / 9 * plot_h
        x, y = center + offset, base_y - height
        out.append(
            f'<rect x="{x}" y="{y:.1f}" width="43" height="{height:.1f}" '
            f'fill="{COLORS[frontend]}" rx="2"/>'
        )
        out.append(
            f'<text x="{x + 21.5:.1f}" y="{y - 7:.1f}" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="12" fill="#17212c">{value:.3f}</text>'
        )
    out.append(
        f'<text x="{center}" y="403" text-anchor="middle" font-family="Arial,sans-serif" '
        f'font-size="13" fill="#34465a">{label}</text>'
    )

out.extend([
    '<rect x="322" y="424" width="15" height="15" fill="#2f6db3"/>',
    '<text x="344" y="437" font-family="Arial,sans-serif" font-size="13" fill="#34465a">Python</text>',
    '<rect x="431" y="424" width="15" height="15" fill="#d07831"/>',
    '<text x="453" y="437" font-family="Arial,sans-serif" font-size="13" fill="#34465a">Rust</text>',
    '<text x="76" y="457" font-family="Arial,sans-serif" font-size="11" fill="#69798a">'
    'Functional probe at source head 5b42f8e906; these are not throughput or repeated timing measurements.</text>',
    '</svg>',
])
(HERE / "weight-checker-latency.svg").write_text("\n".join(out) + "\n")
