"""Local web page for the preprocessing pipeline.

Run:

    python web_app.py

Then open http://127.0.0.1:8000 in a browser, select an image, and the page
shows the original and the processed image side by side, with the results.

The server uses only the Python standard library and listens on this computer
only (127.0.0.1). An uploaded image is processed in a temporary file, which is
deleted after the result is sent. Nothing is saved.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2

import pest_pipeline as pp

MAX_UPLOAD = 40 * 1024 * 1024   # bytes

STAGE_LABELS = {
    "load": "Load and decode", "work_resize": "Working resize",
    "iqa_restore": "Quality check and restoration", "mask": "Green background mask",
    "enhance": "CLAHE and sharpening", "final_resize": "Final resize and padding",
    "save": "JPEG encode",
}
FILTER_LABELS = {
    "none": "No change", "median": "Median", "gaussian": "Gaussian", "wiener": "Wiener",
    "richardson_lucy": "Richardson-Lucy", "total_variation": "Total variation", "sharpen": "Unsharp masking",
}

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pest image preprocessing</title>
<style>
  :root { --bg:#fcfcfb; --card:#ffffff; --ink:#1f1e1b; --ink2:#5f5d57; --line:#e1e0d9; --blue:#2a78d6; --orange:#eb6834; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  main { max-width:1080px; margin:0 auto; padding:28px 16px 60px; }
  h1 { font-size:22px; margin:0 0 4px; }
  h2 { font-size:15px; margin:0 0 10px; }
  p.sub { margin:0 0 20px; color:var(--ink2); }
  .bar { display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin-bottom:22px; }
  .btn { background:var(--blue); color:#fff; border:0; border-radius:8px; padding:10px 18px; font:inherit; font-weight:600; cursor:pointer; }
  .btn:hover { filter:brightness(0.94); }
  #status { color:var(--ink2); }
  #status.err { color:#b3261e; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:720px) { .grid { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
  .card img { width:100%; height:auto; display:block; border-radius:6px; background:#000; }
  .cap { color:var(--ink2); font-size:13px; margin-top:8px; }
  .tiles { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:16px 0; }
  @media (max-width:720px) { .tiles { grid-template-columns:repeat(2,1fr); } }
  .tile { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
  .tile .k { color:var(--ink2); font-size:12px; }
  .tile .v { font-size:20px; font-weight:650; }
  .tile .n { color:var(--ink2); font-size:12px; }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  td { padding:6px 4px; border-bottom:1px solid var(--line); }
  td:last-child { text-align:right; font-variant-numeric:tabular-nums; }
  tr:last-child td { border-bottom:0; }
  #result { display:none; }
  .section { margin-top:16px; }
</style>
</head>
<body>
<main>
  <h1>Pest image preprocessing</h1>
  <p class="sub">Upload one image. The full pipeline runs on this computer: quality check and restoration, green background mask, CLAHE, resize to 512 &times; 512, JPEG.</p>

  <div class="bar">
    <input type="file" id="file" accept="image/*" hidden>
    <button class="btn" id="pick">Upload image</button>
    <span id="status">No image selected.</span>
  </div>

  <div id="result">
    <div class="grid">
      <div class="card"><h2>Original</h2><img id="img-original" alt="Original image"><div class="cap" id="cap-original"></div></div>
      <div class="card"><h2>Processed</h2><img id="img-final" alt="Processed image"><div class="cap" id="cap-final"></div></div>
    </div>

    <div class="tiles" id="tiles"></div>

    <div class="grid">
      <div class="card"><h2>Background mask (white = kept)</h2><img id="img-mask" alt="Mask"></div>
      <div class="card"><h2>CLAHE enhanced, before the mask</h2><img id="img-enhanced" alt="Enhanced image"></div>
    </div>

    <div class="grid section">
      <div class="card"><h2>Results</h2><table id="tbl-results"></table></div>
      <div class="card"><h2>Time for each stage</h2><table id="tbl-times"></table></div>
    </div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const status = (text, err) => { $("status").textContent = text; $("status").className = err ? "err" : ""; };
const rows = (table, pairs) => {
  table.replaceChildren(...pairs.map(([k, v]) => {
    const tr = document.createElement("tr");
    for (const text of [k, v]) { const td = document.createElement("td"); td.textContent = text; tr.appendChild(td); }
    return tr;
  }));
};
const tile = (k, v, n) => {
  const d = document.createElement("div"); d.className = "tile";
  for (const [cls, text] of [["k", k], ["v", v], ["n", n]]) {
    const e = document.createElement("div"); e.className = cls; e.textContent = text; d.appendChild(e);
  }
  return d;
};

$("pick").onclick = () => $("file").click();
$("file").onchange = async () => {
  const file = $("file").files[0];
  if (!file) return;
  status("Processing " + file.name + " ...");
  try {
    const resp = await fetch("/process", { method: "POST", body: file,
      headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(file.name) } });
    const r = await resp.json();
    if (!resp.ok) throw new Error(r.error || "The server could not process the image.");
    for (const k of ["original", "final", "mask", "enhanced"]) $("img-" + k).src = r.images[k];
    $("cap-original").textContent = `${r.raw_width} × ${r.raw_height} px, ${r.raw_kb} kB`;
    $("cap-final").textContent = `${r.size} × ${r.size} px, ${r.processed_kb} kB`;
    $("tiles").replaceChildren(
      tile("File size", `${r.raw_kb} → ${r.processed_kb} kB`, `${r.reduction_percent}% less`),
      tile("Processing time", `${r.total_ms} ms`, "all stages"),
      tile("BRISQUE (lower is better)", `${r.brisque_before} → ${r.brisque_after}`, r.low_quality ? "low quality: restored" : "good quality: not changed"),
      tile("Background mask", r.mask_applied ? "Applied" : "Not applied", `${r.foreground_percent}% of pixels kept`));
    rows($("tbl-results"), r.results);
    rows($("tbl-times"), r.times);
    $("result").style.display = "block";
    status("Done: " + file.name);
  } catch (e) {
    status(e.message, true);
  }
  $("file").value = "";
};
</script>
</body>
</html>
"""


def data_uri(bgr_or_gray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", bgr_or_gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("cannot encode the image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def process_upload(data: bytes, suffix: str) -> dict | None:
    """Run the pipeline on the uploaded bytes. Returns None if it is not an image."""
    fd, name = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        res = pp.process_image(Path(name))
    finally:
        os.unlink(name)
    if res is None:
        return None

    r, m = res.restore, res.mask
    raw_kb, proc_kb = len(data) / 1024, len(res.jpeg) / 1024
    total_ms = sum(res.times_ms.values())
    results = [
        ("Original size", f"{res.raw_shape[1]} x {res.raw_shape[0]} px, {raw_kb:.1f} kB"),
        ("Processed size", f"{pp.TARGET_SIZE} x {pp.TARGET_SIZE} px, {proc_kb:.1f} kB"),
        ("Size reduction", f"{100 * (1 - proc_kb / raw_kb):.1f}%"),
        ("BRISQUE before", f"{r.brisque_before:.2f}"),
        ("BRISQUE after restoration", f"{r.brisque_after:.2f}"),
        (f"Low quality (BRISQUE > {pp.BRISQUE_THRESHOLD:.0f})", "Yes" if r.flagged else "No"),
        ("Restoration filter selected", FILTER_LABELS.get(r.filter, r.filter)),
        ("Green pixels (HSV range)", f"{100 * m.green_fraction:.1f}%"),
        ("Otsu threshold on chroma distance", f"{m.otsu_t} (used: {m.dist_t})"),
        ("Foreground kept", f"{100 * m.fg_fraction:.1f}%"),
        ("Background mask", "Applied" if m.applied else "Not applied (fallback: full image kept)"),
    ]
    if r.flagged and r.scores:
        for name_, score in sorted(r.scores.items(), key=lambda kv: kv[1]):
            results.append((f"  BRISQUE with {FILTER_LABELS.get(name_, name_).lower()}", f"{score:.2f}"))
    times = [(STAGE_LABELS[s], f"{res.times_ms[s]:.1f} ms") for s in pp.STAGES]
    times.append(("Total", f"{total_ms:.1f} ms"))
    return {
        "images": {"original": data_uri(res.original), "final": data_uri(res.final, 92),
                   "mask": data_uri(m.foreground), "enhanced": data_uri(res.enhanced)},
        "raw_width": res.raw_shape[1], "raw_height": res.raw_shape[0], "size": pp.TARGET_SIZE,
        "raw_kb": round(raw_kb, 1), "processed_kb": round(proc_kb, 1),
        "reduction_percent": round(100 * (1 - proc_kb / raw_kb), 1),
        "total_ms": round(total_ms),
        "brisque_before": round(r.brisque_before, 1), "brisque_after": round(r.brisque_after, 1),
        "low_quality": bool(r.flagged), "mask_applied": bool(m.applied),
        "foreground_percent": round(100 * m.fg_fraction, 1) if m.applied else 100.0,
        "results": results, "times": times,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/process":
            self._send(404, b"Not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            self._json(400, {"error": "No image was sent."})
            return
        if length > MAX_UPLOAD:
            self._json(413, {"error": f"The file is larger than {MAX_UPLOAD // (1024 * 1024)} MB."})
            return
        data = self.rfile.read(length)
        # Only the extension of the file name is used, and only if it is a known image type.
        suffix = Path(self.headers.get("X-Filename", "")).suffix.lower()
        if suffix not in pp.IMAGE_EXTENSIONS:
            suffix = ".jpg"
        try:
            out = process_upload(data, suffix)
        except Exception as exc:   # report the error on the page, keep the server running
            self._json(500, {"error": f"Processing failed: {exc}"})
            return
        if out is None:
            self._json(400, {"error": "The file is not an image that can be read."})
            return
        self._json(200, out)

    def log_message(self, fmt: str, *args) -> None:
        print("  " + fmt % args, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web page for the preprocessing pipeline")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Open http://127.0.0.1:{args.port} in a browser. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
