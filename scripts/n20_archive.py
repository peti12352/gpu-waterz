#!/usr/bin/env python3
"""P0: curl-archive N20 papers. Stamp HTTP. Do not cite as read if not 200 PDF/HTML."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "papers/fetched"
LEDGER = ROOT / "papers/n20_http.json"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

URLS = [
    ("pandora_2401.06089.pdf", "https://arxiv.org/pdf/2401.06089"),
    ("chamfer_2602.10444.pdf", "https://arxiv.org/pdf/2602.10444"),
    ("centroid_2406.05066.pdf", "https://arxiv.org/pdf/2406.05066"),
    ("dyn_sld_2506.18384.pdf", "https://arxiv.org/pdf/2506.18384"),
    ("gshac_2604.11656.pdf", "https://arxiv.org/pdf/2604.11656"),
    ("pcbs_2411.10290.pdf", "https://arxiv.org/pdf/2411.10290"),
    ("parhac_2206.11654.pdf", "https://arxiv.org/pdf/2206.11654"),
    ("cuml_agg_2608.html",
     "https://docs.nvidia.com/cuml/26.08/api/generated/cuml.cluster.AgglomerativeClustering/"),
    ("forum_24259.json",
     "https://forums.developer.nvidia.com/t/kernel-produces-only-sometimes-right-results-hierarchical-clustering-bachelor-thesis/24259.json"),
    ("forum_8876.json",
     "https://forums.developer.nvidia.com/t/minimal-spanning-tree-on-cuda/8876.json"),
    ("forum_191439.json",
     "https://forums.developer.nvidia.com/t/gpu-accelerated-hierarchical-dbscan-with-rapids-cuml-let-s-get-back-to-the-future/191439.json"),
    ("gpu_upgma_2015.pdf",
     "https://www.cs.nthu.edu.tw/~ychung/Journal/2015-CCPE.pdf"),
]


def curl_one(name, url):
    dest = OUT / name
    cmd = [
        "curl", "-L", "-A", UA, "-o", str(dest), "-w", "%{http_code} %{size_download}",
        "--retry", "2", "--max-time", "90", url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    parts = (r.stdout or "").strip().split()
    code = int(parts[0]) if parts and parts[0].isdigit() else -1
    nbytes = dest.stat().st_size if dest.is_file() else 0
    # HTML cookie walls are not PDFs
    kind = "ok"
    if dest.is_file() and name.endswith(".pdf"):
        head = dest.read_bytes()[:8]
        if not head.startswith(b"%PDF"):
            kind = "not_pdf"
    return {"file": name, "url": url, "http": code, "bytes": nbytes, "kind": kind}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, url in URLS:
        print(f"fetch {name}", flush=True)
        try:
            rows.append(curl_one(name, url))
        except Exception as e:
            rows.append({"file": name, "url": url, "http": -1, "error": str(e)})
        time.sleep(0.4)
    LEDGER.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
