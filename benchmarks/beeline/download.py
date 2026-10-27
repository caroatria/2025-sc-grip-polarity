#!/usr/bin/env python3
"""
Download the BEELINE benchmark data (Zenodo, concept DOI 10.5281/zenodo.3378975)
plus TRRUST v2 signed TF-target tables, and unpack them into ./data.

    python download_beeline.py            # everything
    python download_beeline.py --no-trrust
"""
import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ZENODO_CONCEPT = "3378975"
TRRUST = {
    "trrust_rawdata.human.tsv": "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv",
    "trrust_rawdata.mouse.tsv": "https://www.grnpedia.org/trrust/data/trrust_rawdata.mouse.tsv",
}


def fetch(url, dest: Path):
    if dest.exists():
        print(f"  have {dest.name}")
        return
    print(f"  downloading {dest.name} ...")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--no-trrust", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    # concept record id -> latest version (v4 is the one BEELINE v1.1 expects)
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{ZENODO_CONCEPT}") as r:
        rec = json.load(r)
    print(f"Zenodo: {rec['metadata']['title']} (version {rec['metadata'].get('version')})")
    for f in rec["files"]:
        dest = out / f["key"]
        fetch(f["links"]["self"], dest)
        if dest.suffix == ".zip":
            print(f"  unzipping {dest.name}")
            with zipfile.ZipFile(dest) as z:
                z.extractall(out)

    if not a.no_trrust:
        for name, url in TRRUST.items():
            fetch(url, out / name)
    print("done ->", out.resolve())


if __name__ == "__main__":
    main()
