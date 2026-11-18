"""Convert the O'Neil supplementary Excel workbook to CSV before grid preparation.

Input:
    data/raw/oneil/oneil_sup.xlsx (O'Neil et al. 2016 supplementary workbook)
    Sheet name: "combination response"

Output:
    data/raw/oneil/oneil_raw.csv
"""
import argparse
import csv
import os
import time
from pathlib import Path

import openpyxl

ROOT = str(Path(__file__).resolve().parents[2])
DEFAULT_SRC = os.path.join(ROOT, "data", "raw", "oneil", "oneil_sup.xlsx")
DEFAULT_DST = os.path.join(ROOT, "data", "raw", "oneil", "oneil_raw.csv")
SHEET = "combination response"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--sheet", default=SHEET)
    args = ap.parse_args()

    if not os.path.exists(args.src):
        raise FileNotFoundError(f"Workbook not found: {args.src}")
    t0 = time.time()
    wb = openpyxl.load_workbook(args.src, read_only=True, data_only=True)
    if args.sheet not in wb.sheetnames:
        raise KeyError(f"Sheet {args.sheet!r} not found; available sheets: {wb.sheetnames}")
    ws = wb[args.sheet]
    log(f"sheet {args.sheet!r}: {ws.max_row} rows x {ws.max_column} columns")

    dst_dir = os.path.dirname(os.path.abspath(args.dst))
    os.makedirs(dst_dir, exist_ok=True)
    with open(args.dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        n = 0
        for row in ws.iter_rows(values_only=True):
            w.writerow(["" if v is None else v for v in row])
            n += 1
            if n % 100000 == 0:
                log(f"  {n:,} rows ({time.time() - t0:.0f}s)")
    wb.close()
    log(f"Done: {n:,} rows -> {args.dst} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
