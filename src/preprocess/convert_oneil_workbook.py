# -*- coding: utf-8 -*-
"""
O'Neil 补充材料 xlsx → CSV（数据处理链条的第一步）
====================================================

输入（AACR 期刊补充材料，扩展名 .xls 但实为 xlsx）：
    raw/oneil/oneil_sup.xlsx
    来源：O'Neil et al. 2016, Mol Cancer Ther 15(6):1155-62 的补充材料
    Sheet 名："combination response"

输出：
    raw/oneil/oneil_raw.csv     供 prep_oneil.py 读取

为什么单独一步：
    xlsx 单 sheet 的 XML 有 229 MB，openpyxl 逐行迭代约 47 秒。
    转成 CSV 后 prep_oneil.py 用 pandas 读取只需 ~1 秒，且便于用文本工具检查。

用法：
    python prep_oneil_xlsx.py                       # 用默认路径
    python prep_oneil_xlsx.py --src <xlsx> --dst <csv>
"""
import argparse
import csv
import os
import time

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_SRC = os.path.join(ROOT, "data", "raw", "oneil", "oneil_sup.xlsx")
DEFAULT_DST = os.path.join(ROOT, "data", "raw", "oneil", "oneil_raw.csv")
SHEET = "combination response"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--sheet", default=SHEET)
    args = ap.parse_args()

    if not os.path.exists(args.src):
        raise FileNotFoundError(f"找不到 {args.src}")
    t0 = time.time()
    wb = openpyxl.load_workbook(args.src, read_only=True, data_only=True)
    if args.sheet not in wb.sheetnames:
        raise KeyError(f"找不到 sheet {args.sheet!r}；实际有 {wb.sheetnames}")
    ws = wb[args.sheet]
    log(f"sheet {args.sheet!r}: {ws.max_row} 行 × {ws.max_column} 列")

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    with open(args.dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        n = 0
        for row in ws.iter_rows(values_only=True):
            w.writerow(["" if v is None else v for v in row])
            n += 1
            if n % 100000 == 0:
                log(f"  {n:,} 行  ({time.time()-t0:.0f}s)")
    wb.close()
    log(f"完成: {n:,} 行 → {args.dst}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
