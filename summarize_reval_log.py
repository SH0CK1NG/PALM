#!/usr/bin/env python3
import argparse
import csv
import os
import re
from typing import List, Dict


def parse_args():
    p = argparse.ArgumentParser(description="Summarize re-eval fullgrid log to CSV")
    p.add_argument("--log", required=True, help="Path to re-eval log file")
    p.add_argument("--out", default="", help="Output CSV path; default evaluation_results/<logname>_summary.csv")
    return p.parse_args()


def extract_records(lines: List[str]) -> List[Dict]:
    records: List[Dict] = []
    cur_method = None
    cur_avg_fpr = None
    cur_avg_auroc = None
    cur_retain_acc = None
    cur_forget_fpr = None
    cur_forget_auroc = None
    cur_forget_auin = None

    method_re = re.compile(r"method=([^\n]+)")
    avg_re = re.compile(r"AVG\s+([0-9.]+)\s+([0-9.]+)")
    retain_re = re.compile(r"Retain-Acc:\s*([0-9.]+)")
    forget_block_re = re.compile(r"FPR:\s*([0-9.]+)\s+AUROC:\s*([0-9.]+)\s+AUIN:\s*([0-9.]+)")

    for line in lines:
        m = method_re.search(line)
        if m:
            # flush previous
            if cur_method is not None and cur_avg_fpr is not None and cur_avg_auroc is not None:
                records.append({
                    "MethodTag": cur_method.strip(),
                    "AVG-FPR": float(cur_avg_fpr),
                    "AVG-AUROC": float(cur_avg_auroc),
                    "Retain-Acc": float(cur_retain_acc) * 100.0 if cur_retain_acc is not None else 0.0,
                    "Forget-FPR": float(cur_forget_fpr) if cur_forget_fpr is not None else None,
                    "Forget-AUROC": float(cur_forget_auroc) if cur_forget_auroc is not None else None,
                    "Forget-AUIN": float(cur_forget_auin) if cur_forget_auin is not None else None,
                })
            cur_method = m.group(1)
            cur_avg_fpr = None
            cur_avg_auroc = None
            cur_retain_acc = None
            cur_forget_fpr = None
            cur_forget_auroc = None
            cur_forget_auin = None
            continue

        m = avg_re.search(line)
        if m:
            cur_avg_fpr = m.group(1)
            cur_avg_auroc = m.group(2)
            continue

        m = retain_re.search(line)
        if m:
            cur_retain_acc = m.group(1)
            continue

        m = forget_block_re.search(line)
        if m:
            cur_forget_fpr = m.group(1)
            cur_forget_auroc = m.group(2)
            cur_forget_auin = m.group(3)
            continue

    # flush last
    if cur_method is not None and cur_avg_fpr is not None and cur_avg_auroc is not None:
        records.append({
            "MethodTag": cur_method.strip(),
            "AVG-FPR": float(cur_avg_fpr),
            "AVG-AUROC": float(cur_avg_auroc),
            "Retain-Acc": float(cur_retain_acc) * 100.0 if cur_retain_acc is not None else 0.0,
            "Forget-FPR": float(cur_forget_fpr) if cur_forget_fpr is not None else None,
            "Forget-AUROC": float(cur_forget_auroc) if cur_forget_auroc is not None else None,
            "Forget-AUIN": float(cur_forget_auin) if cur_forget_auin is not None else None,
        })

    # enrich parsed hparams from method tag
    for r in records:
        tag = r["MethodTag"]
        ep = re.search(r"-e(\d+)-", tag)
        lr = re.search(r"-lr([0-9.]+)-", tag)
        fl = re.search(r"-fl([0-9.]+)-", tag)
        r["Epochs"] = int(ep.group(1)) if ep else None
        r["LR"] = float(lr.group(1)) if lr else None
        r["ForgetLambda"] = float(fl.group(1)) if fl else None
        # score: higher is better
        ra = r.get("Retain-Acc", 0.0)
        ffpr = r.get("Forget-FPR", 0.0) if r.get("Forget-FPR", None) is not None else 0.0
        # New score: AVG-AUROC - 0.5*AVG-FPR - 0.5*Forget-FPR + 0.2*Retain-Acc
        r["Score"] = r["AVG-AUROC"] - 0.5 * r["AVG-FPR"] - 0.5 * ffpr + 0.2 * ra

    # sort by score desc, then by AUROC desc, FPR asc
    records.sort(key=lambda x: (x.get("Score", -1e9), x.get("AVG-AUROC", -1e9), -x.get("AVG-FPR", 1e9)), reverse=True)
    return records


def write_csv(path: str, records: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "MethodTag", "Epochs", "LR", "ForgetLambda",
        "AVG-FPR", "AVG-AUROC",
        "Forget-FPR", "Forget-AUROC", "Forget-AUIN",
        "Retain-Acc", "Score"
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fields}
            w.writerow(row)


def main():
    args = parse_args()
    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    recs = extract_records(lines)
    base = os.path.splitext(os.path.basename(args.log))[0]
    out_path = args.out or os.path.join("evaluation_results", f"{base}_summary.csv")
    write_csv(out_path, recs)
    print(out_path)


if __name__ == "__main__":
    main()


