"""Merge tester workbooks into one consolidated result list (MSP-45).

Usage (from the repo root):
  python docs/test-cases/merge_results.py out.xlsx file1.xlsx file2.xlsx file3.xlsx

Each input needs a sheet "Test Cases" with the workbook's columns. For every TC ID the row with a
Result other than "Not run" wins; if two files both recorded a result, the script stops and names the case.
Pending Cn cases are counted as Pending and never as passed.
"""
import sys

import pandas as pd

RESULTS = ["Pass", "Fail", "Blocked", "Pending", "Not run"]


def load(path):
    frame = pd.read_excel(path, sheet_name="Test Cases")
    frame["Result"] = frame["Result"].fillna("Not run")
    return frame


def merge(paths):
    merged = {}
    for path in paths:
        for _, row in load(path).iterrows():
            key = row["TC ID"]
            old = merged.get(key)
            if old is None or old["Result"] == "Not run":
                merged[key] = row
            elif row["Result"] != "Not run" and row["Result"] != "Pending":
                raise SystemExit(f"{key} has a result in two files: {old['Result']} and {row['Result']}")
    return pd.DataFrame(list(merged.values())).sort_values("TC ID")


def summary(frame):
    by_feature = frame.groupby(["Feature", "Result"]).size().unstack(fill_value=0)
    for name in RESULTS:
        if name not in by_feature:
            by_feature[name] = 0
    by_feature = by_feature[RESULTS]
    by_feature["Total"] = by_feature.sum(axis=1)
    high = frame[(frame["Risk"] == "H") & (frame["Result"] == "Not run")]
    return by_feature, high


def main(out, paths):
    frame = merge(paths)
    by_feature, high = summary(frame)
    print(by_feature.to_string())
    print("\nTotals:", frame["Result"].value_counts().to_dict(), "of", len(frame), "cases")
    print("High-risk cases still Not run:", len(high))
    fails = frame[frame["Result"] == "Fail"][["TC ID", "Feature", "Risk", "Defect ID"]]
    print("\nFailures:\n" + (fails.to_string(index=False) if len(fails) else "none"))
    with pd.ExcelWriter(out) as writer:
        frame.to_excel(writer, sheet_name="Test Cases", index=False)
        by_feature.to_excel(writer, sheet_name="By feature")
        high.to_excel(writer, sheet_name="High not run", index=False)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
