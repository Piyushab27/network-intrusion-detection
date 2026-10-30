"""
Majority vote of three submissions (final selected submission, public 0.84614).

For each row, take the label that at least two of the three files agree on.
If all three disagree, fall back to the first file.

Usage:
  python src/majority_vote.py submission_pseudo_1round.csv submission_random_cv.csv \
      submission_top2_blend.csv --out submission_majority_vote.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs=3, type=Path)
    parser.add_argument("--out", type=Path, default=Path("submission_majority_vote.csv"))
    args = parser.parse_args()

    first, second, third = (pd.read_csv(f) for f in args.files)
    if not ((first.id.values == second.id.values).all() and (first.id.values == third.id.values).all()):
        raise ValueError("submission files are not in the same id order")

    agree_first = (first.Target == second.Target) | (first.Target == third.Target)
    vote = first.Target.where(agree_first, second.Target.where(second.Target == third.Target, first.Target))
    pd.DataFrame({"id": first.id, "Target": vote}).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
