"""Pipeline runner for the "Buy or Wait?" agent.

Runs loader -> reconstruct -> decision over every row in dataset/requests.csv and
writes output.csv to the repo root with the exact required columns, in order, one
row per request_id.

Deterministic, no LLM. Blank-amount events remain unresolved for now (current
reconstruct behavior); image resolution is a later step.

Usage:
    python3 code/main.py
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, List

import decision
from loader import load_dataset

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CODE_DIR)
OUTPUT_PATH = os.path.join(_REPO_ROOT, "output.csv")

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def run(output_path: str = OUTPUT_PATH, income_policy: str = "A") -> str:
    ds = load_dataset(verbose=False)
    profiles: Dict[str, Dict[str, str]] = {
        p["user_id"]: p for p in ds.financial_profiles
    }
    options: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for opt in ds.request_payment_options:
        options[opt["request_id"]].append(opt)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(OUTPUT_COLUMNS)
        for req in ds.requests:
            d = decision.decide(
                req,
                profiles[req["user_id"]],
                options.get(req["request_id"], []),
                income_policy=income_policy,
            )
            writer.writerow(
                [
                    d.request_id,
                    decision._num(d.amount_safe_to_pay),
                    d.affordability_status,
                    d.recommended_payment_method,
                    d.payment_plan,
                    d.earliest_date_for_full_payment,
                    d.spending_changes_needed,
                    d.decision_explanation,
                ]
            )
    print(f"Wrote {len(ds.requests)} predictions to {output_path}")
    return output_path


if __name__ == "__main__":
    run()
