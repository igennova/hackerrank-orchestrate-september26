"""Validator + sample scorer for the "Buy or Wait?" agent.

Two modes:
  * validate <output.csv>  -- structural/constraint checks against dataset/requests.csv
  * sample                 -- run the decision engine on sample_requests.csv and print
                              a per-field match table vs the completed sample outputs

Usage:
    python3 code/validate.py            # validate repo-root output.csv + sample scoring
    python3 code/validate.py <path>     # validate a specific output.csv
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from datetime import date
from typing import Dict, List, Tuple

import decision
from loader import load_dataset

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CODE_DIR)
DEFAULT_OUTPUT = os.path.join(_REPO_ROOT, "output.csv")

REQUIRED_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]
STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
EPS = 1e-4


def _num(s: str) -> float:
    return float(s)


def _parse_plan(plan: str) -> List[Tuple[str, float]]:
    if plan.strip() in ("none", ""):
        return []
    out = []
    for part in plan.split("|"):
        d, a = part.rsplit(":", 1)
        out.append((d, float(a)))
    return out


def validate(output_path: str) -> bool:
    ds = load_dataset(verbose=False)
    requests = {r["request_id"]: r for r in ds.requests}
    # flexible recurring event ids per user (for spending-change checks)
    flex_events: Dict[str, Dict[str, str]] = {}
    for ev in ds.financial_events:
        if ev["flexibility"] in ("reducible", "stoppable", "reducible_or_stoppable"):
            flex_events[ev["event_id"]] = ev
    # installment option ids per request
    opt_ids: Dict[str, set] = defaultdict(set)
    inst_options: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for o in ds.request_payment_options:
        opt_ids[o["request_id"]].add(o["payment_option_id"])
        if o["payment_method"] == "installments":
            inst_options[o["request_id"]][o["payment_option_id"]] = o

    with open(output_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        rows = list(reader)

    problems: List[str] = []

    # column names + order
    if header != REQUIRED_COLUMNS:
        problems.append(f"HEADER mismatch: {header}")

    # row count + coverage
    if len(rows) != len(requests):
        problems.append(f"ROW COUNT {len(rows)} != {len(requests)} requests")
    out_ids = [r["request_id"] for r in rows]
    if len(set(out_ids)) != len(out_ids):
        problems.append("DUPLICATE request_id rows present")
    missing = set(requests) - set(out_ids)
    extra = set(out_ids) - set(requests)
    if missing:
        problems.append(f"MISSING {len(missing)} request_ids, e.g. {sorted(missing)[:5]}")
    if extra:
        problems.append(f"UNKNOWN {len(extra)} request_ids, e.g. {sorted(extra)[:5]}")

    for r in rows:
        rid = r["request_id"]
        req = requests.get(rid)
        if req is None:
            continue
        A = float(req["requested_amount"])

        if r["affordability_status"] not in STATUSES:
            problems.append(f"{rid}: bad affordability_status {r['affordability_status']!r}")
        if r["recommended_payment_method"] not in METHODS:
            problems.append(f"{rid}: bad recommended_payment_method {r['recommended_payment_method']!r}")

        # amount bounds
        try:
            safe = _num(r["amount_safe_to_pay"])
            if not (-EPS <= safe <= A + EPS):
                problems.append(f"{rid}: amount_safe_to_pay {safe} outside [0, {A}]")
        except ValueError:
            problems.append(f"{rid}: amount_safe_to_pay not numeric {r['amount_safe_to_pay']!r}")

        method = r["recommended_payment_method"]
        plan = _parse_plan(r["payment_plan"])

        # affordable_now => earliest == request_date
        if r["affordability_status"] == "affordable_now" and r["earliest_date_for_full_payment"] != req["request_date"]:
            problems.append(f"{rid}: affordable_now but earliest != request_date")

        # installments must match a supplied option's schedule
        if method == "installments":
            matched = False
            for oid, o in inst_options.get(rid, {}).items():
                n = int(o["number_of_payments"])
                if len(plan) != n:
                    continue
                pay = round(float(o["payment_amount"]), 2)
                if all(abs(a - pay) < 0.01 for _, a in plan):
                    matched = True
                    break
            if not matched:
                problems.append(f"{rid}: installments plan matches no supplied option")

        # partial => exactly two payments summing to requested
        if method == "partial_payment":
            if len(plan) != 2:
                problems.append(f"{rid}: partial_payment must have exactly 2 payments")
            else:
                total = sum(a for _, a in plan)
                if abs(total - A) > 0.01:
                    problems.append(f"{rid}: partial payments sum {total} != requested {A}")

        # spending changes: flexible recurring events, <=3, stop/reduce distinct events
        changes = r["spending_changes_needed"].strip()
        if changes not in ("none", ""):
            parts = changes.split("|")
            if len(parts) > 3:
                problems.append(f"{rid}: >3 spending changes")
            stop_ids, reduce_ids = set(), set()
            for p in parts:
                seg = p.split(":")
                if seg[0] == "stop" and len(seg) == 2:
                    eid = seg[1]
                    stop_ids.add(eid)
                elif seg[0] == "reduce_to" and len(seg) == 3:
                    eid = seg[1]
                    reduce_ids.add(eid)
                    ev = flex_events.get(eid)
                    if ev and ev["minimum_allowed_amount"]:
                        if float(seg[2]) < float(ev["minimum_allowed_amount"]) - EPS:
                            problems.append(f"{rid}: reduce_to below minimum_allowed for {eid}")
                else:
                    problems.append(f"{rid}: malformed spending change {p!r}")
                    continue
                if eid not in flex_events:
                    problems.append(f"{rid}: spending change targets non-flexible event {eid}")
            if stop_ids & reduce_ids:
                problems.append(f"{rid}: stop and reduce target same event {stop_ids & reduce_ids}")

    print("=" * 70)
    print(f"VALIDATE {output_path}")
    print("=" * 70)
    if problems:
        print(f"FAIL -- {len(problems)} problem(s):")
        for p in problems[:60]:
            print(f"  - {p}")
        if len(problems) > 60:
            print(f"  ... and {len(problems) - 60} more")
    else:
        print(f"PASS -- {len(rows)} rows, all constraints satisfied.")
    return not problems


def sample_scoring() -> None:
    ds = load_dataset(verbose=False)
    profiles = {p["user_id"]: p for p in ds.financial_profiles}
    options = defaultdict(list)
    for o in ds.request_payment_options:
        options[o["request_id"]].append(o)
    samples = ds.sample_requests

    fields = [
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
    ]
    counts = {f: 0 for f in fields}
    print("\n" + "=" * 70)
    print("SAMPLE SCORING (decision engine vs sample_requests.csv)")
    print("=" * 70)
    print(f"{'request':<11}{'status':<7}{'method':<8}{'safe':<7}{'earliest':<10}{'plan':<6}{'changes'}")
    for r in samples:
        d = decision.decide(r, profiles[r["user_id"]], options[r["request_id"]])
        got = {
            "amount_safe_to_pay": decision._num(d.amount_safe_to_pay),
            "affordability_status": d.affordability_status,
            "recommended_payment_method": d.recommended_payment_method,
            "payment_plan": d.payment_plan,
            "earliest_date_for_full_payment": d.earliest_date_for_full_payment,
            "spending_changes_needed": d.spending_changes_needed,
        }
        marks = {}
        for f in fields:
            truth = r[f].strip()
            pred = str(got[f]).strip()
            if f == "amount_safe_to_pay":
                ok = abs(float(truth) - float(pred)) < 0.01
            elif f in ("payment_plan", "spending_changes_needed"):
                ok = _norm_multi(truth) == _norm_multi(pred)
            else:
                ok = truth == pred
            counts[f] += ok
            marks[f] = "Y" if ok else "."
        print(
            f"{r['request_id']:<11}{marks['affordability_status']:<7}"
            f"{marks['recommended_payment_method']:<8}{marks['amount_safe_to_pay']:<7}"
            f"{marks['earliest_date_for_full_payment']:<10}{marks['payment_plan']:<6}"
            f"{marks['spending_changes_needed']}"
        )
    n = len(samples)
    print("-" * 70)
    for f in fields:
        print(f"  {f:<34} {counts[f]}/{n}")


def _norm_multi(s: str) -> set:
    """Compare plan/changes as a set of numerically-normalized tokens."""
    if s.strip() in ("none", ""):
        return set()
    toks = set()
    for part in s.split("|"):
        segs = part.split(":")
        norm = []
        for seg in segs:
            try:
                norm.append(f"{float(seg):.2f}")
            except ValueError:
                norm.append(seg)
        toks.add(":".join(norm))
    return toks


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT
    ok = validate(out)
    sample_scoring()
    sys.exit(0 if ok else 1)
