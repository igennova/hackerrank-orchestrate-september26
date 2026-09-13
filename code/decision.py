"""Deterministic decision engine for the "Buy or Wait?" agent.

No LLM. Consumes the projected forecast and the forecast primitive
(``reconstruct.build_forecast`` -> trough + suffix-min breach dates) to produce the
seven required output fields for one request.

Pipeline (problem_statement.md: Allowed values, Choosing Between Safe Plans,
partial_payment rules, spending_changes_needed):
  1. amount_safe_to_pay  = largest lump payable on request_date keeping the 90-day
                           trough >= minimum (capped at requested_amount).
  2. earliest_date_for_full_payment = first date a full lump stays safe; else "".
     (a pure capacity measure, before optional spending changes.)
  3. Enumerate ELIGIBLE plans (method must be in payment_methods_user_will_consider;
     wait needs full_payment accepted; installments must match a supplied option;
     partial only when allowed/accepted and 0 < safe < requested and the second
     payment lands by desired_completion_date).
  4. Spending-change rescue: stop/reduce only recurring FLEXIBLE expenses the user
     permits (respect min_allowed; <=3 changes; stop and reduce target different
     events); re-forecast to test safety. Used to enable full payment in time.
  5. Rank safe eligible plans by the spec's 6 levels and pick the winner.
  6. Read affordability_status off the winner.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import reconstruct as R

EPS = 1e-6


@dataclass
class Decision:
    request_id: str
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str


@dataclass
class _Plan:
    method: str  # full_payment | partial_payment | installments | wait
    payments: List[Tuple[date, float]]
    total_paid: float
    start_date: date
    changes: List[str] = field(default_factory=list)  # stop:/reduce_to: strings
    option_id: Optional[int] = None
    completes_by_dcd: bool = True

    @property
    def uses_changes(self) -> bool:
        return bool(self.changes)

    @property
    def num_payments(self) -> int:
        return len(self.payments)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _num(v: float) -> str:
    """Compact number: integer if whole, else up to 2 decimals (no trailing zeros)."""
    v = round(v + 0.0, 2)
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _money(v: float, cur: str) -> str:
    v = round(v + 0.0, 2)
    if v == int(v):
        return f"{cur} {int(v):,}"
    return f"{cur} {v:,.2f}"


def _long_date(d: date) -> str:
    return f"{d.day} {calendar.month_name[d.month]} {d.year}"


def _plan_str(payments: List[Tuple[date, float]]) -> str:
    if not payments:
        return "none"
    return "|".join(f"{d.isoformat()}:{_num(a)}" for d, a in payments)


def _split(field_value: str) -> set:
    return {c for c in (field_value or "").split("|") if c}


# ---------------------------------------------------------------------------
# Spending-change candidates
# ---------------------------------------------------------------------------
@dataclass
class _ChangeOption:
    kind: str  # "stop" | "reduce"
    event_id: str
    category: str
    scale: float  # multiplier for that category's projected expense (0 => stop)
    label: str  # for spending_changes_needed
    description: str


def _change_options(
    series: List[R.RecurringSeries], profile: Dict[str, str]
) -> List[_ChangeOption]:
    """Permitted stop/reduce actions on flexible recurring EXPENSE series."""
    protect = _split(profile["expense_categories_to_protect"])
    willing_reduce = _split(profile["expense_categories_user_is_willing_to_reduce"])
    willing_stop = _split(profile["expense_categories_user_is_willing_to_stop"])

    out: List[_ChangeOption] = []
    for s in series:
        if s.direction != "debit" or s.kind not in ("monthly", "daily_drip"):
            continue
        if s.category in protect:
            continue
        latest = max(s.cashflows, key=lambda c: c.hit_date)
        if latest.original_amount is None:
            continue
        if latest.can_stop and s.category in willing_stop:
            out.append(
                _ChangeOption(
                    "stop", latest.event_id, s.category, 0.0,
                    f"stop:{latest.event_id}", latest.description,
                )
            )
        if (
            latest.can_reduce
            and s.category in willing_reduce
            and latest.min_allowed_amount is not None
            and latest.min_allowed_amount < latest.original_amount
        ):
            new_amt = latest.min_allowed_amount
            scale = new_amt / latest.original_amount
            out.append(
                _ChangeOption(
                    "reduce", latest.event_id, s.category, scale,
                    f"reduce_to:{latest.event_id}:{_num(new_amt)}", latest.description,
                )
            )
    return out


def _find_rescue(
    user_id: str, rd: date, B: float, M: float, A: float,
    change_options: List[_ChangeOption],
) -> Optional[List[_ChangeOption]]:
    """Smallest set (<=3) of changes that makes a full lump today safe.

    stop and reduce may not target the same event. Prefer fewer changes, then the
    combination that frees the most headroom, then lowest event ids (determinism).
    """
    for size in (1, 2, 3):
        best: Optional[Tuple] = None
        for combo in combinations(change_options, size):
            events = [c.event_id for c in combo]
            if len(set(events)) != len(events):
                continue  # stop and reduce can't hit the same event
            scale: Dict[str, float] = {}
            for c in combo:
                # If two changes touch the same category, keep the deeper cut.
                scale[c.category] = min(scale.get(c.category, 1.0), c.scale)
            fc = R.build_forecast(
                user_id, rd, B, extra_payments=[(rd, A)], series_scale=scale
            )
            if fc.trough >= M - EPS:
                key = (
                    sum(1.0 - c.scale for c in combo),  # more savings first (desc)
                    tuple(sorted(events)),
                )
                # maximize savings -> minimize negative
                score = (-key[0], key[1])
                if best is None or score < best[0]:
                    best = (score, list(combo))
        if best is not None:
            return best[1]
    return None


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def decide(
    request: Dict[str, str],
    profile: Dict[str, str],
    payment_options: List[Dict[str, str]],
    income_policy: str = "A",
) -> Decision:
    user_id = request["user_id"]
    cur = profile["home_currency"]
    B = float(profile["current_available_balance"])
    M = float(profile["minimum_balance_to_keep"])
    A = float(request["requested_amount"])
    rd = date.fromisoformat(request["request_date"])
    dcd = date.fromisoformat(request["desired_completion_date"])
    methods = _split(profile["payment_methods_user_will_consider"])
    allows_partial = request["allows_partial_payment"].strip().lower() == "true"
    max_inst_months = profile["max_installment_months"].strip()

    base = R.build_forecast(user_id, rd, B, income_policy=income_policy)

    # 1. amount_safe_to_pay (before optional spending changes)
    safe = max(0.0, min(base.trough - M, A))
    safe = round(safe, 2)

    # 2. earliest_date_for_full_payment (capacity, before spending changes)
    earliest_full: Optional[date] = None
    for d in base.days:
        if base.suffix_min(d) >= M + A - EPS:
            earliest_full = d
            break

    # 3-4. Enumerate eligible, safe candidate plans.
    plans: List[_Plan] = []
    full_today_safe = earliest_full == rd

    # full_payment today, no changes
    if "full_payment" in methods and full_today_safe:
        plans.append(_Plan("full_payment", [(rd, A)], A, rd, completes_by_dcd=rd <= dcd))

    # full_payment today enabled by spending changes (only when not already safe
    # today, and only if it helps complete in time)
    if "full_payment" in methods and not full_today_safe:
        series = R.detect_recurrence(R.reconstruct(user_id), income_policy=income_policy)
        rescue = _find_rescue(user_id, rd, B, M, A, _change_options(series, profile))
        if rescue is not None:
            plans.append(
                _Plan(
                    "full_payment", [(rd, A)], A, rd,
                    changes=[c.label for c in rescue], completes_by_dcd=rd <= dcd,
                )
            )

    # partial_payment: pay safe today, remainder on earliest_full (by deadline)
    if (
        allows_partial
        and "partial_payment" in methods
        and 0 < safe < A
        and earliest_full is not None
        and earliest_full <= dcd
    ):
        remainder = round(A - safe, 2)
        fc = R.build_forecast(
            user_id, rd, B, income_policy=income_policy,
            extra_payments=[(rd, safe), (earliest_full, remainder)],
        )
        if fc.trough >= M - EPS:
            plans.append(
                _Plan(
                    "partial_payment", [(rd, safe), (earliest_full, remainder)], A, rd,
                    completes_by_dcd=True,
                )
            )

    # installments: must match a supplied option, respect max_installment_months
    if "installments" in methods and max_inst_months:
        max_months = int(max_inst_months)
        for opt in payment_options:
            if opt["payment_method"] != "installments" or not opt["first_payment_date"]:
                continue
            n = int(opt["number_of_payments"])
            freq = int(opt["payment_frequency_days"] or 0)
            fp = date.fromisoformat(opt["first_payment_date"])
            pay = float(opt["payment_amount"])
            last = fp + timedelta(days=(n - 1) * freq)
            months = round((last - fp).days / 30) + 1
            if months > max_months:
                continue
            schedule = [(fp + timedelta(days=k * freq), pay) for k in range(n)]
            fc = R.build_forecast(
                user_id, rd, B, income_policy=income_policy,
                extra_payments=schedule, horizon_days=max((last - rd).days, R.FORECAST_DAYS),
            )
            if fc.trough >= M - EPS:
                total = float(opt["total_payable_amount"])
                oid = int(opt["payment_option_id"].rsplit("_", 1)[-1])
                plans.append(
                    _Plan(
                        "installments", schedule, total, fp,
                        option_id=oid, completes_by_dcd=last <= dcd,
                    )
                )

    # wait: full becomes safe later, paid as a single full lump (needs full_payment)
    if "full_payment" in methods and earliest_full is not None and earliest_full > rd:
        plans.append(
            _Plan(
                "wait", [(earliest_full, A)], A, earliest_full,
                completes_by_dcd=earliest_full <= dcd,
            )
        )

    # 5. Rank and pick.
    earliest_str = earliest_full.isoformat() if earliest_full is not None else ""
    if not plans:
        return _not_recommended(request["request_id"], safe, earliest_str, cur, M, dcd)

    def rank_key(p: _Plan):
        return (
            0 if p.completes_by_dcd else 1,       # L1 complete by deadline
            1 if p.uses_changes else 0,           # L2 avoid spending changes
            round(p.total_paid, 2),               # L3 minimize total paid
            p.start_date.toordinal(),             # L4 start earlier
            p.num_payments,                       # L5 fewer payments
            p.option_id if p.option_id is not None else -1,  # L6 lowest option id
        )

    winner = min(plans, key=rank_key)

    # 6. Status from the winner.
    if winner.method == "full_payment" and not winner.uses_changes:
        status = "affordable_now"
    elif winner.method == "wait":
        status = "affordable_later"
    else:  # partial, installments, or full_payment with changes
        status = "affordable_with_plan"

    explanation = _explain(winner, status, cur, M, A, safe, dcd, profile)
    return Decision(
        request_id=request["request_id"],
        amount_safe_to_pay=safe,
        affordability_status=status,
        recommended_payment_method=winner.method,
        payment_plan=_plan_str(winner.payments),
        earliest_date_for_full_payment=earliest_str,
        spending_changes_needed="|".join(winner.changes) if winner.changes else "none",
        decision_explanation=explanation,
    )


def _not_recommended(rid, safe, earliest_str, cur, M, dcd) -> Decision:
    return Decision(
        request_id=rid,
        amount_safe_to_pay=safe,
        affordability_status="not_affordable",
        recommended_payment_method="not_recommended",
        payment_plan="none",
        earliest_date_for_full_payment=earliest_str,
        spending_changes_needed="none",
        decision_explanation=(
            f"Do not make this payment by {_long_date(dcd)}. "
            f"None of the available options keeps the {_money(M, cur)} minimum protected."
        ),
    )


def _explain(winner, status, cur, M, A, safe, dcd, profile) -> str:
    if winner.method == "full_payment" and not winner.uses_changes:
        return (
            f"Pay {_money(A, cur)} today. This leaves at least {_money(M, cur)} "
            "available over the next 90 days."
        )
    if winner.method == "full_payment" and winner.uses_changes:
        return (
            f"Apply spending changes ({', '.join(winner.changes)}), then pay "
            f"{_money(A, cur)} today. This keeps the {_money(M, cur)} minimum protected."
        )
    if winner.method == "partial_payment":
        d1, a1 = winner.payments[0]
        d2, a2 = winner.payments[1]
        return (
            f"Pay {_money(a1, cur)} today and the remaining {_money(a2, cur)} on "
            f"{_long_date(d2)}. This completes the full request and keeps the "
            f"{_money(M, cur)} minimum protected."
        )
    if winner.method == "installments":
        d1, a1 = winner.payments[0]
        return (
            f"Use {winner.num_payments} installments of {_money(a1, cur)}, starting "
            f"{_long_date(d1)}. This leaves at least {_money(M, cur)} available."
        )
    if winner.method == "wait":
        d1, a1 = winner.payments[0]
        return (
            f"Pay {_money(a1, cur)} in full on {_long_date(d1)}. Paying earlier would "
            f"take the balance below the {_money(M, cur)} minimum."
        )
    return ""
