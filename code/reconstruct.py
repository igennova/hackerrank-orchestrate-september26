"""Financial-state reconstruction (slice 1 of 2) for the "Buy or Wait?" agent.

This slice does THREE jobs only and nothing else:

  1. NORMALIZE each financial event into a signed, home-currency ``Cashflow``.
  2. FILTER out events the spec says to ignore (failed, cancelled, pending
     credits, and unrealized / non-cash investment values), recording a reason.
  3. DEDUPLICATE repeated representations of the same underlying event that are
     linked via ``linked_event_id``, keeping the most authoritative per the
     spec's conflict order, and recording what was merged.

There is deliberately NO recurring detection, NO forward projection, and NO
decision logic here -- those are the second slice. Pure and deterministic;
nothing on disk is mutated. Joins come from ``loader`` and rate lookups from
``currency`` -- neither is reimplemented here.

Spec references (problem_statement.md):
  - "Reserve pending debits. Do not count pending credits ... refunds ...
     or investment gains until they settle."
  - "Ignore pending credits, failed or cancelled transactions, duplicate
     records, and unrealized investments."
  - "do not treat unrealized investment value as available cash."
  - Conflict order: explicit cancellation/settlement/amendment; then newer
     record from the same source; then a settled event over an estimate/
     forecast; then the financially safer interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import currency
from loader import load_dataset

# Statuses/dirs the spec tells us to ignore, mapped to a short drop reason.
# (Pending credits are handled separately because they depend on direction.)
_DROP_STATUS_REASON = {
    "failed": "failed",
    "cancelled": "cancelled",
    "unrealized": "unrealized_investment",
}

# Authority ranking for choosing the survivor among duplicate representations.
# Higher wins: a settled record beats a scheduled one beats a pending one.
_STATUS_AUTHORITY = {"settled": 3, "scheduled": 2, "pending": 1}


@dataclass
class Cashflow:
    """One normalized, signed, home-currency cash movement."""

    event_id: str
    hit_date: date
    signed_amount: Optional[float]  # None when the amount must come from an image
    home_currency: str
    category: str
    is_essential: bool
    flexibility: str  # raw value: fixed | reducible | stoppable | reducible_or_stoppable
    can_reduce: bool
    can_stop: bool
    min_allowed_amount: Optional[float]
    status: str
    event_type: str
    direction: str
    linked_event_id: str
    needs_image_amount: bool
    # Provenance for auditing / explanations (not used for decisions here).
    original_amount: Optional[float]
    original_currency: str
    conversion_rate_date: Optional[str]


@dataclass
class ReconstructionResult:
    user_id: str
    home_currency: str
    total_events_in: int
    cashflows: List[Cashflow]
    dropped: List[Tuple[str, str]] = field(default_factory=list)  # (event_id, reason)
    merged: List[Tuple[str, str, str]] = field(  # (kept_id, dropped_id, reason)
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_date(value: str) -> Optional[date]:
    value = (value or "").strip()
    if not value:
        return None
    return date.fromisoformat(value)


def _parse_amount(value: str) -> Optional[float]:
    value = (value or "").strip()
    if value == "":
        return None
    return float(value)


def _flex_flags(flexibility: str) -> Tuple[bool, bool]:
    """(can_reduce, can_stop) derived from the raw flexibility value."""
    can_reduce = flexibility in ("reducible", "reducible_or_stoppable")
    can_stop = flexibility in ("stoppable", "reducible_or_stoppable")
    return can_reduce, can_stop


def _drop_reason(ev: Dict[str, str]) -> Optional[str]:
    """Return a drop reason if this event is on the spec's ignore list, else None."""
    status = ev["status"]
    if status in _DROP_STATUS_REASON:
        return _DROP_STATUS_REASON[status]
    if ev["direction"] == "non_cash":
        return "non_cash_valuation"
    if status == "pending" and ev["direction"] == "credit":
        return "pending_credit"
    return None


def _normalize(ev: Dict[str, str], home_currency: str, protect: set) -> Cashflow:
    event_type = ev["event_type"]
    direction = ev["direction"]

    # hit_date: settlement_date for income, event_date otherwise (fall back if blank).
    settle = _parse_date(ev["settlement_date"])
    occur = _parse_date(ev["event_date"])
    if event_type == "income":
        hit_date = settle or occur
    else:
        hit_date = occur or settle

    original_currency = ev["currency"]
    original_amount = _parse_amount(ev["amount"])
    needs_image_amount = original_amount is None

    signed_amount: Optional[float] = None
    conversion_rate_date: Optional[str] = None
    if not needs_image_amount:
        # Convert foreign cash events on their settlement date (spec), else event_date.
        conv_date = ev["settlement_date"] or ev["event_date"]
        if original_currency != home_currency:
            rate, rate_date = currency.lookup_rate(
                original_currency, home_currency, conv_date
            )
            magnitude = original_amount * rate
            conversion_rate_date = rate_date.isoformat()
        else:
            magnitude = original_amount
        # debit/expense -> negative; credit/income -> positive.
        signed_amount = -magnitude if direction == "debit" else magnitude

    can_reduce, can_stop = _flex_flags(ev["flexibility"])
    return Cashflow(
        event_id=ev["event_id"],
        hit_date=hit_date,
        signed_amount=signed_amount,
        home_currency=home_currency,
        category=ev["category"],
        is_essential=ev["category"] in protect,
        flexibility=ev["flexibility"],
        can_reduce=can_reduce,
        can_stop=can_stop,
        min_allowed_amount=_parse_amount(ev["minimum_allowed_amount"]),
        status=ev["status"],
        event_type=event_type,
        direction=direction,
        linked_event_id=ev["linked_event_id"],
        needs_image_amount=needs_image_amount,
        original_amount=original_amount,
        original_currency=original_currency,
        conversion_rate_date=conversion_rate_date,
    )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def _authority(cf: Cashflow) -> Tuple:
    """Sort key for picking the survivor among duplicate representations.

    Conflict order, strongest first: a settled record over an estimate/forecast,
    then the newer record, then a stable id tie-break. Higher tuple wins.
    """
    status_rank = _STATUS_AUTHORITY.get(cf.status, 0)
    ordinal = cf.hit_date.toordinal() if cf.hit_date else 0
    # Numeric event_id tail keeps ordering deterministic and human-sensible.
    try:
        id_num = int(cf.event_id.rsplit("_", 1)[-1])
    except ValueError:
        id_num = 0
    return (status_rank, ordinal, id_num)


def _dup_key(cf: Cashflow) -> Optional[Tuple]:
    """Duplicate-representation key: same direction, amount, and currency.

    A refund (opposite direction) or a sale (different amount) is NOT a duplicate
    of its linked purchase, so those keep both records. Blank-amount events are
    never merged -- we cannot confirm equality without resolving the image.
    """
    if cf.original_amount is None:
        return None
    return (cf.direction, round(abs(cf.original_amount), 2), cf.original_currency)


def _dedupe(
    survivors: List[Cashflow],
) -> Tuple[List[Cashflow], List[Tuple[str, str, str]]]:
    """Merge duplicate representations connected through ``linked_event_id``.

    Only events that are linked to each other AND share a duplicate key are
    merged; the most authoritative one is kept.
    """
    by_id = {cf.event_id: cf for cf in survivors}
    survivor_ids = set(by_id)

    # Union-find over linked_event_id edges where BOTH endpoints survived.
    parent: Dict[str, str] = {cid: cid for cid in survivor_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for cf in survivors:
        link = cf.linked_event_id
        if link and link in survivor_ids:
            union(cf.event_id, link)

    # Within each connected component, bucket by duplicate key.
    components: Dict[str, List[Cashflow]] = {}
    for cf in survivors:
        components.setdefault(find(cf.event_id), []).append(cf)

    kept: List[Cashflow] = []
    merged: List[Tuple[str, str, str]] = []
    for members in components.values():
        buckets: Dict[Tuple, List[Cashflow]] = {}
        singles: List[Cashflow] = []  # blank-amount (unmergeable) go straight through
        for cf in members:
            key = _dup_key(cf)
            if key is None:
                singles.append(cf)
            else:
                buckets.setdefault(key, []).append(cf)
        kept.extend(singles)
        for group in buckets.values():
            if len(group) == 1:
                kept.append(group[0])
                continue
            group.sort(key=_authority, reverse=True)
            winner = group[0]
            kept.append(winner)
            for loser in group[1:]:
                merged.append(
                    (
                        winner.event_id,
                        loser.event_id,
                        f"duplicate of {winner.event_id} "
                        f"({loser.status} representation dropped, kept {winner.status})",
                    )
                )
    kept.sort(key=lambda cf: (cf.hit_date or date.min, cf.event_id))
    return kept, merged


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def reconstruct_detailed(user_id: str) -> ReconstructionResult:
    ds = load_dataset(verbose=False)
    profile = ds.get_profile(user_id)
    if profile is None:
        raise KeyError(f"no financial_profile for user {user_id!r}")
    home_currency = profile["home_currency"]
    protect = {
        c for c in (profile["expense_categories_to_protect"] or "").split("|") if c
    }

    events = ds.get_user_events(user_id)

    dropped: List[Tuple[str, str]] = []
    survivors: List[Cashflow] = []
    for ev in events:
        reason = _drop_reason(ev)
        if reason is not None:
            dropped.append((ev["event_id"], reason))
            continue
        survivors.append(_normalize(ev, home_currency, protect))

    cashflows, merged = _dedupe(survivors)

    return ReconstructionResult(
        user_id=user_id,
        home_currency=home_currency,
        total_events_in=len(events),
        cashflows=cashflows,
        dropped=dropped,
        merged=merged,
    )


def reconstruct(user_id: str) -> List[Cashflow]:
    """Clean list of normalized Cashflow records for a user, sorted by hit_date."""
    return reconstruct_detailed(user_id).cashflows


# ---------------------------------------------------------------------------
# Demo / manual review
# ---------------------------------------------------------------------------
def _print_summary(result: ReconstructionResult, full: bool = False) -> None:
    print("=" * 78)
    print(f"USER {result.user_id}  (home currency: {result.home_currency})")
    print("=" * 78)
    print(f"  events in            : {result.total_events_in}")
    print(f"  dropped (ignored)    : {len(result.dropped)}")
    for eid, reason in result.dropped:
        print(f"      - {eid}: {reason}")
    print(f"  merged (deduplicated): {len(result.merged)}")
    for kept, loser, reason in result.merged:
        print(f"      - dropped {loser} -> {reason}")
    print(f"  clean cashflows out  : {len(result.cashflows)}")

    if full:
        print("\n  Clean Cashflow list (sorted by hit_date):")
        print(
            f"    {'hit_date':<12} {'event_id':<12} {'signed_amt':>14} "
            f"{'category':<16} {'ess':<4} {'flex':<22} {'status':<9} {'type'}"
        )
        for cf in result.cashflows:
            amt = "   (needs image)" if cf.signed_amount is None else f"{cf.signed_amount:>14.2f}"
            print(
                f"    {cf.hit_date.isoformat():<12} {cf.event_id:<12} {amt:>14} "
                f"{cf.category:<16} {str(cf.is_essential):<4} {cf.flexibility:<22} "
                f"{cf.status:<9} {cf.event_type}"
            )


if __name__ == "__main__":
    # Required: user_09 (fully clean) printed in full.
    _print_summary(reconstruct_detailed("user_09"), full=True)

    # Extra: prove the drop + merge paths actually fire.
    print("\n\n### Extra users to exercise filter + dedup paths ###\n")
    print(">>> user_149 exercises FILTER drops (pending credit + cancelled):")
    _print_summary(reconstruct_detailed("user_149"), full=False)
    print("\n>>> user_138 exercises DEDUP (linked settled+pending duplicate):")
    _print_summary(reconstruct_detailed("user_138"), full=False)
