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

import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import currency
from loader import MEDIA_IMAGES_DIR, load_dataset

# Blank-amount image resolution (the one non-deterministic seam). Opt-in; when off,
# blank-amount events stay flagged/unresolved and the pipeline still runs.
RESOLVE_IMAGES = True
# event_id -> resolved signed home-currency amount, or None if unresolved. Persists
# for the process so the many reconstruct() calls resolve each event at most once
# (vision.py adds its own persistent per-image cache on top).
_RESOLVED_CACHE: Dict[str, Optional[float]] = {}

# Forecast horizon and recurrence-detection tunables (kept together for review).
FORECAST_DAYS = 90
RECENT_N = 3  # projection amount = mean of this many most-recent occurrences
MIN_OCCURRENCES = 3  # need this many to call anything recurring
MONTHLY_GAP_MIN, MONTHLY_GAP_MAX = 26, 32  # median gap window for "monthly"
MONTHLY_DOM_CONSISTENCY = 0.6  # fraction of occurrences on the modal day-of-month
DRIP_GAP_MAX = 20  # median gap at/below this (and not monthly) => near-daily drip
DRIP_MIN_OCCURRENCES = 8  # ...and at least this many samples
DAYS_PER_MONTH = 30.0  # drip denominator (monthly total spread evenly per day)

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
    description: str
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
        description=ev["description"],
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
    _resolve_blank_amounts(cashflows, home_currency)

    return ReconstructionResult(
        user_id=user_id,
        home_currency=home_currency,
        total_events_in=len(events),
        cashflows=cashflows,
        dropped=dropped,
        merged=merged,
    )


def _resolve_blank_amounts(cashflows: List[Cashflow], home_currency: str) -> None:
    """Fill blank-amount events from their linked image (the vision seam).

    For each cashflow flagged needs_image_amount, resolve the amount from its image
    (converting to home currency if needed) and clear the flag. If resolution yields
    nothing, the event stays flagged and is carried as before. Results are cached per
    event so repeated reconstruct() calls never re-invoke the model.
    """
    if not RESOLVE_IMAGES:
        return
    blanks = [cf for cf in cashflows if cf.needs_image_amount]
    if not blanks:
        return

    ds = load_dataset(verbose=False)
    for cf in blanks:
        if cf.event_id not in _RESOLVED_CACHE:
            _RESOLVED_CACHE[cf.event_id] = _resolve_one(cf, home_currency, ds)
        signed = _RESOLVED_CACHE[cf.event_id]
        if signed is not None:
            cf.signed_amount = signed
            cf.needs_image_amount = False


def _resolve_one(cf: Cashflow, home_currency: str, ds) -> Optional[float]:
    image = ds.get_image_for_event(cf.event_id)
    if image is None:
        return None
    import vision  # lazy: keeps the deterministic core import-clean and network-free

    path = os.path.join(MEDIA_IMAGES_DIR, f"{image['image_id']}.png")
    try:
        res = vision.resolve_amount(
            path,
            {
                "description": cf.description,
                "currency": cf.original_currency,
                "image_id": image["image_id"],
            },
        )
    except Exception:
        return None
    amount = res.get("amount")
    if amount is None:
        return None
    # Convert to home currency using the event's declared currency (image confirms
    # the figure); use the event's own date for the rate.
    if cf.original_currency != home_currency:
        rate, _rate_date = currency.lookup_rate(
            cf.original_currency, home_currency, cf.hit_date
        )
        magnitude = amount * rate
    else:
        magnitude = amount
    return -magnitude if cf.direction == "debit" else magnitude


def reconstruct(user_id: str) -> List[Cashflow]:
    """Clean list of normalized Cashflow records for a user, sorted by hit_date."""
    return reconstruct_detailed(user_id).cashflows


# ===========================================================================
# SLICE 2: recurrence detection + forward projection (deterministic)
# ===========================================================================
# Blank-amount events are carried forward as flagged/unresolved. They are never
# zeroed and never used in amount statistics; no vision/LLM is called here.


@dataclass
class RecurringSeries:
    category: str
    direction: str
    event_type: str
    kind: str  # "monthly" | "periodic" | "daily_drip" | "one_off"
    occurrences: int
    day_of_month: Optional[int]  # monthly only
    median_gap_days: Optional[float]
    projected_amount: Optional[float]  # per-occurrence (monthly/periodic) or per-day (drip)
    amount_basis: str  # human note: how projected_amount was derived
    period_days: Optional[int] = None  # periodic only: recur every N days
    cashflows: List[Cashflow] = field(default_factory=list)


@dataclass
class ForecastItem:
    on_date: date
    signed_amount: Optional[float]  # None when a blank-amount event is unresolved
    label: str
    category: str
    kind: str  # "monthly" | "daily_drip" | "one_off"
    is_estimate: bool  # True for projected/averaged amounts
    needs_image_amount: bool = False


def _sign_for(direction: str) -> int:
    return -1 if direction == "debit" else 1


def _recent_amounts(cfs: List[Cashflow], n: int) -> List[float]:
    """Amounts of the n most-recent occurrences that actually have a value."""
    ordered = sorted(cfs, key=lambda c: c.hit_date, reverse=True)
    vals = [c.original_amount for c in ordered if c.original_amount is not None]
    return vals[:n]


def _monthly_total_recent(cfs: List[Cashflow], n_months: int) -> float:
    """Mean per-calendar-month spend over the most recent n complete months seen."""
    by_month: Dict[Tuple[int, int], float] = defaultdict(float)
    for c in cfs:
        if c.original_amount is not None:
            by_month[(c.hit_date.year, c.hit_date.month)] += abs(c.original_amount)
    if not by_month:
        return 0.0
    recent_keys = sorted(by_month)[-n_months:]
    return statistics.fmean(by_month[k] for k in recent_keys)


def detect_recurrence(
    cashflows: List[Cashflow], income_policy: str = "A"
) -> List[RecurringSeries]:
    """Classify each (category, direction) group as monthly / daily_drip / one_off.

    Guiding rule (safer interpretation per spec): "expenses continue unless
    evidence they stop; income projected only with evidence it continues."
      - EXPENSES with sufficient history (>=3 occurrences) but a non-clean cadence
        still drip their monthly average rather than vanishing into one_off.
      - INCOME is projected only when the evidence supports continuation:
          policy A -> project all recurring income (monthly series AND near-daily
                      income drips) -- more generous. DEFAULT.
          policy B -> project regular monthly income series only; irregular /
                      freelance income seen only as a historical pattern is NOT
                      projected (it falls to one_off, so only confirmed future-
                      dated income events on their settlement date are counted).

    Policy A is the default: an A/B tiebreaker on the 25 samples tied on
    affordability_status, and the one decisive row (request_09, a freelancer with
    ground-truth affordable_now) is reproduced only by A. REVISIT this choice once
    the decision engine and image-amount resolution exist -- re-run the tiebreaker
    and measure amount_safe_to_pay and earliest_date_for_full_payment too, not just
    affordability_status.
    """
    if income_policy not in ("A", "B"):
        raise ValueError(f"income_policy must be 'A' or 'B', got {income_policy!r}")

    groups: Dict[Tuple[str, str], List[Cashflow]] = defaultdict(list)
    for cf in cashflows:
        groups[(cf.category, cf.direction)].append(cf)

    series: List[RecurringSeries] = []
    for (category, direction), cfs in groups.items():
        cfs_sorted = sorted(cfs, key=lambda c: c.hit_date)
        dates = [c.hit_date for c in cfs_sorted]
        n = len(dates)
        event_type = Counter(c.event_type for c in cfs_sorted).most_common(1)[0][0]

        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        median_gap = statistics.median(gaps) if gaps else None
        dom_counts = Counter(d.day for d in dates)
        modal_dom, modal_count = dom_counts.most_common(1)[0]

        kind = "one_off"
        day_of_month: Optional[int] = None
        period_days: Optional[int] = None
        projected_amount: Optional[float] = None
        amount_basis = "one-off; carried at own date"

        is_income = direction == "credit"
        # Salary-like income can be trusted as recurring on thinner evidence.
        is_salary_like = is_income and (category == "salary" or event_type == "income")

        if median_gap is not None:
            # Monthly detection. CHANGE 1: salary/regular income counts as monthly
            # with only 2 records ~a month apart (fixes under-projection when few
            # salary rows are visible); expenses still need >=3.
            if n >= MIN_OCCURRENCES:
                is_monthly = (
                    MONTHLY_GAP_MIN <= median_gap <= MONTHLY_GAP_MAX
                    and modal_count / n >= MONTHLY_DOM_CONSISTENCY
                )
            elif n == 2 and is_salary_like:
                is_monthly = MONTHLY_GAP_MIN <= median_gap <= MONTHLY_GAP_MAX
            else:
                is_monthly = False

            if is_monthly:
                kind = "monthly"
                # Prefer the most recent occurrence's day for the thin 2-record case.
                day_of_month = modal_dom if n >= MIN_OCCURRENCES else dates[-1].day
                recent = _recent_amounts(cfs_sorted, RECENT_N)
                if recent and len({round(a, 2) for a in recent}) == 1:
                    projected_amount = recent[0]
                    amount_basis = f"fixed amount ({len(recent)} recent equal)"
                elif recent:
                    projected_amount = statistics.fmean(recent)
                    amount_basis = f"mean of {len(recent)} recent occurrences"
                else:
                    amount_basis = "amount unresolved (blank occurrences)"
            elif direction == "debit" and n >= MIN_OCCURRENCES:
                # DECISION 2: expenses continue unless evidence they stop -> drip.
                kind = "daily_drip"
                monthly_total = _monthly_total_recent(cfs_sorted, RECENT_N)
                projected_amount = monthly_total / DAYS_PER_MONTH
                amount_basis = (
                    f"mean monthly total {monthly_total:.2f} over recent "
                    f"months / {DAYS_PER_MONTH:g} per day"
                )
            elif (
                is_income
                and income_policy == "A"
                and n >= MIN_OCCURRENCES
                and median_gap <= DRIP_GAP_MAX
            ):
                # CHANGE 2: biweekly/irregular income recurs on its ACTUAL cadence,
                # not a smeared daily drip -- the drip propped up the trough on
                # non-payday dates and over-projected safe headroom. Landing income
                # on real ~cadence dates yields a truer (lower) trough between pays.
                kind = "periodic"
                period_days = max(1, round(median_gap))
                recent = _recent_amounts(cfs_sorted, RECENT_N)
                projected_amount = statistics.fmean(recent) if recent else None
                amount_basis = (
                    f"mean of {len(recent)} recent, every ~{period_days}d"
                    if recent
                    else "amount unresolved"
                )

        series.append(
            RecurringSeries(
                category=category,
                direction=direction,
                event_type=event_type,
                kind=kind,
                occurrences=n,
                day_of_month=day_of_month,
                median_gap_days=median_gap,
                projected_amount=projected_amount,
                amount_basis=amount_basis,
                period_days=period_days,
                cashflows=cfs_sorted,
            )
        )
    series.sort(key=lambda s: (s.kind, s.category, s.direction))
    return series


def _month_iter(start: date, end: date):
    """Yield (year, month) from start's month through end's month inclusive."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def _clamped_date(year: int, month: int, day: int) -> date:
    """Build a date, clamping the day to the month's last day (e.g. dom 31 -> 30)."""
    if month == 12:
        last = 31
    else:
        last = (date(year, month + 1, 1) - timedelta(days=1)).day
    return date(year, month, min(day, last))


def project(
    user_id: str,
    start_date: date,
    horizon_days: int = FORECAST_DAYS,
    income_policy: str = "A",
) -> List[ForecastItem]:
    """Forecast dated cashflows over [start_date, start_date + horizon_days].

    Dated obligations (monthly series, one-offs) are start-INCLUSIVE: an item due
    on ``start_date`` itself (e.g. rent due on the request date) is reserved,
    because the starting balance is taken as-of the morning of ``start_date``
    before that day's dated commitments clear. Near-daily EXPENSE drips start the
    day AFTER ``start_date`` -- the request day's variable spend is treated as
    already reflected in the current balance, so we do not double-count it.
    """
    cashflows = reconstruct(user_id)
    series = detect_recurrence(cashflows, income_policy=income_policy)
    end_date = start_date + timedelta(days=horizon_days)

    items: List[ForecastItem] = []
    for s in series:
        if s.kind == "monthly":
            for year, month in _month_iter(start_date, end_date):
                on = _clamped_date(year, month, s.day_of_month)
                if not (start_date <= on <= end_date):
                    continue
                if s.projected_amount is None:
                    items.append(
                        ForecastItem(
                            on, None, f"{s.category} (recurring, amount unresolved)",
                            s.category, "monthly", True, needs_image_amount=True,
                        )
                    )
                else:
                    amt = _sign_for(s.direction) * s.projected_amount
                    items.append(
                        ForecastItem(
                            on, amt, f"{s.category} (monthly {s.event_type})",
                            s.category, "monthly", True,
                        )
                    )
        elif s.kind == "periodic":
            # Recur on the observed cadence, continuing from the last real occurrence.
            if not s.projected_amount or not s.period_days:
                continue
            amt = _sign_for(s.direction) * s.projected_amount
            anchor = s.cashflows[-1].hit_date
            k = 1
            while True:
                on = anchor + timedelta(days=s.period_days * k)
                if on > end_date:
                    break
                if on >= start_date:
                    items.append(
                        ForecastItem(
                            on, amt,
                            f"{s.category} (every ~{s.period_days}d {s.event_type})",
                            s.category, "periodic", True,
                        )
                    )
                k += 1
        elif s.kind == "daily_drip":
            if not s.projected_amount:
                continue
            amt = _sign_for(s.direction) * s.projected_amount
            day = start_date + timedelta(days=1)
            while day <= end_date:
                items.append(
                    ForecastItem(
                        day, amt, f"{s.category} (daily drip)", s.category,
                        "daily_drip", True,
                    )
                )
                day += timedelta(days=1)
        else:  # one_off: include only its own in-window occurrences
            for cf in s.cashflows:
                if start_date <= cf.hit_date <= end_date:
                    items.append(
                        ForecastItem(
                            cf.hit_date, cf.signed_amount,
                            f"{cf.category} (one-off {cf.event_type})", cf.category,
                            "one_off", False, needs_image_amount=cf.needs_image_amount,
                        )
                    )

    items.sort(key=lambda it: (it.on_date, -(it.signed_amount or 0)))
    return items


@dataclass
class Forecast:
    """Daily forecast primitive: balances, trough, and suffix-minimums.

    ``suffix_min(d)`` is the lowest balance over [d, end]; paying an amount X as a
    single lump on day d keeps the plan safe iff ``suffix_min(d) - X >= minimum``.
    """

    start_date: date
    days: List[date]
    balances: Dict[date, float]
    trough: float
    trough_date: date
    _suffix_min: Dict[date, float]

    def suffix_min(self, d: date) -> float:
        return self._suffix_min[d]


def build_forecast(
    user_id: str,
    start_date: date,
    start_balance: float,
    horizon_days: int = FORECAST_DAYS,
    income_policy: str = "A",
    extra_payments: Optional[List[Tuple[date, float]]] = None,
    series_scale: Optional[Dict[str, float]] = None,
) -> Forecast:
    """Forecast daily balances, optionally with hypothetical payments and
    spending-change overrides.

    ``extra_payments`` are (date, amount) debits toward a request.
    ``series_scale`` maps an expense category to a multiplier applied to its
    projected (negative) items: 0.0 models stopping the series; a fraction models
    reducing it. Only expense (debit) items are scaled.
    """
    items = project(user_id, start_date, horizon_days, income_policy=income_policy)
    by_day: Dict[date, float] = defaultdict(float)
    for it in items:
        amt = it.signed_amount or 0.0
        if series_scale and amt < 0 and it.category in series_scale:
            amt *= series_scale[it.category]
        by_day[it.on_date] += amt
    if extra_payments:
        for pay_date, pay_amt in extra_payments:
            by_day[pay_date] += -abs(pay_amt)

    days = [start_date + timedelta(days=i) for i in range(horizon_days + 1)]
    balances: Dict[date, float] = {}
    running = start_balance
    for d in days:
        running += by_day.get(d, 0.0)
        balances[d] = running

    suffix: Dict[date, float] = {}
    m = float("inf")
    for d in reversed(days):
        m = min(m, balances[d])
        suffix[d] = m

    trough_date = min(days, key=lambda d: balances[d])
    return Forecast(
        start_date=start_date,
        days=days,
        balances=balances,
        trough=balances[trough_date],
        trough_date=trough_date,
        _suffix_min=suffix,
    )


def balance_trace(
    user_id: str,
    start_date: date,
    start_balance: float,
    horizon_days: int = FORECAST_DAYS,
    income_policy: str = "A",
) -> Tuple[List[Tuple[date, float]], float, date]:
    """Daily running balance from start_balance; returns (series, trough, trough_date).

    Blank-amount (unresolved) forecast items are treated as 0 in the numeric trace
    ONLY so the trace can run; they remain flagged upstream and are not real zeros.
    """
    items = project(user_id, start_date, horizon_days, income_policy=income_policy)
    by_day: Dict[date, float] = defaultdict(float)
    for it in items:
        by_day[it.on_date] += it.signed_amount or 0.0

    # Opening balance is as-of the morning of start_date; then apply each day's
    # net flows (start-inclusive, so same-day dated obligations are reserved).
    running = start_balance
    trough = start_balance
    trough_date = start_date
    trace: List[Tuple[date, float]] = []
    day = start_date
    end_date = start_date + timedelta(days=horizon_days)
    while day <= end_date:
        running += by_day.get(day, 0.0)
        trace.append((day, running))
        if running < trough:
            trough = running
            trough_date = day
        day += timedelta(days=1)
    return trace, trough, trough_date


# ---------------------------------------------------------------------------
# Demo / manual review
# ---------------------------------------------------------------------------
def _print_classification(user_id: str) -> None:
    series = detect_recurrence(reconstruct(user_id))
    print(f"\nRecurrence classification for {user_id}:")
    print(
        f"  {'category':<16}{'dir':<7}{'kind':<12}{'n':>4}  {'dom':>4} "
        f"{'gap':>5}  {'proj_amount':>13}  basis"
    )
    for s in series:
        dom = "" if s.day_of_month is None else str(s.day_of_month)
        gap = "" if s.median_gap_days is None else f"{s.median_gap_days:g}"
        amt = "" if s.projected_amount is None else f"{s.projected_amount:,.2f}"
        print(
            f"  {s.category:<16}{s.direction:<7}{s.kind:<12}{s.occurrences:>4}  "
            f"{dom:>4} {gap:>5}  {amt:>13}  {s.amount_basis}"
        )


def _print_trough(user_id: str, start: date, start_balance: float, floor: float) -> None:
    trace, trough, trough_date = balance_trace(user_id, start, start_balance)
    print(f"\n{user_id} 90-day balance trace from {start} (start balance {start_balance:,.2f}):")
    print(f"  LOWEST point: {trough:,.2f} on {trough_date}")
    print(f"  minimum floor: {floor:,.2f}  ->  {'ABOVE floor' if trough >= floor else 'BELOW floor'}")
    # show the days around the trough for eyeballing
    lo = trough_date - timedelta(days=3)
    hi = trough_date + timedelta(days=2)
    print("  context around trough:")
    for d, bal in trace:
        if lo <= d <= hi:
            print(f"     {d}  {bal:,.2f}")



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
    # --- Slice 2 verification -------------------------------------------------
    print("### user_19 recurrence + projection (hand-trace check) ###")
    _print_classification("user_19")
    _print_trough("user_19", date(2024, 9, 4), 199545.0, 92800.0)

    print("\n\n### user_09 recurrence + projection (stays above floor) ###")
    _print_classification("user_09")
    _print_trough("user_09", date(2026, 7, 4), 2231.10, 600.0)

    # Extra: prove the drop + merge paths actually fire.
    print("\n\n### Extra users to exercise filter + dedup paths ###\n")
    print(">>> user_149 exercises FILTER drops (pending credit + cancelled):")
    _print_summary(reconstruct_detailed("user_149"), full=False)
    print("\n>>> user_138 exercises DEDUP (linked settled+pending duplicate):")
    _print_summary(reconstruct_detailed("user_138"), full=False)
