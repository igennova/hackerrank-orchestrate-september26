"""Currency conversion for the "Buy or Wait?" agent.

Small, pure, deterministic. No LLM calls, no network, nothing mutated on disk.

The dataset only ever needs these five directed conversions, and every one of
them exists directly in ``exchange_rates.csv``:

    USD->INR, USD->IDR, USD->EUR, EUR->ZAR, EUR->USD

So there is deliberately NO inversion and NO triangulation here. If a pair is
missing we raise, rather than guess.

Rates are published only on the 1st and 15th of each month, so exact-date hits
are rare: we pick the most recent ``rate_date`` on or before ``on_date`` (and if
``on_date`` precedes the earliest rate for the pair, we fall back to the earliest).
"""

from __future__ import annotations

import csv
import os
from datetime import date
from typing import Dict, List, Optional, Tuple, Union

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CODE_DIR)
_RATES_PATH = os.path.join(_REPO_ROOT, "dataset", "exchange_rates.csv")

DateLike = Union[str, date]


def _to_date(value: DateLike) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


# (from_currency, to_currency) -> list of (rate_date, rate), sorted ascending by date.
_RATES: Optional[Dict[Tuple[str, str], List[Tuple[date, float]]]] = None


def _load_rates() -> Dict[Tuple[str, str], List[Tuple[date, float]]]:
    """Parse exchange_rates.csv once into a per-pair ascending-by-date table."""
    global _RATES
    if _RATES is None:
        table: Dict[Tuple[str, str], List[Tuple[date, float]]] = {}
        with open(_RATES_PATH, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row["from_currency"], row["to_currency"])
                table.setdefault(key, []).append(
                    (_to_date(row["rate_date"]), float(row["rate"]))
                )
        for series in table.values():
            series.sort(key=lambda pair: pair[0])
        _RATES = table
    return _RATES


def lookup_rate(
    from_currency: str, to_currency: str, on_date: DateLike
) -> Tuple[float, date]:
    """Return (rate, rate_date_used) for a directed pair as of ``on_date``.

    Uses the most recent rate on or before ``on_date``; if ``on_date`` precedes
    the earliest available rate for the pair, uses the earliest. Raises
    ``KeyError`` if the pair does not exist at all.
    """
    series = _load_rates().get((from_currency, to_currency))
    if not series:
        raise KeyError(
            f"No exchange rate for {from_currency}->{to_currency}; "
            "refusing to guess (no inversion/triangulation by design)."
        )

    target = _to_date(on_date)
    chosen = series[0]  # earliest, used when target precedes all available rates
    for rate_date, rate in series:
        if rate_date <= target:
            chosen = (rate_date, rate)
        else:
            break
    rate_date, rate = chosen
    return rate, rate_date


def convert(
    amount: Union[str, float, int],
    from_currency: str,
    to_currency: str,
    on_date: DateLike,
) -> float:
    """Convert ``amount`` from ``from_currency`` to ``to_currency`` as of ``on_date``.

    Same-currency conversions return the amount unchanged (no lookup). Otherwise
    the dated rate is applied. Raises ``KeyError`` if the pair is unknown.
    """
    value = float(amount)
    if from_currency == to_currency:
        return value
    rate, _rate_date = lookup_rate(from_currency, to_currency, on_date)
    return value * rate


# ---------------------------------------------------------------------------
# Self-check (no ground truth required)
# ---------------------------------------------------------------------------
def _self_check() -> None:
    home_by_user: Dict[str, str] = {}
    profiles_path = os.path.join(_REPO_ROOT, "dataset", "financial_profiles.csv")
    with open(profiles_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            home_by_user[row["user_id"]] = row["home_currency"]

    events_path = os.path.join(_REPO_ROOT, "dataset", "financial_events.csv")
    converted = 0
    examples: List[str] = []
    with open(events_path, newline="", encoding="utf-8") as fh:
        for ev in csv.DictReader(fh):
            home = home_by_user.get(ev["user_id"])
            if not home or ev["currency"] == home:
                continue
            # Use the settlement date for a cash event; fall back to event_date.
            on_date = ev["settlement_date"] or ev["event_date"]

            # Resolution is about the rate, which must exist regardless of amount.
            rate, rate_date = lookup_rate(ev["currency"], home, on_date)
            converted += 1

            if len(examples) < 3 and (ev["amount"] or "").strip():
                out = convert(ev["amount"], ev["currency"], home, on_date)
                examples.append(
                    f"  {ev['event_id']}: {ev['amount']} {ev['currency']} "
                    f"-> {out:.2f} {home}  "
                    f"(rate {rate} on {rate_date.isoformat()}, as of {on_date})"
                )

    print(f"Converted (rate resolved) foreign-currency event rows: {converted}")
    print("Worked examples:")
    for line in examples:
        print(line)

    assert converted == 140, f"expected 140 conversions, got {converted}"
    print("OK: all foreign-currency events resolved; count == 140.")


if __name__ == "__main__":
    _self_check()
