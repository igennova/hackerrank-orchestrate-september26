"""Dataset loader for the "Buy or Wait?" agent.

Pure, deterministic loading + joining of the raw CSVs in ``dataset/``.
No decision logic, no currency math, no LLM calls, and nothing here mutates
any file under ``dataset/``.

The only public entry point is :func:`get_request_context`, which returns the
RAW joined records for a single request (the request row, the user's profile,
the user's financial events, the request's payment options, and any messages /
images tied to that user, request, or one of those events).
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so the loader works from any CWD)
# ---------------------------------------------------------------------------
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CODE_DIR)
DATASET_DIR = os.path.join(_REPO_ROOT, "dataset")
MEDIA_IMAGES_DIR = os.path.join(DATASET_DIR, "media", "images")

# Row = a single CSV record as an ordered dict of {column: raw string value}.
Row = Dict[str, str]


def _read_csv(name: str) -> List[Row]:
    """Read ``dataset/<name>.csv`` into a list of dict rows (values unchanged)."""
    path = os.path.join(DATASET_DIR, f"{name}.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class Dataset:
    """Holds every CSV in memory plus a few lookup indexes for fast joins."""

    def __init__(self) -> None:
        # Raw tables, exactly as stored on disk.
        self.requests: List[Row] = _read_csv("requests")
        self.sample_requests: List[Row] = _read_csv("sample_requests")
        self.financial_profiles: List[Row] = _read_csv("financial_profiles")
        self.financial_events: List[Row] = _read_csv("financial_events")
        self.request_payment_options: List[Row] = _read_csv("request_payment_options")
        self.exchange_rates: List[Row] = _read_csv("exchange_rates")
        self.messages: List[Row] = _read_csv("messages")
        self.images: List[Row] = _read_csv("images")

        self._build_indexes()

    # -- indexes ------------------------------------------------------------
    def _build_indexes(self) -> None:
        # request_id -> request row (eval requests take priority; samples fill in).
        self._request_by_id: Dict[str, Row] = {}
        for row in self.sample_requests:
            self._request_by_id.setdefault(row["request_id"], row)
        for row in self.requests:  # eval set wins on any id collision
            self._request_by_id[row["request_id"]] = row

        # user_id -> profile row.
        self._profile_by_user: Dict[str, Row] = {
            row["user_id"]: row for row in self.financial_profiles
        }

        # user_id -> [event rows]; event_id -> user_id (for message/image joins).
        self._events_by_user: Dict[str, List[Row]] = {}
        self._user_by_event: Dict[str, str] = {}
        for row in self.financial_events:
            self._events_by_user.setdefault(row["user_id"], []).append(row)
            self._user_by_event[row["event_id"]] = row["user_id"]

        # request_id -> [payment option rows].
        self._options_by_request: Dict[str, List[Row]] = {}
        for row in self.request_payment_options:
            self._options_by_request.setdefault(row["request_id"], []).append(row)

    # -- accessors ----------------------------------------------------------
    def get_request_row(self, request_id: str) -> Optional[Row]:
        return self._request_by_id.get(request_id)

    def get_request_context(self, request_id: str) -> Dict[str, Any]:
        """Return the RAW joined records for ``request_id`` (no transformation).

        Keys:
          request         request row (from requests.csv or sample_requests.csv)
          is_sample       True if it came from sample_requests.csv
          profile         the user's financial_profile row (or None)
          events          that user's financial_events rows
          payment_options the request's payment options
          messages        messages tied to the user, request, or those events
          images          images tied to the user, request, or those events
        """
        request = self.get_request_row(request_id)
        if request is None:
            raise KeyError(
                f"request_id {request_id!r} not found in requests.csv or sample_requests.csv"
            )

        is_sample = request_id not in {r["request_id"] for r in self.requests}
        user_id = request["user_id"]

        profile = self._profile_by_user.get(user_id)
        events = list(self._events_by_user.get(user_id, []))
        payment_options = list(self._options_by_request.get(request_id, []))

        # Event ids belonging to this user, for related_event_id joins.
        user_event_ids = {ev["event_id"] for ev in events}

        def _tied(row: Row) -> bool:
            related = row.get("related_event_id") or ""
            return (
                row.get("user_id") == user_id
                or row.get("request_id") == request_id
                or (related != "" and related in user_event_ids)
            )

        messages = [m for m in self.messages if _tied(m)]
        images = [im for im in self.images if _tied(im)]

        return {
            "request": request,
            "is_sample": is_sample,
            "profile": profile,
            "events": events,
            "payment_options": payment_options,
            "messages": messages,
            "images": images,
        }


# Module-level singleton so callers can just import and use.
_DATASET: Optional[Dataset] = None


def load_dataset(verbose: bool = True) -> Dataset:
    """Load (once) and return the shared Dataset, printing row counts."""
    global _DATASET
    if _DATASET is None:
        _DATASET = Dataset()
        if verbose:
            _print_row_counts(_DATASET)
    return _DATASET


def get_request_context(request_id: str) -> Dict[str, Any]:
    """Convenience wrapper that loads the dataset if needed, then joins."""
    return load_dataset(verbose=False).get_request_context(request_id)


def _print_row_counts(ds: Dataset) -> None:
    print("Loaded dataset row counts:")
    for label, rows in (
        ("requests", ds.requests),
        ("sample_requests", ds.sample_requests),
        ("financial_profiles", ds.financial_profiles),
        ("financial_events", ds.financial_events),
        ("request_payment_options", ds.request_payment_options),
        ("exchange_rates", ds.exchange_rates),
        ("messages", ds.messages),
        ("images", ds.images),
    ):
        print(f"  {label:<24} {len(rows):>6}")


# ---------------------------------------------------------------------------
# Readable dump for manual sanity-checking
# ---------------------------------------------------------------------------
def _dump_row(row: Optional[Row], indent: str = "    ") -> None:
    if row is None:
        print(f"{indent}(none)")
        return
    for key, value in row.items():
        print(f"{indent}{key}: {value}")


def print_request_context(request_id: str) -> None:
    ctx = get_request_context(request_id)
    src = "sample_requests.csv" if ctx["is_sample"] else "requests.csv"

    print("=" * 72)
    print(f"CONTEXT FOR {request_id}  (source: {src})")
    print("=" * 72)

    print("\n[REQUEST]")
    _dump_row(ctx["request"])

    print("\n[FINANCIAL PROFILE]")
    _dump_row(ctx["profile"])

    events = ctx["events"]
    print(f"\n[FINANCIAL EVENTS]  ({len(events)} rows)")
    for i, ev in enumerate(events, 1):
        print(f"  -- event {i}/{len(events)} --")
        _dump_row(ev, indent="      ")

    options = ctx["payment_options"]
    print(f"\n[PAYMENT OPTIONS]  ({len(options)} rows)")
    for i, opt in enumerate(options, 1):
        print(f"  -- option {i}/{len(options)} --")
        _dump_row(opt, indent="      ")

    messages = ctx["messages"]
    print(f"\n[MESSAGES]  ({len(messages)} rows)")
    for i, msg in enumerate(messages, 1):
        print(f"  -- message {i}/{len(messages)} --")
        _dump_row(msg, indent="      ")

    images = ctx["images"]
    print(f"\n[IMAGES]  ({len(images)} rows)")
    for i, img in enumerate(images, 1):
        image_id = img.get("image_id", "")
        path = os.path.join(MEDIA_IMAGES_DIR, f"{image_id}.png")
        exists = "exists" if os.path.exists(path) else "MISSING"
        print(f"  -- image {i}/{len(images)} --")
        _dump_row(img, indent="      ")
        print(f"      file: dataset/media/images/{image_id}.png ({exists})")


if __name__ == "__main__":
    load_dataset(verbose=True)  # prints row counts
    print()
    print_request_context("request_09")
