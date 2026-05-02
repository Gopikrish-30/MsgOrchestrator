"""
context_store.py — Versioned in-memory context storage.
Thread-safe. Idempotent on same (scope, context_id, version).
Higher version replaces lower version atomically.
"""
import threading
from typing import Any, Dict, Optional, Tuple

_lock = threading.Lock()
# Key: (scope, context_id) → {"version": int, "payload": dict}
_store: Dict[Tuple[str, str], Dict[str, Any]] = {}
# Fired suppression keys — prevents re-sending same trigger
_suppressed: set = set()
# Category index for fast lookup
_categories: Dict[str, dict] = {}
# Merchant index keyed by merchant_id
_merchants: Dict[str, dict] = {}
# Customer index keyed by customer_id
_customers: Dict[str, dict] = {}
# Trigger index keyed by trigger_id
_triggers: Dict[str, dict] = {}


def upsert(scope: str, context_id: str, version: int, payload: dict) -> Tuple[bool, int]:
    """
    Returns (accepted, current_version).
    Same-version reposts are idempotent no-ops.
    accepted=False only if incoming version is older than stored version.
    """
    key = (scope, context_id)
    with _lock:
        cur = _store.get(key)
        if cur:
            cur_version = cur["version"]
            if cur_version > version:
                return False, cur_version
            if cur_version == version:
                return True, cur_version
        _store[key] = {"version": version, "payload": payload}
        # Maintain fast-access indexes
        if scope == "category":
            _categories[payload.get("slug", context_id)] = payload
        elif scope == "merchant":
            _merchants[payload.get("merchant_id", context_id)] = payload
        elif scope == "customer":
            _customers[payload.get("customer_id", context_id)] = payload
        elif scope == "trigger":
            _triggers[payload.get("id", context_id)] = payload
        return True, version


def get(scope: str, context_id: str) -> Optional[dict]:
    key = (scope, context_id)
    with _lock:
        entry = _store.get(key)
        return entry["payload"] if entry else None


def get_version(scope: str, context_id: str) -> int:
    key = (scope, context_id)
    with _lock:
        entry = _store.get(key)
        return entry["version"] if entry else 0


def get_category(slug: str) -> Optional[dict]:
    with _lock:
        return _categories.get(slug)


def get_merchant(merchant_id: str) -> Optional[dict]:
    with _lock:
        return _merchants.get(merchant_id)


def get_customer(customer_id: str) -> Optional[dict]:
    with _lock:
        return _customers.get(customer_id)


def get_trigger(trigger_id: str) -> Optional[dict]:
    with _lock:
        return _triggers.get(trigger_id)


def get_all_triggers() -> list:
    with _lock:
        return list(_triggers.values())


def counts() -> dict:
    with _lock:
        return {
            "category": len(_categories),
            "merchant": len(_merchants),
            "customer": len(_customers),
            "trigger": len(_triggers),
        }


def suppress(key: str):
    with _lock:
        _suppressed.add(key)


def is_suppressed(key: str) -> bool:
    with _lock:
        return key in _suppressed


def clear():
    """Teardown — wipe all state."""
    with _lock:
        _store.clear()
        _categories.clear()
        _merchants.clear()
        _customers.clear()
        _triggers.clear()
        _suppressed.clear()
