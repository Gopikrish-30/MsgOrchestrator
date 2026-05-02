"""
seed_loader.py — Pre-loads the full dataset into context_store at startup.

Reads from dataset/categories/*.json and dataset/*_seed.json
so the bot is warm before the judge starts pushing context.

The judge will ALSO push these via /v1/context — that's fine,
the idempotency logic handles it (same or higher version wins).
"""
import json
import os
import logging
from pathlib import Path

import context_store

logger = logging.getLogger("seed_loader")

DATASET_DIR = Path(os.getenv("DATASET_DIR", "dataset"))
EXPANDED_DIR = Path(os.getenv("EXPANDED_DIR", "expanded"))


def load_categories():
    """Load all 5 category JSONs."""
    cat_dir = DATASET_DIR / "categories"
    if not cat_dir.exists():
        logger.warning(f"Category dir not found: {cat_dir}")
        return 0

    count = 0
    for f in cat_dir.glob("*.json"):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
            slug = payload.get("slug", f.stem)
            accepted, _ = context_store.upsert("category", slug, 1, payload)
            if accepted:
                count += 1
                logger.info(f"Loaded category: {slug}")
        except Exception as e:
            logger.error(f"Failed to load category {f}: {e}")

    return count


def load_merchants(source_dir: Path = None):
    """Load merchants from expanded individual files, or seed file."""
    dirs_to_try = []
    if source_dir:
        dirs_to_try.append(source_dir)
    dirs_to_try.extend([EXPANDED_DIR, DATASET_DIR])

    # First try expanded individual-files format: merchants/*.json
    for d in dirs_to_try:
        subdir = d / "merchants"
        if subdir.exists() and subdir.is_dir():
            count = 0
            for f in subdir.glob("*.json"):
                try:
                    item = json.loads(f.read_text(encoding="utf-8"))
                    mid = item.get("merchant_id", f.stem)
                    accepted, _ = context_store.upsert("merchant", mid, 1, item)
                    if accepted:
                        count += 1
                except Exception as e:
                    logger.error(f"Failed loading merchant {f}: {e}")
            if count:
                logger.info(f"Loaded {count} merchants from {subdir}/")
                return count

    # Fall back to seed JSON list format
    for d in dirs_to_try:
        for name in ["merchants_seed.json", "merchants.json"]:
            f = d / name
            if f.exists():
                return _load_json_list(f, "merchant", "merchant_id")

    logger.warning("No merchants file found")
    return 0


def load_customers(source_dir: Path = None):
    dirs_to_try = []
    if source_dir:
        dirs_to_try.append(source_dir)
    dirs_to_try.extend([EXPANDED_DIR, DATASET_DIR])

    for d in dirs_to_try:
        subdir = d / "customers"
        if subdir.exists() and subdir.is_dir():
            count = 0
            for f in subdir.glob("*.json"):
                try:
                    item = json.loads(f.read_text(encoding="utf-8"))
                    cid = item.get("customer_id", f.stem)
                    accepted, _ = context_store.upsert("customer", cid, 1, item)
                    if accepted:
                        count += 1
                except Exception as e:
                    logger.error(f"Failed loading customer {f}: {e}")
            if count:
                logger.info(f"Loaded {count} customers from {subdir}/")
                return count

    for d in dirs_to_try:
        for name in ["customers_seed.json", "customers.json"]:
            f = d / name
            if f.exists():
                return _load_json_list(f, "customer", "customer_id")

    logger.warning("No customers file found")
    return 0


def load_triggers(source_dir: Path = None):
    dirs_to_try = []
    if source_dir:
        dirs_to_try.append(source_dir)
    dirs_to_try.extend([EXPANDED_DIR, DATASET_DIR])

    for d in dirs_to_try:
        subdir = d / "triggers"
        if subdir.exists() and subdir.is_dir():
            count = 0
            for f in subdir.glob("*.json"):
                try:
                    item = json.loads(f.read_text(encoding="utf-8"))
                    tid = item.get("id", f.stem)
                    accepted, _ = context_store.upsert("trigger", tid, 1, item)
                    if accepted:
                        count += 1
                except Exception as e:
                    logger.error(f"Failed loading trigger {f}: {e}")
            if count:
                logger.info(f"Loaded {count} triggers from {subdir}/")
                return count

    for d in dirs_to_try:
        for name in ["triggers_seed.json", "triggers.json"]:
            f = d / name
            if f.exists():
                return _load_json_list(f, "trigger", "id")

    logger.warning("No triggers file found")
    return 0


def _load_json_list(filepath: Path, scope: str, id_field: str) -> int:
    """Load a JSON file that is either a list or {"merchants": [...]} style."""
    try:
        raw = json.loads(filepath.read_text(encoding="utf-8"))

        # Handle both top-level list and keyed dict
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            # Try common keys
            for key in ["merchants", "customers", "triggers", "data", "items"]:
                if key in raw:
                    items = raw[key]
                    break
            else:
                items = list(raw.values())
        else:
            logger.error(f"Unexpected format in {filepath}")
            return 0

        count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = item.get(id_field, "")
            if not item_id:
                continue
            accepted, _ = context_store.upsert(scope, item_id, 1, item)
            if accepted:
                count += 1

        logger.info(f"Loaded {count} {scope}s from {filepath}")
        return count

    except Exception as e:
        logger.error(f"Failed to load {scope}s from {filepath}: {e}")
        return 0


def load_all():
    """Load the full base dataset. Called at startup."""
    logger.info("=== Seed loader starting ===")

    n_cat = load_categories()
    n_mer = load_merchants()
    n_cus = load_customers()
    n_trg = load_triggers()

    total = n_cat + n_mer + n_cus + n_trg
    logger.info(
        f"=== Seed loader done: {n_cat} categories, "
        f"{n_mer} merchants, {n_cus} customers, {n_trg} triggers "
        f"= {total} total ==="
    )
    return {
        "category": n_cat,
        "merchant": n_mer,
        "customer": n_cus,
        "trigger": n_trg,
    }


if __name__ == "__main__":
    load_all()
    print(context_store.counts())
