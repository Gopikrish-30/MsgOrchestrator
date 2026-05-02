#!/usr/bin/env python3
"""
generate_submission.py — Generates submission.jsonl with 30 pre-composed messages.

Run AFTER expanding the dataset:
    python dataset/generate_dataset.py --seed-dir dataset --out expanded
    python generate_submission.py

Outputs:
    submission.jsonl  — 30 lines, one per canonical test pair
"""

import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

import context_store
import seed_loader
from composer_enhanced import compose

DATASET_DIR = Path("dataset")
EXPANDED_DIR = Path("expanded")
OUT_FILE = Path("submission.jsonl")


def load_dataset():
    """Load categories + expanded dataset."""
    print("Loading categories...")
    seed_loader.load_categories()

    if not (EXPANDED_DIR / "merchants").exists():
        raise SystemExit(
            "Expanded dataset not found. Run: python dataset/generate_dataset.py --seed-dir dataset --out expanded"
        )

    print("Loading expanded dataset (50 merchants)...")
    seed_loader.load_merchants(EXPANDED_DIR)
    seed_loader.load_customers(EXPANDED_DIR)
    seed_loader.load_triggers(EXPANDED_DIR)

    counts = context_store.counts()
    print(f"Loaded: {counts}")
    return counts


def load_test_pairs():
    """Load the 30 canonical test pairs."""
    test_file = EXPANDED_DIR / "test_pairs.json"
    if not test_file.exists():
        raise SystemExit(
            "expanded/test_pairs.json not found. Run dataset/generate_dataset.py before generate_submission.py"
        )

    data = json.loads(test_file.read_text())
    pairs = data.get("pairs", [])
    if len(pairs) != 30:
        raise SystemExit(f"Expected 30 canonical test pairs, found {len(pairs)}")
    return pairs


def run():
    load_dataset()
    pairs = load_test_pairs()
    print(f"\nGenerating {len(pairs)} compositions...")

    results = []
    errors = 0

    for i, pair in enumerate(pairs):
        tid = pair["trigger_id"]
        mid = pair["merchant_id"]
        cid = pair.get("customer_id")
        test_id = pair["test_id"]

        merchant = context_store.get_merchant(mid) or context_store.get("merchant", mid)
        trigger = context_store.get_trigger(tid) or context_store.get("trigger", tid)
        customer = None
        if cid:
            customer = context_store.get_customer(cid) or context_store.get("customer", cid)

        if not merchant or not trigger:
            print(f"  [{test_id}] SKIP — missing merchant={mid} or trigger={tid}")
            errors += 1
            continue

        cat_slug = merchant.get("category_slug", "")
        category = context_store.get_category(cat_slug) or context_store.get("category", cat_slug)

        if not category:
            print(f"  [{test_id}] SKIP — missing category={cat_slug}")
            errors += 1
            continue

        try:
            start = time.time()
            result = compose(
                category=category,
                merchant=merchant,
                trigger=trigger,
                customer=customer,
                conversation_history=[],
                is_first_turn=True,
            )
            elapsed = time.time() - start

            row = {
                "test_id": test_id,
                "trigger_id": tid,
                "merchant_id": mid,
                "customer_id": cid,
                "trigger_kind": trigger.get("kind", "unknown"),
                "body": result.get("body", ""),
                "cta": result.get("cta", "open_ended"),
                "send_as": result.get("send_as", "vera"),
                "suppression_key": result.get("suppression_key", ""),
                "rationale": result.get("rationale", ""),
            }
            results.append(row)

            body_preview = result.get("body", "")[:60]
            print(f"  [{test_id}] OK ({elapsed:.1f}s) kind={trigger.get('kind')} — \"{body_preview}...\"")

        except Exception as e:
            print(f"  [{test_id}] ERROR — {e}")
            errors += 1

    # Write JSONL
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✓ Wrote {len(results)} entries to {OUT_FILE} ({errors} errors)")
    print(f"  Average body length: {sum(len(r['body']) for r in results)//max(len(results),1)} chars")

    # Show breakdown by trigger kind
    by_kind = {}
    for r in results:
        k = r["trigger_kind"]
        by_kind[k] = by_kind.get(k, 0) + 1
    print("\nBy trigger kind:")
    for k, n in sorted(by_kind.items()):
        print(f"  {k}: {n}")


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: Set GROQ_API_KEY in .env or environment before running.")
        sys.exit(1)
    run()
