"""
main.py — Vera Challenge Bot — Complete implementation.

FastAPI + Groq (meta-llama/llama-4-scout-17b-16e-instruct)
All 5 endpoints: /healthz, /metadata, /context, /tick, /reply

ARCHITECTURE:
  POST /v1/context  → context_store (versioned, idempotent)
  POST /v1/tick     → composer_enhanced (trigger dispatch → LLM)
  POST /v1/reply    → conversation + composer (state machine + reply logic)
  
PHILOSOPHY:
  - Zero hallucinations: LLM uses only provided facts
  - Deterministic (temperature=0)
  - Category-specific voice enforcement
  - Smart trigger dispatch (25+ kinds handled)
  - Robust conversation state tracking
"""

import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv(override=True)

import context_store
import conversation_enhanced as conversation
import seed_loader
from composer_enhanced import compose, compose_reply

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vera-bot")

# ── Lifespan + App setup ──────────────────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pre-load all seed data
    logger.info("Starting up — pre-loading seed dataset...")
    seed_loader.load_all()
    logger.info(f"Startup complete. Contexts loaded: {context_store.counts()}")
    yield
    # Shutdown
    logger.info("Shutting down.")

app = FastAPI(
    title="Vera Challenge Bot",
    description="magicpin Vera AI Challenge — Message Composition Engine v2.0",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

# ── State tracking ─────────────────────────────────────────────────────────
SUPPRESSED_CONVOS: set = set()  # conversation_id → suppressed (merchant said stop)
CONV_META: Dict[str, Dict[str, Any]] = {}  # conv_id → {merchant_id, trigger_id, customer_id}


# ── Pydantic models ────────────────────────────────────────────────────────
class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str = ""


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str = "merchant"
    message: str
    received_at: str = ""
    turn_number: int = 1


# ── Helper: resolve contexts by ID ─────────────────────────────────────────
def _resolve_contexts(merchant_id: str, trigger_id: str, customer_id: Optional[str]):
    """Load category, merchant, trigger, customer from store by ID."""
    merchant = context_store.get_merchant(merchant_id)
    if not merchant:
        merchant = context_store.get("merchant", merchant_id)
    if not merchant:
        return None, None, None, None

    cat_slug = merchant.get("category_slug", "")
    category = context_store.get_category(cat_slug)
    if not category:
        category = context_store.get("category", cat_slug)

    trigger = context_store.get_trigger(trigger_id)
    if not trigger:
        trigger = context_store.get("trigger", trigger_id)

    customer = None
    if customer_id:
        customer = context_store.get_customer(customer_id)
        if not customer:
            customer = context_store.get("customer", customer_id)

    return category, merchant, trigger, customer


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1: GET /v1/healthz
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    """Health check endpoint — confirms bot is ready."""
    counts = context_store.counts()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2: GET /v1/metadata
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/metadata")
async def metadata():
    """Metadata about the bot, team, and approach."""
    return {
        "team_name": os.getenv("TEAM_NAME", "Vera Engine v2"),
        "team_members": [m.strip() for m in os.getenv("TEAM_MEMBERS", "Solo Builder").split(",")],
        "model": f"{os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')} via Groq",
        "approach": (
            "4-context dispatch engine: "
            "1) Extract concrete facts only (never hallucinate). "
            "2) Trigger-kind dispatch (25+ strategies). "
            "3) Category voice enforcement (clinical/visual/timely/utility). "
            "4) LLM composition at temp=0 (deterministic). "
            "5) Conversation state machine (auto-reply detection, intent clarity, graceful exit). "
            "6) All outputs grounded in provided context."
        ),
        "contact_email": os.getenv("CONTACT_EMAIL", "builder@example.com"),
        "github": os.getenv("GITHUB_REPO", ""),
        "version": "2.0.0-enhanced",
        "rubric_targets": {
            "decision_quality": "Trigger fit + merchant state + category fit all evident",
            "specificity": "Real numbers, offers, dates — extracted only, never invented",
            "category_fit": "Voice true to vertical (dentist=peer-clinical, salon=warm, etc.)",
            "merchant_fit": "Personalized to metrics, offers, conversation history",
            "engagement_compulsion": "2-3 levers per message (proof, urgency, curiosity, effort-external)"
        },
        "submitted_at": os.getenv("SUBMITTED_AT", datetime.now(timezone.utc).isoformat()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3: POST /v1/context
# ─────────────────────────────────────────────────────────────────────────────

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


@app.post("/v1/context")
async def push_context(body: ContextBody, response: Response):
    """Store context (category, merchant, customer, trigger) — idempotent by version."""
    if body.scope not in VALID_SCOPES:
        response.status_code = 400
        return {
            "accepted": False,
            "reason": "invalid_scope",
            "details": f"scope must be one of {VALID_SCOPES}",
        }

    accepted, current_version = context_store.upsert(
        body.scope, body.context_id, body.version, body.payload
    )

    if not accepted:
        response.status_code = 409
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": current_version,
        }

    ack_id = f"ack_{body.context_id}_v{body.version}"
    stored_at = datetime.now(timezone.utc).isoformat()

    logger.info(
        f"Context stored: scope={body.scope} id={body.context_id} v{body.version}"
    )

    return {
        "accepted": True,
        "ack_id": ack_id,
        "stored_at": stored_at,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4: POST /v1/tick
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/v1/tick")
async def tick(body: TickBody):
    """
    Main message dispatch endpoint.
    Processes available triggers, composes messages, returns actions.
    """
    actions = []
    now_str = body.now

    # Score triggers by urgency, process highest priority first
    scored_triggers = []
    for trg_id in body.available_triggers:
        # Load trigger from store
        trg_payload = context_store.get_trigger(trg_id)
        if not trg_payload:
            trg_payload = context_store.get("trigger", trg_id)
        if not trg_payload:
            logger.debug(f"Trigger not found: {trg_id}")
            continue

        # Skip globally suppressed
        sup_key = trg_payload.get("suppression_key", "")
        if context_store.is_suppressed(sup_key):
            logger.info(f"Skipping suppressed trigger: {trg_id}")
            continue

        urgency = trg_payload.get("urgency", 0)
        scored_triggers.append((urgency, trg_id, trg_payload))

    # Sort highest urgency first
    scored_triggers.sort(key=lambda x: x[0], reverse=True)

    for urgency, trg_id, trg_payload in scored_triggers:
        if len(actions) >= 20:  # Spec max 20 actions per tick
            break

        merchant_id = trg_payload.get("merchant_id")
        customer_id = trg_payload.get("customer_id")
        if not merchant_id:
            logger.debug(f"Trigger {trg_id} missing merchant_id")
            continue

        # Build conversation_id
        trg_kind = trg_payload.get("kind", "generic")
        safe_mid = merchant_id.replace(" ", "_").replace("/", "_")
        safe_tid = trg_id.replace(" ", "_").replace("/", "_")
        conv_id = f"conv_{safe_mid}_{safe_tid}"

        # Skip if conversation already suppressed
        if conv_id in SUPPRESSED_CONVOS:
            logger.debug(f"Conversation suppressed: {conv_id}")
            continue

        # Resolve all contexts
        category, merchant, trigger, customer = _resolve_contexts(
            merchant_id, trg_id, customer_id
        )

        if not merchant or not category or not trigger:
            logger.warning(
                f"Missing context for trigger: merchant={merchant_id} "
                f"trigger={trg_id} (has_cat={category is not None})"
            )
            continue

        # Compose message
        try:
            result = compose(
                category=category,
                merchant=merchant,
                trigger=trigger,
                customer=customer,
                conversation_history=merchant.get("conversation_history", []),
                is_first_turn=True,
            )
        except Exception as e:
            logger.error(f"Compose error for {merchant_id}/{trg_id}: {e}", exc_info=True)
            continue

        body_text = (result.get("body") or "").strip()
        if not body_text:
            logger.warning(f"Empty body for {merchant_id}/{trg_id}, skipping")
            continue

        # Suppress this trigger so it won't fire again
        sup_key = trg_payload.get("suppression_key", "")
        if sup_key:
            context_store.suppress(sup_key)

        # Record bot turn in conversation state
        conversation.get_or_create(conv_id, merchant_id=merchant_id, trigger_id=trg_id)
        conversation.record_bot_turn(conv_id, body_text)

        # Save conversation metadata for /reply lookups
        CONV_META[conv_id] = {
            "merchant_id": merchant_id,
            "trigger_id": trg_id,
            "customer_id": customer_id,
        }

        # Determine send_as
        send_as = result.get("send_as", "vera")
        if customer_id and trigger.get("scope") == "customer":
            send_as = "merchant_on_behalf"

        # Build action
        identity = merchant.get("identity", {})
        owner = identity.get("owner_first_name") or identity.get("name", "them")
        template_params = [
            owner,
            trg_kind,
            body_text[:100],
        ]

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id or None,
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": f"vera_{trg_kind}_v1",
            "template_params": template_params,
            "body": body_text,
            "cta": result.get("cta", "open_ended"),
            "suppression_key": sup_key or f"{trg_kind}:{merchant_id}",
            "rationale": result.get("rationale", "Composed from 4-context framework"),
        }
        actions.append(action)
        logger.info(
            f"Action queued: conv={conv_id[:20]}... kind={trg_kind} to {merchant_id}"
        )

    logger.info(f"Tick complete: {len(actions)} actions queued")
    return {"actions": actions}


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 5: POST /v1/reply
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    """
    Handle incoming reply from merchant or customer.
    Update conversation state, detect intent/auto-reply, compose next message.
    """
    conv_id = body.conversation_id
    merchant_id = body.merchant_id or CONV_META.get(conv_id, {}).get("merchant_id")
    customer_id = body.customer_id or CONV_META.get(conv_id, {}).get("customer_id")
    trigger_id = CONV_META.get(conv_id, {}).get("trigger_id")
    message = body.message.strip()
    turn_number = body.turn_number

    if not merchant_id or not trigger_id:
        return {
            "action": "end",
            "conversation_state": "unknown",
            "rationale": "Missing conversation context",
            "reply": ""
        }

    # Get conversation state
    conv = conversation.get_or_create(
        conv_id, merchant_id=merchant_id, trigger_id=trigger_id
    )

    # Record merchant's turn + analyze
    analysis = conversation.record_merchant_turn(conv_id, message)

    logger.info(
        f"Reply: conv={conv_id[:20]}... is_auto={analysis['is_auto_reply']} "
        f"intent={analysis['intent']} turn={turn_number}"
    )

    # If exit intent or suppressed — end gracefully
    if analysis["intent"] == "exit":
        SUPPRESSED_CONVOS.add(conv_id)
        response = {
            "action": "end",
            "conversation_state": "ended",
            "suppression_key": f"exit:{conv_id}",
            "rationale": "Merchant explicitly opted out. Conversation closed for this session.",
            "reply": ""
        }
        logger.info(f"Conversation ended: {conv_id} (merchant exit)")
        return response

    # If auto-reply, handle gracefully
    if analysis["is_auto_reply"]:
        auto_count = conv.get("auto_reply_count", 0)
        if auto_count >= 3:
            # Hard end after 3 auto-replies
            SUPPRESSED_CONVOS.add(conv_id)
            response = {
                "action": "end",
                "conversation_state": "ended",
                "suppression_key": f"auto_reply_hard_end:{conv_id}",
                "rationale": "Auto-reply detected repeatedly; closing conversation.",
                "reply": ""
            }
            logger.info(f"Hard end after 3 auto-replies: {conv_id}")
            return response
        elif auto_count == 1:
            # First auto-reply, back off immediately for a fresh conversation.
            response = {
                "action": "wait",
                "wait_seconds": 14400,
                "conversation_state": "paused",
                "suppression_key": f"auto_reply_first:{conv_id}",
                "rationale": "Detected merchant auto-reply; backing off before the next attempt.",
                "reply": ""
            }
            logger.info(f"Backing off after auto-reply: {conv_id}")
            return response
        else:
            # 2nd auto-reply, wait 24h
            response = {
                "action": "wait",
                "wait_seconds": 86400,
                "conversation_state": "paused",
                "suppression_key": f"auto_reply_wait_24h:{conv_id}",
                "rationale": "Repeated auto-reply detected; waiting 24 hours.",
                "reply": ""
            }
            logger.info(f"Waiting 24h after 2nd auto-reply: {conv_id}")
            return response

    # If action intent, move to execution mode
    if analysis["intent"] == "action":
        try:
            category, merchant, trigger, customer = _resolve_contexts(
                merchant_id, trigger_id, customer_id
            )
            if merchant and category and trigger:
                # Compose execution-mode message
                result = compose_reply(
                    merchant=merchant,
                    customer=customer or {},
                    trigger=trigger,
                    category=category,
                    merchant_reply=message,
                    conversation_history=conv.get("turns", [])
                )
                
                reply_text = (result.get("body") or "").strip()
                if reply_text:
                    conversation.record_bot_turn(conv_id, reply_text)
                    response = {
                            "action": result.get("action", "send"),
                            "reply": reply_text,
                            "body": reply_text,
                            "cta": result.get("cta", "open_ended"),
                            "rationale": result.get("rationale", "Composed execution-mode reply."),
                        "conversation_state": "active",
                        "suppression_key": f"execution_mode:{conv_id}",
                    }
                    logger.info(f"Execution mode reply: {conv_id}")
                    return response
        except Exception as e:
            logger.error(f"Execution mode compose error: {e}", exc_info=True)

    # Default: acknowledgment + keep conversation alive
    response = {
        "action": "send",
        "reply": "Thanks for the update. Anything else I can help with?",
        "body": "Thanks for the update. Anything else I can help with?",
        "cta": "open_ended",
        "conversation_state": "active",
        "rationale": "Default acknowledgment; keep the conversation alive.",
        "suppression_key": f"default_ack:{conv_id}",
    }
    conversation.record_bot_turn(conv_id, response["reply"])
    logger.info(f"Default acknowledgment: {conv_id}")
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Startup logging
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting Vera Bot on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
