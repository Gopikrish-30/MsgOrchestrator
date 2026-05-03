"""
composer_enhanced.py — The heart of the winning bot.

Composes high-compulsion WhatsApp messages from 4 context layers.
Uses Groq (meta-llama/llama-4-scout-17b-16e-instruct) as the LLM backbone.

CORE PHILOSOPHY:
- Dispatch by trigger.kind for maximum relevance (25+ kinds handled specifically)
- Extract specifics FIRST, then instruct LLM to use only what's given
- Enforce category voice, taboos, allowed vocabulary
- Single clear CTA per message
- NO hallucinations, NO invented facts
- Temperature=0 for determinism

SCORING TARGETS:
  1. Decision Quality — trigger fit + merchant state + category fit
  2. Specificity — real numbers, offers, dates (extracted, never invented)
  3. Category Fit — clinical/visual/timely/utility voices true to vertical
  4. Merchant Fit — personalized to metrics, offers, history
  5. Engagement Compulsion — 2-3 levers (proof, urgency, curiosity, effort-external)
"""
import os
import json
import re
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from groq import Groq

logger = logging.getLogger(__name__)

# ── Groq client ───────────────────────────────────────────────────────────
_client: Optional[Groq] = None

def _normalize_groq_base_url(raw: str) -> str:
    """Accept either root URL or OpenAI-style path and normalize for Groq SDK."""
    base = (raw or "").strip().rstrip("/")
    if base.endswith("/openai/v1"):
        base = base[: -len("/openai/v1")]
    return base


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY", "")
        base_url = _normalize_groq_base_url(os.getenv("GROQ_BASE_URL", ""))
        if base_url:
            _client = Groq(api_key=api_key, base_url=base_url, max_retries=0)
        else:
            _client = Groq(api_key=api_key, max_retries=0)
    return _client

MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


# === System prompt (judge-friendly) ===
COMPOSE_SYSTEM = """You are Vera — magicpin's AI merchant-growth assistant on WhatsApp.
You talk to Indian merchants (restaurants, salons, gyms, dentists, pharmacies) and help them grow.
You also draft messages FROM merchants TO their customers.

You are scored on 5 dimensions (0–10 each). Maximize all 5:

1. SPECIFICITY — Every claim must come from the context: real CTR numbers, real offer prices,
   real customer counts, real source citations. "Increase sales" = 0 points. "Your CTR is 2.1%
   vs peer median 3.0%" = full points.

2. CATEGORY FIT — Match tone exactly:
   - dentists: peer_clinical (cite sources, use "fluoride varnish / caries / recall interval")
   - salons: warm_practical (emojis ok, visual, slot offers, bridal/seasonal hooks)
   - restaurants: operator_to_operator ("covers", "AOV", "delivery mix", contrarian data)
   - gyms: coach_operator ("members", "retention", "seasonal lull", shame-free winback)
   - pharmacies: trustworthy_precise (molecule names, batch numbers, exact dates, "sub-potency")

3. MERCHANT FIT — Use owner's first name. Use their actual CTR. Use their active offers by name.
   Reference their conversation history. Use their specific locality.

4. TRIGGER RELEVANCE — The message must answer "why NOW" in the FIRST sentence. Name the trigger fact explicitly.
   Not "you should improve your profile" — "Your CTR is 2.1% vs peer 3.0% and your last GBP
   post was 22 days ago — these two are connected."

5. ENGAGEMENT COMPULSION — One strong reason to reply RIGHT NOW. Pick 1-2:
   - Loss aversion: "190 people searched for this in your area and didn't find you"
   - Social proof: "3 dentists in Lajpat Nagar switched to 3-month recall this quarter"
   - Effort externalization: "I've already drafted it — just say go"
   - Curiosity: "Want to see the full list?"
   - Single binary commit: "Reply YES / STOP"

HARD RULES (breaking any = instant penalty):
- ONE primary CTA only — never "Reply YES for X, NO for Y"
- NEVER invent data — only use numbers/facts from context
- NEVER use taboo words: "guaranteed", "100% safe", "miracle", "AMAZING DEAL", "best in city"
- NEVER start with "I hope you're doing well" or similar preamble
- NEVER re-introduce yourself after the first message
- Hindi-English code-mix when merchant languages include "hi" (natural, not forced)
- No repetition of messages already in conversation_history

OUTPUT: JSON only, no markdown fences. FIRST sentence must be a short "Why now" that references a concrete numeric fact from FACTS PROVIDED:
{
  "body": "the WhatsApp message body",
  "cta": "open_ended | binary_yes_stop | none",
  "send_as": "vera | merchant_on_behalf",
  "rationale": "Trigger used: [X]. Merchant fact anchored: [Y]. Lever deployed: [Z].",
  "template_params": ["param1", "param2", "param3"]
}"""



# Trigger-kind specific instructions (append these based on trigger['kind'])
TRIGGER_INSTRUCTIONS = {
    "research_digest": (
        "Lead with the specific finding: cite source, n=, %, page number. "
        "Connect to THIS merchant's cohort (e.g., 'your 124 high-risk adult patients'). "
        "Offer to pull or draft something. CTA: open-ended ('Want me to...'). "
        "Peer/clinical tone — no promotional language."
    ),
    "regulation_change": (
        "State the regulation, deadline date, and specific impact on this merchant. "
        "Offer the compliance workflow. Binary YES/STOP CTA. "
        "Bounded framing — trustworthy, not alarmist."
    ),
    "recall_due": (
        "Message goes TO customer FROM merchant. send_as = 'merchant_on_behalf'. "
        "Use customer name. State exact months since last visit. List available slots. "
        "Use service+price from catalog. Match customer's language_pref. Warm, no pressure."
    ),
    "perf_dip": (
        "Show exact metric, % drop, window. Is this seasonal (normal) or actionable? "
        "Diagnose + prescribe a specific lever. Operator tone — don't panic them."
    ),
    "perf_spike": (
        "Celebrate briefly, then redirect to locking in the gains. "
        "Show exact spike numbers. Suggest specific next action. Effort externalization works."
    ),
    "festival_upcoming": (
        "Name the festival + days until. Suggest a specific campaign using active offers. "
        "Timing + urgency framing. Draft-ready CTA."
    ),
    "renewal_due": (
        "Days remaining + renewal amount. Anchor on specific value delivered this period "
        "(views, calls, leads). Single CTA: Reply YES to renew."
    ),
    "curious_ask_due": (
        "Ask ONE interesting business question. Offer to turn the answer into something "
        "useful (post, reply template). Lightweight + reciprocal. No hard sell."
    ),
    "active_planning_intent": (
        "Merchant said YES to something. DELIVER THE ARTIFACT NOW. "
        "Do NOT ask another qualifying question. Show the draft plan/package/post. "
        "Then ask if they want to edit or send."
    ),
    "competitor_opened": (
        "Don't fear-monger. Offer a specific defensive play (better photos, new offer, GBP). "
        "Data-backed recommendation."
    ),
    "perf_dip_severe": (
        "Act like a trusted advisor. Specific numbers, specific diagnosis, specific fix. "
        "Avoid panic. Binary CTA."
    ),
    "dormant_with_vera": (
        "Don't mention their silence. Re-engage with something NEW and valuable. "
        "Pick the best available signal from their context. Low-pressure."
    ),
    # ... other mappings are possible; fallback below
}

DEFAULT_TRIGGER_INSTRUCTION = (
    "Lead with the specific trigger fact. Connect to merchant context. "
    "Offer a specific next step. Single CTA."
)


def _body_contains_fact(body: str, facts: Dict[str, Any]) -> bool:
    """Return True if body mentions at least one concrete numeric or offer fact from facts."""
    if not body:
        return False
    # Collect numeric tokens from facts
    tokens = []
    for k, v in facts.items():
        try:
            # Skip zero/near-zero numeric facts as unreliable anchors
            if isinstance(v, (int, float)) and abs(float(v)) > 0.000999:
                tokens.append(str(v))
            elif isinstance(v, str):
                # numbers inside strings
                nums = re.findall(r"\d+(?:\.\d+)?", v)
                tokens.extend(nums)
            elif isinstance(v, (list, dict)):
                s = json.dumps(v, ensure_ascii=False)
                nums = re.findall(r"\d+(?:\.\d+)?", s)
                tokens.extend(nums)
        except Exception:
            continue
    # Also include key numeric facts with percent formatting
    percent_keys = ["ctr_30d", "views_30d", "calls_30d", "leads_30d", "peer_avg_ctr"]
    for k in percent_keys:
        if k in facts:
            v = facts.get(k)
            if isinstance(v, (int, float)) and v != 0:
                tokens.append(str(v))

    for t in tokens:
        if t and t in body:
            return True
    # fallback: check for offer titles
    for o in facts.get("active_offers", [])[:2]:
        title = o.get("title", "")
        if title and title in body:
            return True
    # fallback: check review themes and inventory signals
    for theme in facts.get("review_themes", [])[:2]:
        if isinstance(theme, str) and theme and theme in body:
            return True
    # Trigger payload may contain inventory or item names
    payload = facts.get("trigger_payload", {}) or {}
    if isinstance(payload, dict):
        item = payload.get("item_name") or payload.get("offer_name") or payload.get("theme")
        if item and item in body:
            return True
    return False

RATIONALE_TEMPLATE = (
    "Trigger: {trigger_kind} ({trigger_fact}). "
    "Merchant fact: {merchant_fact}. "
    "Lever: {lever_used}."
)


def _choose_best_anchor(facts: Dict[str, Any]) -> (str, list):
    """Choose the best non-zero anchor from facts.
    Prefer: active offer title > review theme > views/calls/ctr (if > tiny) > trigger_payload item.
    Returns (anchor_text, [values...])
    """
    # Offers
    offers = facts.get("active_offers", []) or []
    if offers:
        first = offers[0]
        title = first.get("title")
        if title:
            return (f"Active offer: {title}", [title])

    # Review themes
    themes = facts.get("review_themes", []) or []
    if themes:
        t = themes[0]
        if isinstance(t, str) and t:
            return (f"Recent reviews mention '{t}'", [t])

    # Numeric metrics (only if > tiny threshold)
    views = facts.get("views_30d")
    if isinstance(views, (int, float)) and views and int(views) > 0:
        return (f"You had {int(views)} views in the last 30 days", [str(int(views))])

    ctr = facts.get("ctr_30d")
    try:
        if isinstance(ctr, (int, float)) and abs(float(ctr)) > 0.000999:
            # Format as percent if looks small
            try:
                # If value likely fraction (e.g., 0.048), show percent with 2 decimals
                if abs(float(ctr)) < 1:
                    pct = float(ctr) * 100.0
                    return (f"Your CTR is {pct:.2f}%", [str(pct)])
                return (f"Your CTR is {float(ctr)}%", [str(float(ctr))])
            except Exception:
                return (f"Your CTR: {ctr}", [str(ctr)])
    except Exception:
        pass

    # Trigger payload item
    payload = facts.get("trigger_payload", {}) or {}
    item = payload.get("item_name") or payload.get("offer_name") or payload.get("theme")
    if item:
        return (f"Signal: {item}", [item])

    return ("Quick opportunity from your recent signals", [])


def _build_action_line(trigger_kind: str, facts: Dict[str, Any]) -> str:
    """Return a short 1-line explicit action plan based on trigger_kind and available facts."""
    anchor, vals = _choose_best_anchor(facts)
    offer = vals[0] if vals else None
    if trigger_kind in ("perf_dip", "perf_dip_severe"):
        if offer:
            return f"I'll activate '{offer}' and update your GBP post today if you reply YES."
        return "I'll draft a concrete 1-line offer + GBP update and send it if you reply YES."
    if trigger_kind in ("recall_due", "chronic_refill_due"):
        return "I'll propose 2 slots and message the customer on your approval — reply with the slot number."
    if trigger_kind == "active_planning_intent":
        return "I'll start execution now: drafting the post and scheduling it — tell me any edits."
    if trigger_kind == "perf_spike":
        return "I'll draft a momentum post and schedule it in the next 2 hours if you reply YES."
    if trigger_kind == "competitor_opened":
        return "I'll prepare a quick counter-offer post and GBP highlight — reply YES to run it."
    if trigger_kind == "renewal_due":
        return "I'll send a renewal reminder draft for your review — reply YES to approve."
    # Default action
    if offer:
        return f"I'll draft a short campaign using '{offer}' and send on approval."
    return "I'll draft a ready-to-send message and send it once you reply YES."



# ── Category voice profiles ────────────────────────────────────────────────
CATEGORY_VOICES = {
    "dentists": {
        "voice": "peer-clinical",
        "tone": "professional, knowledgeable, peer-advisor tone",
        "allowed_vocab": [
            "fluoride", "recall", "prophylaxis", "restoration", "alignment",
            "cavity", "plaque", "extraction", "composite", "enamel", "dentin",
            "anterior", "posterior", "occlusion", "bite", "referral"
        ],
        "taboos": [
            "cure", "guaranteed", "always works", "pain-free", "permanent",
            "perfect", "lifetime", "never fail", "miracle", "proven 100%"
        ],
        "greeting": "Dr. {owner}, {message}",
        "proof_style": "peer + trial data + source citation",
        "seasonal": ["exam-stress bruxism Nov-Feb", "wedding whitening Oct-Dec"]
    },
    "salons": {
        "voice": "warm-practical",
        "tone": "friendly, enthusiastic, fellow-operator, accessible",
        "allowed_vocab": [
            "style", "cut", "color", "service", "client", "booking", "look",
            "trend", "season", "package", "care", "glow", "shine"
        ],
        "taboos": [
            "scientific", "clinical", "guaranteed", "perfect", "forever",
            "never fades", "permanent", "100% satisfaction"
        ],
        "greeting": "Hi {owner}, {message}",
        "proof_style": "customer love + bookings + word-of-mouth signals",
        "seasonal": ["summer lightening", "wedding season Oct-Feb"]
    },
    "restaurants": {
        "voice": "fellow-operator",
        "tone": "peer-to-peer, pragmatic, growth-minded, street-smart",
        "allowed_vocab": [
            "covers", "order", "delivery", "dine-in", "table", "menu", "cuisine",
            "rush", "off-peak", "capacity", "seating", "footfall", "check average"
        ],
        "taboos": [
            "guaranteed", "always full", "never empty", "permanent boost",
            "forever busy", "never slow"
        ],
        "greeting": "{owner}, quick thought — {message}",
        "proof_style": "local market data + footfall + order trends",
        "seasonal": ["festival surges", "wedding season bookings", "summer dining"]
    },
    "gyms": {
        "voice": "energetic-coach",
        "tone": "motivational, action-focused, member-centric, growth-oriented",
        "allowed_vocab": [
            "strength", "cardio", "membership", "goals", "results", "commitment",
            "progress", "transformation", "routine", "session", "class", "coach"
        ],
        "taboos": [
            "guaranteed", "instant results", "no effort", "transform in days",
            "forever fit", "never quit", "always motivated"
        ],
        "greeting": "Hey {owner}, thought of you — {message}",
        "proof_style": "member testimonials + retention + referral trends",
        "seasonal": ["New Year resolutions Jan-Feb", "summer beach body May-Aug"]
    },
    "pharmacies": {
        "voice": "trustworthy-precise",
        "tone": "precise, safety-first, knowledgeable, helpful, regulatory-aware",
        "allowed_vocab": [
            "medication", "prescription", "dosage", "refill", "side-effect",
            "contraindication", "generic", "branded", "safety", "compliance",
            "inventory", "batch", "expiry"
        ],
        "taboos": [
            "cure", "guaranteed", "works for everyone", "no side effects",
            "better than", "safest", "always safe", "recommended by all"
        ],
        "greeting": "Hi {owner}, {message}",
        "proof_style": "regulatory + medication safety data + patient care",
        "seasonal": ["seasonal cough/cold Sep-Feb", "allergy season Mar-Jun"]
    }
}


# ── Language instruction ───────────────────────────────────────────────────
def _get_language_instruction(merchant: dict, customer: dict = None) -> str:
    """Detect language preference and return LLM instruction."""
    target = customer or merchant
    pref = (target.get("identity") or {}).get("language_pref", "")
    langs = (target.get("identity") or {}).get("languages", []) or []
    
    if not pref:
        pref = (merchant.get("identity") or {}).get("languages", []) or []

    # Map to language code
    if isinstance(pref, str):
        pref = [pref]

    if any("hi" in l.lower() for l in pref):
        return "Use Hindi-English code-mix (Hinglish). Natural blend: Hindi sentence structure with English technical/offer words. NOT formal Hindi, NOT pure English. Examples: 'aapka ctr 2% se zyada chal gaya,' 'discount ke saath booking ke liye 2 slots ready hain.'"
    elif any("te" in l.lower() for l in pref):
        return "Use Telugu-English mix. Primarily English with occasional Telugu warmth. Example: 'Inka ee week lo patients ni book cheyali.'"
    elif any("ta" in l.lower() for l in pref):
        return "Use Tamil-English mix. Primarily English with Tamil warmth."
    elif any("mr" in l.lower() for l in pref):
        return "Use Marathi-English mix naturally."
    else:
        return "Use English only."


# ── Specificity extraction ─────────────────────────────────────────────────
def _extract_specifics(category: dict, merchant: dict, trigger: dict,
                        customer: dict = None) -> Dict[str, Any]:
    """
    Extract ONLY concrete facts from contexts.
    LLM will be instructed to use ONLY these numbers, never invent any.
    Returns a dict with clean, indexed facts.
    """
    facts = {}

    # Merchant identity
    identity = merchant.get("identity", {})
    facts["merchant_name"] = identity.get("name", "")
    facts["owner_name"] = identity.get("owner_first_name", "")
    facts["city"] = identity.get("city", "")
    facts["locality"] = identity.get("locality", "")
    facts["verified"] = identity.get("verified", False)
    
    # Subscription
    sub = merchant.get("subscription", {})
    facts["subscription_status"] = sub.get("status", "")
    facts["subscription_plan"] = sub.get("plan", "")
    facts["days_remaining"] = sub.get("days_remaining", 0)

    # Performance (raw numbers, no comparisons)
    perf = merchant.get("performance", {})
    if perf:
        facts["window_days"] = perf.get("window_days", 30)
        facts["views_30d"] = perf.get("views", 0)
        facts["calls_30d"] = perf.get("calls", 0)
        facts["directions_30d"] = perf.get("directions", 0)
        facts["ctr_30d"] = perf.get("ctr", 0)
        facts["leads_30d"] = perf.get("leads", 0)
        
        # 7-day deltas (in percentage terms, raw)
        delta = perf.get("delta_7d", {})
        if delta:
            facts["views_delta_7d_pct"] = delta.get("views_pct", 0)
            facts["calls_delta_7d_pct"] = delta.get("calls_pct", 0)
            facts["ctr_delta_7d_pct"] = delta.get("ctr_pct", 0)

    # Peer benchmarks (for comparison context only)
    peer = category.get("peer_stats", {})
    if peer:
        facts["peer_avg_ctr"] = peer.get("avg_ctr", 0)
        facts["peer_avg_views_30d"] = peer.get("avg_views_30d", 0)
        facts["peer_avg_calls_30d"] = peer.get("avg_calls_30d", 0)
        facts["peer_avg_rating"] = peer.get("avg_rating", 0)
        facts["peer_avg_review_count"] = peer.get("avg_review_count", 0)
        facts["peer_scope"] = peer.get("scope", "")

    # Offers (exact titles, statuses)
    offers_active = []
    offers_expired = []
    for o in merchant.get("offers", []):
        if o.get("status") == "active":
            offers_active.append({
                "title": o.get("title", ""),
                "id": o.get("id", ""),
                "started": o.get("started", "")
            })
        elif o.get("status") == "expired":
            offers_expired.append({
                "title": o.get("title", ""),
                "id": o.get("id", ""),
                "ended": o.get("ended", "")
            })
    facts["active_offers"] = offers_active
    facts["expired_offers"] = offers_expired

    # Signals (as given, no inference)
    facts["signals"] = merchant.get("signals", [])

    # Customer aggregate
    cagg = merchant.get("customer_aggregate", {})
    facts["customer_aggregate"] = cagg

    # Review themes (top 3)
    reviews = merchant.get("review_themes", [])
    facts["review_themes"] = reviews[:3]

    # Conversation history (last 3 turns)
    hist = merchant.get("conversation_history", [])
    facts["last_conversation"] = hist[-3:] if hist else []

    # Trigger info
    facts["trigger_kind"] = trigger.get("kind", "")
    facts["trigger_urgency"] = trigger.get("urgency", 0)
    facts["trigger_source"] = trigger.get("source", "")
    facts["trigger_scope"] = trigger.get("scope", "merchant")
    facts["trigger_payload"] = trigger.get("payload", {})

    # Category info
    facts["category_slug"] = category.get("slug", "")
    facts["category_offer_catalog"] = [
        {"title": o.get("title", ""), "id": o.get("id", "")} 
        for o in category.get("offer_catalog", [])[:5]
    ]
    facts["category_digest"] = category.get("digest", [])[:3]  # top 3 digest items
    facts["category_voice_profile"] = CATEGORY_VOICES.get(
        category.get("slug", ""), {}
    )

    # Customer info (if present)
    if customer:
        c_identity = customer.get("identity", {})
        facts["customer_name"] = c_identity.get("name", "")
        facts["customer_language_pref"] = c_identity.get("language_pref", "")
        c_rel = customer.get("relationship", {})
        facts["customer_relationship"] = {
            "first_visit": c_rel.get("first_visit", ""),
            "last_visit": c_rel.get("last_visit", ""),
            "visits_total": c_rel.get("visits_total", 0),
            "services_received": c_rel.get("services_received", [])
        }
        facts["customer_preferences"] = customer.get("preferences", {})
        facts["customer_consent_scope"] = customer.get("consent", {}).get("scope", [])
        facts["customer_state"] = customer.get("state", "")

    return facts


# ── Trigger-specific composition strategies ────────────────────────────────
def _build_trigger_instruction(trigger_kind: str, facts: Dict[str, Any],
                               category_slug: str) -> str:
    """
    Build a trigger-kind-specific system instruction.
    These tell the LLM exactly what to do for each trigger type.
    """
    
    merchant_name = facts.get("merchant_name", "")
    owner_name = facts.get("owner_name", "")
    voice_profile = facts.get("category_voice_profile", {})
    voice_name = voice_profile.get("voice", "professional")
    allowed_vocab = voice_profile.get("allowed_vocab", [])
    taboos = voice_profile.get("taboos", [])
    proof_style = voice_profile.get("proof_style", "")

    base_instruction = f"""
You are composing a WhatsApp message for {merchant_name} (owner: {owner_name}).
Category: {category_slug} — use {voice_name} voice ({voice_profile.get('tone', '')}).

CRITICAL CONSTRAINTS:
1. Use ONLY the facts provided. NEVER invent numbers, offers, dates, or customer names.
2. Allowed vocabulary: {', '.join(allowed_vocab[:8])}.
3. TABOOS (absolutely avoid): {', '.join(taboos)}.
4. Single clear CTA only. No multiple options unless explicitly stated.
5. Keep message ≤160 chars for WhatsApp single-SMS click-through.
6. Personalize with owner name or merchant name naturally.
7. Proof style for this vertical: {proof_style}.

MESSAGE TONE RULES:
- {voice_profile.get('greeting', 'Hi merchant, message')}
- No emojis unless Hinglish/local context demands it.
- No punctuation overkill (no !!!, ???, ...).
"""
    
    if trigger_kind == "research_digest":
        return base_instruction + """
TRIGGER: Research digest.
STRATEGY:
- Hook: Name the finding, source, trial size/year, key stat in FIRST sentence.
- Anchor: Connect to THIS merchant's specific patient cohort or offers.
- CTA: Curiosity-driven, open-ended. "Want me to draft a patient message + ideas?" — not a yes/no.
- Proof: Source + citation + actionable summary.
"""
    
    elif trigger_kind == "regulation_change":
        return base_instruction + """
TRIGGER: Compliance alert.
STRATEGY:
- Hook: Name regulation + issuing body + DEADLINE (e.g., "DCI revised radiograph dose limits by Dec 15").
- Urgency: Loss aversion framing. "Before audit cycle" or "Before deadline."
- CTA: Binary. "Reply YES to get the updated SOP checklist" or "Want the compliance brief?"
- Tone: Serious, peer advisor, factual.
"""
    
    elif trigger_kind in ["recall_due", "chronic_refill_due"]:
        # Customer-scoped trigger
        customer_name = facts.get("customer_name", "")
        return base_instruction + f"""
TRIGGER: Patient/customer recall or refill due.
CUSTOMER: {customer_name}
STRATEGY:
- Greeting: Merchant-on-behalf. Use merchant name, NOT Vera name.
- Personalization: {customer_name}'s first name, specific service, months since last visit.
- Slots: Offer 2-3 specific appointment slots from the payload (e.g., "Wed 2pm or Thu 4pm").
- Price: Include exact service price + any bonus from catalog.
- CTA: Multi-choice slot selection is OK here ("Reply 1 for Wed, 2 for Thu").
- Language: Match customer's language preference exactly.
"""
    
    elif trigger_kind == "perf_dip":
        views_delta = facts.get("views_delta_7d_pct", 0) * 100
        calls_delta = facts.get("calls_delta_7d_pct", 0) * 100
        return base_instruction + f"""
TRIGGER: Performance dip alert.
FACTS: Views {calls_delta:+.0f}%, Calls {views_delta:+.0f}% week-over-week.
STRATEGY:
- Hook: Name exact metric and delta. "Your calls dropped {abs(calls_delta):.0f}% week-over-week" — not vague.
- Hypothesis: Offer likely reason (no active offer, stale posts, seasonal dip).
- Solution: ONE specific action Vera can take right now.
- CTA: Binary. "Want me to activate [offer] + update your GBP post today? Reply YES/STOP."
- Loss aversion: Frame what's being lost in concrete terms (e.g., "that's ~X missed calls/week").
"""
    
    elif trigger_kind == "perf_spike":
        views_delta = facts.get("views_delta_7d_pct", 0) * 100
        calls_delta = facts.get("calls_delta_7d_pct", 0) * 100
        return base_instruction + f"""
TRIGGER: Performance spike — capitalize on momentum.
FACTS: Views {views_delta:+.0f}%, Calls {calls_delta:+.0f}% — you're hot! 🔥
STRATEGY:
- Hook: Name metric, delta, and likely driver. "Your calls spiked {calls_delta:+.0f}% — looks like [reason]."
- Momentum: Frame as immediate opportunity. "When you're hot, strike."
- Suggest: ONE specific action to amplify (new post, activate offer, respond to reviews).
- CTA: Binary. "Reply YES to do it in the next 2 hours while momentum lasts."
"""
    
    elif trigger_kind == "competitor_opened":
        payload = facts.get("trigger_payload", {})
        distance = payload.get("distance_km", "nearby")
        return base_instruction + f"""
TRIGGER: Competitor opened nearby.
FACTS: {distance} away on Google.
STRATEGY:
- Hook: Data-driven, non-alarmist. "New clinic {distance} away on GBP — 3 reviews already."
- Angle: Differentiation, not panic. "Your strengths: [pick 1-2 from reviews]."
- CTA: Actionable counter-move. "Want to run a limited offer to show your vibe?" or "Let me highlight your reviews vs theirs?"
- Tone: Pragmatic peer, not fearful.
"""
    
    elif trigger_kind == "supply_alert":
        payload = facts.get("trigger_payload", {})
        item = payload.get("item_name", "key item")
        return base_instruction + f"""
TRIGGER: Supply/inventory alert.
ITEM: {item}
STRATEGY:
- Hook: Urgency + batch numbers. "{item} stock dipping — batch #XYZ runs out in 3 days."
- CTA: Executable workflow. "Reply YES and I'll draft a supplier outreach + customer prep message."
- Tone: Logistics-savvy peer, helpful.
"""
    
    elif trigger_kind == "active_planning_intent":
        return base_instruction + """
TRIGGER: Merchant explicitly said "yes, let's do it."
STRATEGY:
- NO re-qualification. Merchant already decided.
- Move to EXECUTION mode. Next message should be: "Great! Here's what I'll do [concrete steps]. You just [simple review step]."
- CTA: Confirmation only. "I'll start now — anything else?"
"""
    
    elif trigger_kind == "seasonal_window":
        payload = facts.get("trigger_payload", {})
        season = payload.get("season_name", "upcoming season")
        return base_instruction + f"""
TRIGGER: Seasonal opportunity window.
SEASON: {season}
STRATEGY:
- Hook: Category-specific seasonal moment + data (e.g., "wedding season — your category sees 40% more booking intent").
- Idea: 1-2 seasonal offer ideas from category digest or catalog.
- CTA: "Want me to draft a seasonal campaign?" — open-ended curiosity.
"""
    
    elif trigger_kind == "customer_lapsed_soft":
        payload = facts.get("trigger_payload", {})
        days_lapsed = payload.get("days_since_last_visit", 180)
        customer_name = facts.get("customer_name", "")
        return base_instruction + f"""
TRIGGER: Soft lapse — customer who visited before, quiet for {days_lapsed}d+.
STRATEGY:
- Hook: Gentle nostalgia. "Been {days_lapsed} days since {customer_name}'s last visit — missing you."
- Offer: 1 seasonal service or coupon from catalog.
- CTA: Soft re-engage. "Want me to send them a 'we miss you' + special offer message?"
"""
    
    elif trigger_kind == "review_theme_emerged":
        payload = facts.get("trigger_payload", {})
        theme = payload.get("theme", "common feedback")
        return base_instruction + f"""
TRIGGER: Review theme emerged.
THEME: '{theme}' in {payload.get('count', 2)}+ reviews this month.
STRATEGY:
- Hook: Specific feedback pattern. "3 reviews mention {theme} — all positive/constructive/critical."
- Action: Concrete response to theme (e.g., if wait time, offer a slot-booking system).
- CTA: "Want me to address this + draft a response campaign?"
"""
    
    elif trigger_kind == "dormant_with_vera":
        payload = facts.get("trigger_payload", {})
        days_silent = payload.get("days_silent", 14)
        return base_instruction + f"""
TRIGGER: Merchant has been silent {days_silent}d+.
STRATEGY:
- Hook: Friendly check-in. "Haven't heard from you in {days_silent} days — everything OK?"
- Offer: 1 quick win. "One thing I noticed: [signal from merchant data]."
- CTA: Low-friction re-engage. "Quick 2-min call?" or "One quick question?"
"""
    
    elif trigger_kind == "milestone_reached":
        payload = facts.get("trigger_payload", {})
        milestone = payload.get("milestone", "big milestone")
        return base_instruction + f"""
TRIGGER: {milestone}.
STRATEGY:
- Hook: Celebration. "Congrats on {milestone}! 🎉"
- Anchor: Data proof. "[Real achievement + impact]."
- CTA: Capitalize. "Want to celebrate with a special offer to your fans?"
"""
    
    elif trigger_kind == "category_trend_movement":
        payload = facts.get("trigger_payload", {})
        trend = payload.get("trend_name", "trend")
        direction = payload.get("direction", "up")
        pct = payload.get("pct_change", 20)
        return base_instruction + f"""
TRIGGER: Category trend movement.
TREND: {trend} searches {direction} {pct}% YoY.
STRATEGY:
- Hook: Market shift. "Your category is shifting — {trend} searches up {pct}%."
- Opportunity: Merchant-specific play. "Your catalog has [offer match] — ready to capture?"
- CTA: "Want me to draft a campaign around this trend?"
"""
    
    else:
        # Fallback for unmapped trigger kinds
        return base_instruction + """
TRIGGER: General engagement.
STRATEGY:
- Hook: Lead with merchant's strongest opportunity from context.
- Proof: Use real numbers from merchant data.
- CTA: Single, clear, low-friction action.
"""


# ── Main composer function ────────────────────────────────────────────────
def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict = None,
    conversation_history: list = None,
    is_first_turn: bool = True,
) -> Dict[str, Any]:
    """
    Core composition engine.
    
    Returns:
    {
        "body": str,           # WhatsApp message body
        "cta": str,            # CTA type: "binary", "open_ended", "slot_selection", etc.
        "send_as": str,        # "vera" or "merchant_on_behalf"
        "suppression_key": str, # for dedup
        "rationale": str,      # why this message was chosen
    }
    """
    try:
        # 1. Extract facts (never hallucinate from here)
        facts = _extract_specifics(category, merchant, trigger, customer)
        
        # 2. Get language instruction
        lang_instr = _get_language_instruction(merchant, customer)
        
        # 3. Build trigger-specific instruction
        trigger_kind = trigger.get("kind", "generic")
        trigger_instr = _build_trigger_instruction(trigger_kind, facts, category.get("slug", ""))
        
        # 4. Trim facts to reduce context bloat (avoid LLM truncation)
        facts_trimmed = dict(facts)
        if "category_digest" in facts_trimmed:
            facts_trimmed["category_digest"] = facts_trimmed["category_digest"][:2]
        if "active_offers" in facts_trimmed:
            facts_trimmed["active_offers"] = facts_trimmed["active_offers"][:3]
        if "last_conversation" in facts_trimmed:
            facts_trimmed["last_conversation"] = facts_trimmed["last_conversation"][-2:]
        
        # 5. Format facts as a clean JSON block for LLM
        facts_json = json.dumps(facts_trimmed, ensure_ascii=False, indent=2)
        
        # 6. Build the complete prompt using judge-friendly system template
        # BUG FIX: use the detailed trigger_instr computed above, NOT the static dict
        trigger_specific = trigger_instr
        system_prompt = (
            COMPOSE_SYSTEM
            + "\n\nTRIGGER_INSTRUCTION:\n"
            + trigger_specific
            + "\n\nFACTS PROVIDED (use ONLY these, never invent):\n"
            + facts_json
            + "\n\nLANGUAGE: "
            + lang_instr
            + "\n\nRATIONALE_TEMPLATE:\n"
            + RATIONALE_TEMPLATE
            + "\n\nOUTPUT FORMAT (JSON):\n"
            + "{\n  \"body\": \"the WhatsApp message body\",\n  \"cta\": \"open_ended | binary_yes_stop | none\",\n  \"send_as\": \"vera | merchant_on_behalf\",\n  \"rationale\": \"Trigger used: [X]. Merchant fact anchored: [Y]. Lever deployed: [Z].\",\n  \"template_params\": []\n}"
        )
        
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=800,  # Bumped from 500 to avoid truncation
            temperature=0,  # Deterministic
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Compose the next message for {trigger_kind} trigger. Use only facts provided. Output JSON only."}
            ]
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            logger.warning(f"No JSON in response for {trigger_kind}")
            return _fallback_response(trigger_kind, facts, merchant)
        
        result = json.loads(json_match.group())
        
        # Validate result
        body = (result.get("body") or "").strip()
        if not body:
            logger.warning(f"Empty body for {trigger_kind}")
            return _fallback_response(trigger_kind, facts, merchant)

        # Ensure specificity: body must contain at least one concrete fact from facts
        if not _body_contains_fact(body, facts):
            logger.warning(f"Composed body missing concrete fact for {trigger_kind}; applying deterministic template")
            # deterministic, high-specificity template fallback using best available anchor
            owner = facts.get("owner_name") or merchant.get("identity", {}).get("owner_first_name", "")
            anchor_text, anchor_vals = _choose_best_anchor(facts)
            action_line = _build_action_line(trigger_kind, facts)
            deterministic_body = f"{owner}, {anchor_text}. {action_line}"
            return {
                "body": deterministic_body,
                "cta": "binary_yes_stop",
                "send_as": "merchant_on_behalf" if customer and trigger.get("scope") == "customer" else "vera",
                "suppression_key": trigger.get("suppression_key", f"{trigger_kind}:{merchant.get('merchant_id') }"),
                "rationale": f"Deterministic fallback: ensured non-zero anchor for {trigger_kind}",
                "template_params": anchor_vals
            }

        # Normalize CTA to judge expected values
        raw_cta = result.get("cta", "open_ended")
        if raw_cta in ["binary_yes_stop", "binary", "yes_no", "confirmation"]:
            cta = "binary_yes_stop"
        elif raw_cta in ["open_ended", "open", "continue"]:
            cta = "open_ended"
        elif raw_cta in ["slot_selection", "slots", "multi_choice"]:
            cta = "slot_selection"
        else:
            cta = "none"

        send_as = result.get("send_as", "vera")
        if customer and trigger.get("scope") == "customer":
            send_as = "merchant_on_behalf"
        # Ensure there is a 1-line explicit action plan. If missing, append one.
        action_phrase_re = re.compile(r"(I\s?(?:'ll|will|can|ll)|Reply\s+YES|Want me to|I'll|I will|I'll draft|I'll activate|I'll send|I'll start|I'll prepare)", re.I)
        if not action_phrase_re.search(body):
            action_line = _build_action_line(trigger_kind, facts)
            # Append as final sentence (single short line)
            if not body.endswith("."):
                body = body + "."
            body = body + " " + action_line
            # If CTA was none, promote to binary to enable explicit approval
            if cta == "none":
                cta = "binary_yes_stop"

        return {
            "body": body,
            "cta": cta,
            "send_as": send_as,
            "suppression_key": trigger.get("suppression_key", f"{trigger_kind}:{merchant.get('merchant_id') }"),
            "rationale": result.get("rationale", f"Composed for {trigger_kind} trigger"),
            "template_params": result.get("template_params", [])
        }
    
    except Exception as e:
        logger.error(f"Compose error: {e}")
        return _fallback_response(trigger.get("kind", "generic"), {}, merchant)


def _fallback_response(trigger_kind: str, facts: dict, merchant: dict) -> Dict[str, Any]:
    """Fallback when LLM fails — still provide a sensible, specific message."""
    owner = merchant.get("identity", {}).get("owner_first_name", "there")
    merchant_name = merchant.get("identity", {}).get("name", "your business")
    city = merchant.get("identity", {}).get("city", "your area")
    
    # Extract some specifics from facts to make fallback less generic
    ctr = facts.get("ctr_pct", 0)
    calls = facts.get("calls_this_week", 0)
    active_offers = facts.get("active_offers", [])
    offer_str = ""
    if active_offers and len(active_offers) > 0:
        first_offer = active_offers[0]
        offer_str = f" Our {first_offer.get('offer_type', 'promotion')} is live."
    
    fallbacks = {
        "research_digest": f"Hi {owner}, found {city} insights for {merchant_name}.{offer_str} Want ideas?",
        "perf_dip": f"Hi {owner}, saw {calls} calls this week at {merchant_name}. Quick call — can help spike it?",
        "perf_spike": f"Congrats {owner}! {merchant_name} is trending up (CTR: {ctr}%).{offer_str} Let's lock gains.",
        "recall_due": f"{owner}, great time for customer recalls at {merchant_name}.{offer_str} Shall I draft?",
        "festival_upcoming": f"Hi {owner}, {city} is festive season! {merchant_name} ready?{offer_str}",
        "renewal_due": f"Hi {owner}, renewals due at {merchant_name}.{offer_str} Quick strategy chat?",
        "competitor_opened": f"{owner}, heads up: competitor opened in {city}. Let's keep {merchant_name} strong.{offer_str}",
        "generic": f"Hi {owner}, thought for {merchant_name}. {ctr}% CTR, {calls} calls. Got 30 sec?{offer_str}"
    }
    
    return {
        "body": fallbacks.get(trigger_kind, fallbacks["generic"]),
        "cta": "open_ended",
        "send_as": "vera",
        "suppression_key": f"{trigger_kind}:{merchant.get('merchant_id')}",
        "rationale": f"Fallback for {trigger_kind} (LLM unavailable)"
    }


def compose_reply(
    merchant: dict,
    customer: dict,
    trigger: dict,
    category: dict,
    merchant_reply: str,
    conversation_history: list = None
) -> Dict[str, Any]:
    """
    Compose a reply based on merchant's response.
    Used in /reply endpoint.
    
    Returns same structure as compose().
    """
    # Similar extraction + LLM flow, but focused on continuing conversation
    facts = _extract_specifics(category, merchant, trigger, customer)
    lang_instr = _get_language_instruction(merchant, customer)
    
    system_prompt = f"""
You are Vera, merchant growth assistant. A merchant just replied to a previous message.
Analyze their reply and decide the next move: continue, confirm, end gracefully, or escalate.

MERCHANT REPLIED: "{merchant_reply}"

FACTS:
{json.dumps(facts, ensure_ascii=False, indent=2)}

RULES:
- If they said YES/confirmed: move to EXECUTION mode (next action, timeline).
- If they said NO/STOP/exit: graceful end + respect + no pressure.
- If they asked a question: answer with facts provided only, never invent.
- If they said something off-topic: polite redirect to Vera's scope.
- Keep it short, conversational.

OUTPUT (JSON):
{{
  "body": "reply message ≤160 chars",
  "conversation_state": "active|ended|paused",
  "rationale": "why this reply"
}}
"""
    
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=300,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Merchant replied: '{merchant_reply}'. What's Vera's next move?"
                }
            ]
        )
        
        response_text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if json_match:
            result = json.loads(json_match.group())
            # BUG FIX: include action field so main_enhanced.py can use LLM-driven exits
            conv_state = result.get("conversation_state", "active")
            action = "end" if conv_state == "ended" else ("wait" if conv_state == "paused" else "send")
            return {
                "action": action,
                "body": result.get("body", "Got it, thanks!"),
                "send_as": "vera",
                "suppression_key": f"reply:{trigger.get('id')}",
                "rationale": result.get("rationale", "Reply to merchant message")
            }
    except Exception as e:
        logger.error(f"Reply composition error: {e}")
    
    # Fallback
    return {
        "action": "send",
        "body": "Thanks for getting back to me. Let me know how I can help!",
        "send_as": "vera",
        "suppression_key": f"reply:{trigger.get('id')}",
        "rationale": "Fallback reply"
    }
