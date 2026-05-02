"""
conversation_enhanced.py — Advanced conversation state machine.

Tracks multi-turn conversations, detects:
- Auto-reply signatures (12+ patterns + repeat detection)
- Action intent ("yes", "let's do it", "karein")
- Exit intent ("stop", "not interested", "band karo")
- Off-topic escalation ("gst", "loan", "insurance")
- Unanswered message guards

Scoring targets: No burnout (3+ auto-replies = hard end), Clear intent (action vs. exit),
Graceful exit (never harsh), Language-matched (Hindi patterns for Hindi merchants).
"""
import re
import threading
from typing import Dict, List, Optional

_lock = threading.Lock()
_conversations: Dict[str, dict] = {}

# ── Auto-reply detection patterns ──────────────────────────────────────────
_AUTOREPLY_PHRASES = {
    # English
    "automated": 0.95, "auto reply": 0.95, "auto-reply": 0.95, "autoreply": 0.95,
    "i am currently": 0.85, "i'm currently": 0.85, "i will get back": 0.9,
    "will respond shortly": 0.9, "will reach out": 0.9,
    "out of office": 0.95, "on leave": 0.95, "on vacation": 0.95,
    "please contact": 0.7, "for assistance": 0.7, "business hours": 0.8,
    "working hours": 0.8, "thank you for contacting": 0.9,
    "thanks for reaching out": 0.85, "we have received": 0.85,
    "our team will reach": 0.85, "we'll get back": 0.85,
    "thank you for your message": 0.9,
    
    # Hindi
    "aapki jaankari ke liye shukriya": 0.95,
    "aapki madad ke liye shukriya": 0.95,
    "main aapki yeh sabhi baatein": 0.85,
    "hamari team tak pahuncha": 0.85,
    "main ek automated assistant hoon": 0.95,
    "yeh ek automated reply hai": 0.95,
    "aapki message milne ke liye dhanyavaad": 0.9,
    "ham aapko jaldi call karenge": 0.85,
    "kripya hamse yaha sampark karen": 0.8,
    "hame aapka message mila": 0.8,
    "office ke samay": 0.85,
    "kary ghanton mein": 0.85,
}

# ── Action intent patterns (high confidence) ────────────────────────────────
_ACTION_INTENT_PATTERNS = {
    # English
    r"\byes\b": 0.95, r"\byep\b": 0.9, r"\byup\b": 0.9,
    r"\bha\b": 0.95, r"\bhaan\b": 0.95, r"\bdone\b": 0.85,
    r"\blet'?s do it\b": 0.95, r"\bgo ahead\b": 0.9, r"\bproceed\b": 0.85,
    r"\bsure\b": 0.85, r"\bagree\b": 0.85, r"\bapproved?\b": 0.9,
    r"\bconfirm\b": 0.85, r"\bcool\b": 0.7,
    
    # Hinglish
    r"\bkarein\b": 0.9, r"\bkaro\b": 0.85, r"\bchalao\b": 0.9,
    r"\btheek hai\b": 0.9, r"\bsend kar\b": 0.9, r"\bsend karo\b": 0.9,
    r"\baage badho\b": 0.9, r"\bjunna hai\b": 0.85, r"\bjoin karna hai\b": 0.9,
    r"\bmujhe join karna hai\b": 0.95, r"\bstart kar do\b": 0.85,
    r"\bshuru karo\b": 0.85, r"\baaj hi kar do\b": 0.9,
}

# ── Exit intent patterns ───────────────────────────────────────────────────
_EXIT_INTENT_PATTERNS = {
    # English
    r"\bnot interested\b": 0.95, r"\bnot intested\b": 0.95,
    r"\bno thanks\b": 0.9, r"\bno thank you\b": 0.9,
    r"\bstop\b": 0.95, r"\bunsubscribe\b": 0.95, r"\bblock\b": 0.95,
    r"\bdon'?t contact\b": 0.95, r"\bleave me\b": 0.95, r"\bgo away\b": 0.95,
    
    # Hinglish / Hindi
    r"\bnahin chahiye\b": 0.95, r"\bkoi zaroorat nahi\b": 0.95,
    r"\bband karo\b": 0.95, r"\bmat bhejo\b": 0.95, r"\bmatlab nahi\b": 0.95,
    r"\bstop\b": 0.95, r"\bblock kar do\b": 0.95,
}

# ── Off-topic escalation patterns ──────────────────────────────────────────
_OFFTOPIC_PATTERNS = {
    r"\bgst\b": 0.8, r"\bincome tax\b": 0.85, r"\bloan\b": 0.85,
    r"\binsurance\b": 0.8, r"\binvestment\b": 0.8, r"\bsalary\b": 0.8,
    r"\brecruit\b": 0.8, r"\bhiring\b": 0.8,
}


def get_or_create(conv_id: str, merchant_id: str = None, trigger_id: str = None) -> dict:
    """Get or create conversation state."""
    with _lock:
        if conv_id not in _conversations:
            _conversations[conv_id] = {
                "id": conv_id,
                "merchant_id": merchant_id,
                "trigger_id": trigger_id,
                "turns": [],
                "state": "active",  # active | paused | ended
                "auto_reply_count": 0,
                "unanswered_count": 0,
                "last_bot_body": None,
                "merchant_replied": False,
                "intent_detected": None,  # action | exit | question | offtopic
                "within_24h_session": False,
                "created_at": None,
            }
        return _conversations[conv_id]


def record_bot_turn(conv_id: str, body: str):
    """Record bot's outgoing message."""
    with _lock:
        conv = _conversations.get(conv_id)
        if not conv:
            return
        conv["turns"].append({"role": "vera", "body": body})
        conv["last_bot_body"] = body
        conv["unanswered_count"] += 1


def record_merchant_turn(conv_id: str, body: str) -> dict:
    """
    Record merchant/customer reply. Analyze and return:
    {
        "is_auto_reply": bool,
        "intent": "action" | "exit" | "question" | "offtopic" | None,
        "confidence": 0.0-1.0,
    }
    """
    with _lock:
        conv = _conversations.get(conv_id)
        if not conv:
            return {
                "is_auto_reply": False,
                "intent": None,
                "confidence": 0.0,
            }

        conv["turns"].append({"role": "merchant", "body": body})
        conv["unanswered_count"] = 0
        conv["merchant_replied"] = True
        conv["within_24h_session"] = True

        # Multi-signal analysis
        is_auto = _detect_auto_reply(body, conv)
        intent, intent_conf = _detect_intent(body)
        
        if is_auto:
            conv["auto_reply_count"] += 1
        if intent == "exit":
            conv["state"] = "ended"
            conv["intent_detected"] = "exit"
        elif intent == "action":
            conv["intent_detected"] = "action"

        return {
            "is_auto_reply": is_auto,
            "intent": intent,
            "confidence": intent_conf,
        }


def _detect_auto_reply(body: str, conv: dict) -> bool:
    """
    Multi-signal auto-reply detection:
    1. Exact repeat of previous message
    2. Keyword match (confidence weighted)
    3. Generic thank-you pattern (low value add)
    """
    lower = body.lower().strip()
    
    # Signal 1: exact repeat of previous message
    prev_bodies = [t["body"] for t in conv["turns"] if t["role"] == "merchant"]
    if prev_bodies and body.strip() == prev_bodies[-1].strip():
        return True
    
    # Signal 2: keyword scoring
    max_conf = 0.0
    for phrase, conf in _AUTOREPLY_PHRASES.items():
        if phrase in lower:
            max_conf = max(max_conf, conf)
    
    if max_conf >= 0.85:  # High confidence threshold
        return True
    
    # Signal 3: generic thank-you without substance
    generic_thanks = [
        "thanks", "thank you", "shukriya", "dhanyawad",
        "received", "noted", "ok thank", "noted thank"
    ]
    if any(lower.startswith(t) for t in generic_thanks) and len(lower) < 40:
        if "?" not in body and not any(c.isdigit() for c in body):
            return True
    
    return False


def _detect_intent(body: str) -> tuple:
    """
    Detect merchant intent: action, exit, question, offtopic, or None.
    Returns (intent, confidence).
    """
    lower = body.lower().strip()
    
    # Check exit first (high priority)
    max_exit_conf = 0.0
    for pattern, conf in _EXIT_INTENT_PATTERNS.items():
        if re.search(pattern, lower, re.IGNORECASE):
            max_exit_conf = max(max_exit_conf, conf)
    
    if max_exit_conf >= 0.9:
        return "exit", max_exit_conf
    
    # Check action intent
    max_action_conf = 0.0
    for pattern, conf in _ACTION_INTENT_PATTERNS.items():
        if re.search(pattern, lower, re.IGNORECASE):
            max_action_conf = max(max_action_conf, conf)
    
    if max_action_conf >= 0.85:
        return "action", max_action_conf
    
    # Check off-topic escalation
    max_offtopic_conf = 0.0
    for pattern, conf in _OFFTOPIC_PATTERNS.items():
        if re.search(pattern, lower, re.IGNORECASE):
            max_offtopic_conf = max(max_offtopic_conf, conf)
    
    if max_offtopic_conf >= 0.8:
        return "offtopic", max_offtopic_conf
    
    # Otherwise: question (contains ?) or None
    if "?" in body:
        return "question", 0.6
    
    return None, 0.0


def get_conversation(conv_id: str) -> Optional[dict]:
    """Retrieve conversation state."""
    with _lock:
        return _conversations.get(conv_id)


def update_state(conv_id: str, new_state: str):
    """Update conversation state."""
    with _lock:
        conv = _conversations.get(conv_id)
        if conv:
            conv["state"] = new_state


def clear():
    """Clear all conversations (for testing)."""
    with _lock:
        _conversations.clear()


def graceful_exit_message(merchant_name: str, is_auto_reply: bool) -> str:
    if is_auto_reply:
        return (
            f"Koi baat nahi, samajh gayi — looks like this number has an auto-reply active. "
            f"Main {merchant_name} ke account mein notes add kar dungi. "
            f"Jab bhi time mile, feel free to reach out! 🙂"
        )
    return (
        f"Bilkul samajh gayi. Koi zarurat ho toh feel free to reach out — "
        f"main hamesha available hoon. 👋"
    )


def alternate_followup_message(merchant_name: str) -> str:
    """Produce a short different-angle follow-up when an auto-reply is detected."""
    return (
        f"Quick follow — if you'd like, I can draft the message and schedule it for {merchant_name}. "
        f"Reply YES to have it ready, or STOP to cancel."
    )
