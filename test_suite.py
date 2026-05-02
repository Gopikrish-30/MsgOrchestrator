"""
COMPREHENSIVE END-TO-END TEST SUITE FOR VERA BOT
Tests all 5 endpoints, message composition, auto-reply detection, conversation flow
Evaluates against magicpin challenge rubric
"""

import requests
import json
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("vera-test")

BASE_URL = "http://127.0.0.1:8080"

# Color codes for output
class Color:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# Global test metrics
TEST_RESULTS = {
    "passed": 0,
    "failed": 0,
    "warnings": 0,
    "endpoints_tested": {},
    "message_samples": [],
    "errors": [],
}

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{Color.BOLD}{Color.BLUE}{'='*70}")
    print(f"{title}")
    print(f"{'='*70}{Color.RESET}\n")

def print_pass(msg):
    print(f"{Color.GREEN}✓ {msg}{Color.RESET}")
    TEST_RESULTS["passed"] += 1

def print_fail(msg):
    print(f"{Color.RED}✗ {msg}{Color.RESET}")
    TEST_RESULTS["failed"] += 1
    TEST_RESULTS["errors"].append(msg)

def print_warn(msg):
    print(f"{Color.YELLOW}⚠ {msg}{Color.RESET}")
    TEST_RESULTS["warnings"] += 1

def print_info(msg):
    print(f"  {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: HEALTHZ & METADATA ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

def test_healthz():
    """Test GET /v1/healthz"""
    print_section("TEST 1: HEALTHZ ENDPOINT")
    
    try:
        r = requests.get(f"{BASE_URL}/v1/healthz", timeout=5)
        
        if r.status_code != 200:
            print_fail(f"healthz returned {r.status_code}, expected 200")
            return False
        
        data = r.json()
        
        # Check required fields
        required = ["status", "uptime_seconds", "contexts_loaded"]
        for field in required:
            if field not in data:
                print_fail(f"healthz missing field: {field}")
                return False
        
        if data["status"] != "ok":
            print_fail(f"healthz status is '{data['status']}', expected 'ok'")
            return False
        
        print_pass("healthz endpoint responds with 200 OK")
        print_info(f"Status: {data['status']}")
        print_info(f"Uptime: {data['uptime_seconds']}s")
        print_info(f"Contexts loaded: {json.dumps(data['contexts_loaded'], indent=20)}")
        
        TEST_RESULTS["endpoints_tested"]["healthz"] = "✓"
        return True
        
    except Exception as e:
        print_fail(f"healthz request failed: {e}")
        TEST_RESULTS["endpoints_tested"]["healthz"] = "✗"
        return False

def test_metadata():
    """Test GET /v1/metadata"""
    print_section("TEST 2: METADATA ENDPOINT")
    
    try:
        r = requests.get(f"{BASE_URL}/v1/metadata", timeout=5)
        
        if r.status_code != 200:
            print_fail(f"metadata returned {r.status_code}, expected 200")
            return False
        
        data = r.json()
        
        # Check required fields
        required = ["team_name", "team_members", "model", "approach", "contact_email"]
        for field in required:
            if field not in data:
                print_fail(f"metadata missing field: {field}")
                return False
        
        if not isinstance(data["team_members"], list):
            print_fail(f"team_members should be list, got {type(data['team_members'])}")
            return False
        
        print_pass("metadata endpoint responds with all required fields")
        print_info(f"Team: {data['team_name']}")
        print_info(f"Members: {', '.join(data['team_members'])}")
        print_info(f"Model: {data['model']}")
        print_info(f"Approach: {data['approach'][:100]}...")
        
        TEST_RESULTS["endpoints_tested"]["metadata"] = "✓"
        return True
        
    except Exception as e:
        print_fail(f"metadata request failed: {e}")
        TEST_RESULTS["endpoints_tested"]["metadata"] = "✗"
        return False

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: CONTEXT ENDPOINT (Versioning, Idempotence)
# ─────────────────────────────────────────────────────────────────────────────

def test_context_endpoint():
    """Test POST /v1/context with versioning and idempotence"""
    print_section("TEST 3: CONTEXT ENDPOINT (Versioning & Idempotence)")
    
    all_pass = True
    now = datetime.utcnow().isoformat() + "Z"
    
    # Test 3.1: Push a category context (v1)
    print("\n[3.1] Push category context v1")
    try:
        category_payload = {
            "slug": "test_restaurants",
            "offer_catalog": [
                {"title": "North Indian Thali @ ₹299", "value": "299", "audience": "new_user"}
            ],
            "voice": {
                "tone": "peer_casual",
                "vocab_allowed": ["fresh", "homemade", "authentic"],
                "taboos": ["guaranteed"]
            },
            "peer_stats": {"avg_rating": 4.3, "avg_reviews": 45, "avg_ctr": 0.025},
            "digest": [
                {
                    "id": "d_test_001",
                    "kind": "research",
                    "title": "Cloud kitchens growing 40% YoY in metros",
                    "source": "FnB India 2026",
                    "trial_n": 5000,
                    "summary": "Test summary"
                }
            ],
            "patient_content_library": [],
            "seasonal_beats": [],
            "trend_signals": []
        }
        
        r = requests.post(f"{BASE_URL}/v1/context", json={
            "scope": "category",
            "context_id": "test_restaurants",
            "version": 1,
            "payload": category_payload,
            "delivered_at": now
        }, timeout=5)
        
        if r.status_code != 200:
            print_fail(f"context POST v1 returned {r.status_code}")
            all_pass = False
        else:
            resp = r.json()
            if resp.get("accepted"):
                print_pass("Category context v1 accepted")
            else:
                print_fail(f"Context rejected: {resp}")
                all_pass = False
    
    except Exception as e:
        print_fail(f"Context v1 request failed: {e}")
        all_pass = False
    
    # Test 3.2: Re-post same version (should be idempotent)
    print("\n[3.2] Re-post category context v1 (idempotence test)")
    try:
        r = requests.post(f"{BASE_URL}/v1/context", json={
            "scope": "category",
            "context_id": "test_restaurants",
            "version": 1,
            "payload": category_payload,
            "delivered_at": now
        }, timeout=5)
        
        if r.status_code == 200 and r.json().get("accepted"):
            print_pass("Re-posting same version is idempotent")
        else:
            print_warn(f"Unexpected response on idempotent re-post: {r.json()}")
    
    except Exception as e:
        print_fail(f"Idempotent re-post failed: {e}")
        all_pass = False
    
    # Test 3.3: Push v2 (should replace v1)
    print("\n[3.3] Push category context v2 (version upgrade)")
    try:
        category_payload["offer_catalog"].append(
            {"title": "Biryani @ ₹399", "value": "399", "audience": "all"}
        )
        
        r = requests.post(f"{BASE_URL}/v1/context", json={
            "scope": "category",
            "context_id": "test_restaurants",
            "version": 2,
            "payload": category_payload,
            "delivered_at": now
        }, timeout=5)
        
        if r.status_code == 200 and r.json().get("accepted"):
            print_pass("Version upgrade (v1 → v2) accepted")
        else:
            print_fail(f"Version upgrade rejected: {r.json()}")
            all_pass = False
    
    except Exception as e:
        print_fail(f"Version upgrade failed: {e}")
        all_pass = False
    
    # Test 3.4: Try to push stale version (should be rejected)
    print("\n[3.4] Try to push stale version (v1 after v2)")
    try:
        r = requests.post(f"{BASE_URL}/v1/context", json={
            "scope": "category",
            "context_id": "test_restaurants",
            "version": 1,
            "payload": {"slug": "old_data"},
            "delivered_at": now
        }, timeout=5)
        
        if r.status_code == 409:
            resp = r.json()
            if resp.get("reason") == "stale_version" and resp.get("current_version") == 2:
                print_pass("Stale version correctly rejected with 409")
            else:
                print_fail(f"409 returned but unexpected response: {resp}")
                all_pass = False
        else:
            print_fail(f"Expected 409 for stale version, got {r.status_code}")
            all_pass = False
    
    except Exception as e:
        print_fail(f"Stale version test failed: {e}")
        all_pass = False
    
    # Test 3.5: Invalid scope
    print("\n[3.5] Try to push invalid scope")
    try:
        r = requests.post(f"{BASE_URL}/v1/context", json={
            "scope": "invalid_scope",
            "context_id": "test",
            "version": 1,
            "payload": {},
            "delivered_at": now
        }, timeout=5)
        
        if r.status_code == 400:
            print_pass("Invalid scope correctly rejected with 400")
        else:
            print_warn(f"Expected 400 for invalid scope, got {r.status_code}")
    
    except Exception as e:
        print_fail(f"Invalid scope test failed: {e}")
        all_pass = False
    
    TEST_RESULTS["endpoints_tested"]["context"] = "✓" if all_pass else "✗"
    return all_pass

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: TICK ENDPOINT (Message Composition)
# ─────────────────────────────────────────────────────────────────────────────

def test_tick_endpoint():
    """Test POST /v1/tick with real context and trigger"""
    print_section("TEST 4: TICK ENDPOINT (Message Composition)")
    
    now_str = datetime.utcnow().isoformat() + "Z"
    
    # First, push test merchant context
    print("\n[4.1] Setting up test data (merchant + trigger)")
    try:
        merchant_payload = {
            "merchant_id": "test_m_001",
            "category_slug": "test_restaurants",
            "identity": {
                "name": "Test Restaurant Co.",
                "city": "Mumbai",
                "locality": "Bandra",
                "place_id": "test_place_123",
                "verified": True,
                "languages": ["en", "hi"]
            },
            "subscription": {"status": "active", "plan": "Pro", "days_remaining": 60},
            "performance": {
                "window_days": 30,
                "views": 1500,
                "calls": 25,
                "directions": 60,
                "ctr": 0.025,
                "delta_7d": {"views_pct": 0.15, "calls_pct": 0.05}
            },
            "offers": [
                {"id": "o_1", "title": "North Indian Thali @ ₹299", "status": "active"}
            ],
            "conversation_history": [],
            "customer_aggregate": {"total_unique_ytd": 450, "lapsed_180d_plus": 50, "retention_6mo_pct": 0.42},
            "signals": []
        }
        
        r = requests.post(f"{BASE_URL}/v1/context", json={
            "scope": "merchant",
            "context_id": "test_m_001",
            "version": 1,
            "payload": merchant_payload,
            "delivered_at": now_str
        }, timeout=5)
        
        if r.status_code != 200:
            print_fail(f"Failed to push merchant context: {r.status_code}")
            return False
        
        print_pass("Test merchant context pushed")
        
        # Push trigger
        trigger_payload = {
            "id": "test_trg_001",
            "scope": "merchant",
            "kind": "research_digest",
            "source": "external",
            "merchant_id": "test_m_001",
            "customer_id": None,
            "payload": {
                "category": "test_restaurants",
                "top_item_id": "d_test_001"
            },
            "urgency": 2,
            "suppression_key": "test_research:2026-W18",
            "expires_at": (datetime.utcnow() + timedelta(days=1)).isoformat() + "Z"
        }
        
        r = requests.post(f"{BASE_URL}/v1/context", json={
            "scope": "trigger",
            "context_id": "test_trg_001",
            "version": 1,
            "payload": trigger_payload,
            "delivered_at": now_str
        }, timeout=5)
        
        if r.status_code != 200:
            print_fail(f"Failed to push trigger: {r.status_code}")
            return False
        
        print_pass("Test trigger context pushed")
        
    except Exception as e:
        print_fail(f"Setup failed: {e}")
        return False
    
    # Now test /tick
    print("\n[4.2] Call POST /v1/tick to compose message")
    try:
        r = requests.post(f"{BASE_URL}/v1/tick", json={
            "now": now_str,
            "available_triggers": ["test_trg_001"]
        }, timeout=30)
        
        if r.status_code != 200:
            print_fail(f"tick returned {r.status_code}, expected 200")
            TEST_RESULTS["endpoints_tested"]["tick"] = "✗"
            return False
        
        data = r.json()
        
        if "actions" not in data:
            print_fail("tick response missing 'actions' field")
            return False
        
        actions = data["actions"]
        
        if not isinstance(actions, list):
            print_fail(f"actions should be list, got {type(actions)}")
            return False
        
        print_pass(f"tick endpoint returned 200 OK with {len(actions)} action(s)")
        
        if len(actions) > 0:
            print("\n[4.3] Analyzing composed message quality")
            action = actions[0]
            
            # Check required fields
            required_action_fields = [
                "conversation_id", "merchant_id", "send_as", "body",
                "cta", "suppression_key", "rationale"
            ]
            
            missing = [f for f in required_action_fields if f not in action]
            if missing:
                print_warn(f"Action missing fields: {missing}")
            else:
                print_pass("Action has all required fields")
            
            # Analyze message body
            body = action.get("body", "")
            print_info(f"Message preview: \"{body[:100]}...\"")
            print_info(f"Message length: {len(body)} chars")
            print_info(f"CTA type: {action.get('cta')}")
            print_info(f"Rationale: {action.get('rationale', 'N/A')[:80]}...")
            
            # Check message quality
            if len(body) < 20:
                print_warn("Composed message is very short (< 20 chars)")
            elif len(body) > 500:
                print_warn("Composed message is very long (> 500 chars)")
            else:
                print_pass(f"Message length reasonable ({len(body)} chars)")
            
            # Store sample for analysis
            TEST_RESULTS["message_samples"].append({
                "trigger": "test_trg_001",
                "merchant": "test_m_001",
                "body": body,
                "cta": action.get("cta"),
                "rationale": action.get("rationale")
            })
        
        else:
            print_warn("tick returned no actions (empty list)")
        
        TEST_RESULTS["endpoints_tested"]["tick"] = "✓"
        return True
        
    except Exception as e:
        print_fail(f"tick request failed: {e}")
        TEST_RESULTS["endpoints_tested"]["tick"] = "✗"
        return False

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: REPLY ENDPOINT (Conversation Flow)
# ─────────────────────────────────────────────────────────────────────────────

def test_reply_endpoint():
    """Test POST /v1/reply and conversation flow"""
    print_section("TEST 5: REPLY ENDPOINT (Conversation Flow)")
    
    all_pass = True
    now_str = datetime.utcnow().isoformat() + "Z"
    
    # First get a real conversation from tick
    print("\n[5.1] Setting up conversation via /v1/tick")
    try:
        r = requests.post(f"{BASE_URL}/v1/tick", json={
            "now": now_str,
            "available_triggers": ["test_trg_001"]
        }, timeout=30)
        
        if r.status_code != 200 or not r.json().get("actions"):
            print_warn("Could not get action from tick for reply test")
            return False
        
        action = r.json()["actions"][0]
        conv_id = action.get("conversation_id")
        merchant_id = action.get("merchant_id")
        
        print_pass(f"Got conversation {conv_id} from tick")
        
    except Exception as e:
        print_fail(f"Failed to get conversation: {e}")
        return False
    
    # Test reply with merchant accept
    print("\n[5.2] Test merchant positive reply")
    try:
        r = requests.post(f"{BASE_URL}/v1/reply", json={
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": None,
            "from_role": "merchant",
            "message": "Yes, this looks good. Send me the full report.",
            "received_at": now_str,
            "turn_number": 2
        }, timeout=30)
        
        if r.status_code != 200:
            print_fail(f"reply returned {r.status_code}, expected 200")
            all_pass = False
        else:
            resp = r.json()
            action = resp.get("action")
            
            valid_actions = ["send", "wait", "end"]
            if action not in valid_actions:
                print_fail(f"reply action '{action}' not in {valid_actions}")
                all_pass = False
            else:
                print_pass(f"reply returned valid action: '{action}'")
                
                if action == "send":
                    body = resp.get("body", "")
                    print_info(f"Bot response: \"{body[:80]}...\"")
                    TEST_RESULTS["message_samples"].append({
                        "context": "reply",
                        "merchant_input": "Yes, this looks good",
                        "bot_response": body,
                        "action": action
                    })
                
                elif action == "wait":
                    wait_secs = resp.get("wait_seconds", 0)
                    print_info(f"Bot will wait {wait_secs}s before next message")
                
                elif action == "end":
                    print_info("Bot ending conversation")
    
    except Exception as e:
        print_fail(f"reply request failed: {e}")
        all_pass = False
    
    # Test reply with auto-reply detection
    print("\n[5.3] Test auto-reply detection")
    try:
        # Create new conversation for auto-reply test
        r = requests.post(f"{BASE_URL}/v1/tick", json={
            "now": now_str,
            "available_triggers": ["test_trg_001"]
        }, timeout=30)
        
        if r.status_code == 200 and r.json().get("actions"):
            action = r.json()["actions"][0]
            conv_id_2 = action.get("conversation_id")
            merchant_id_2 = action.get("merchant_id")
            
            # Send auto-reply signature
            r = requests.post(f"{BASE_URL}/v1/reply", json={
                "conversation_id": conv_id_2,
                "merchant_id": merchant_id_2,
                "customer_id": None,
                "from_role": "merchant",
                "message": "Thank you for contacting us. We will get back to you shortly.",
                "received_at": now_str,
                "turn_number": 2
            }, timeout=30)
            
            if r.status_code == 200:
                action = r.json().get("action")
                if action == "end":
                    print_pass("Auto-reply detected - bot gracefully ended conversation")
                elif action == "wait":
                    print_info("Bot chose to wait (potential auto-reply handling)")
                else:
                    print_warn(f"Bot sent action '{action}' to auto-reply (may indicate weak detection)")
    
    except Exception as e:
        print_warn(f"Auto-reply test failed: {e}")
    
    TEST_RESULTS["endpoints_tested"]["reply"] = "✓" if all_pass else "✗"
    return all_pass

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: MESSAGE QUALITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def test_message_quality():
    """Analyze composed messages against challenge rubric"""
    print_section("TEST 6: MESSAGE QUALITY ANALYSIS")
    
    if not TEST_RESULTS["message_samples"]:
        print_warn("No message samples to analyze")
        return False
    
    print(f"\n[6.1] Analyzing {len(TEST_RESULTS['message_samples'])} message samples")
    
    quality_issues = []
    quality_passes = 0
    
    for i, sample in enumerate(TEST_RESULTS["message_samples"], 1):
        body = sample.get("body", "")
        
        print(f"\n  Sample {i}:")
        print(f"    Message: \"{body[:70]}...\"")
        
        checks = {
            "Length 20-500 chars": 20 <= len(body) <= 500,
            "Has merchant name or reference": any(word in body.lower() for word in ["dr.", "test", "restaurant", "mr.", "ms.", "chef"]),
            "Mentions specific service/offer": any(word in body.lower() for word in ["₹", "offer", "service", "discount", "deal"]),
            "Has clear CTA": sample.get("cta") in ["open_ended", "link_click", "call", "reply"],
            "Not generic copy": len(body) > 30 and not body.lower().startswith("hello"),
        }
        
        passed = sum(1 for v in checks.values() if v)
        quality_passes += passed
        
        if passed >= 3:
            print_pass(f"    Quality: {passed}/5 checks passed")
        else:
            quality_issues.append(f"Sample {i}: Only {passed}/5 quality checks")
            print_warn(f"    Quality: {passed}/5 checks passed")
        
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print_info(f"      {status} {check}")
    
    overall_pass_rate = quality_passes / (len(TEST_RESULTS["message_samples"]) * 5)
    print(f"\n[6.2] Overall message quality: {overall_pass_rate*100:.0f}% ({quality_passes}/{len(TEST_RESULTS['message_samples'])*5} checks)")
    
    if overall_pass_rate >= 0.6:
        print_pass("Message quality meets baseline")
    else:
        print_warn("Message quality below optimal threshold")
    
    return overall_pass_rate >= 0.4

# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_final_report():
    """Print comprehensive final report"""
    print_section("FINAL TEST REPORT")
    
    total_tests = TEST_RESULTS["passed"] + TEST_RESULTS["failed"]
    pass_rate = TEST_RESULTS["passed"] / total_tests if total_tests > 0 else 0
    
    print(f"\n{Color.BOLD}ENDPOINT STATUS:{Color.RESET}")
    for endpoint, status in TEST_RESULTS["endpoints_tested"].items():
        icon = "✓" if status == "✓" else "✗"
        print(f"  {icon} /v1/{endpoint:<12} {status}")
    
    print(f"\n{Color.BOLD}TEST SUMMARY:{Color.RESET}")
    print(f"  Total Tests:     {total_tests}")
    print(f"  Passed:          {Color.GREEN}{TEST_RESULTS['passed']}{Color.RESET}")
    print(f"  Failed:          {Color.RED}{TEST_RESULTS['failed']}{Color.RESET}")
    print(f"  Warnings:        {Color.YELLOW}{TEST_RESULTS['warnings']}{Color.RESET}")
    print(f"  Pass Rate:       {pass_rate*100:.1f}%")
    
    if TEST_RESULTS["errors"]:
        print(f"\n{Color.BOLD}ERRORS FOUND ({len(TEST_RESULTS['errors'])}):{Color.RESET}")
        for i, error in enumerate(TEST_RESULTS["errors"][:5], 1):
            print(f"  {i}. {error}")
        if len(TEST_RESULTS["errors"]) > 5:
            print(f"  ... and {len(TEST_RESULTS['errors'])-5} more")
    
    print(f"\n{Color.BOLD}CHALLENGE READINESS:{Color.RESET}")
    if pass_rate >= 0.9:
        print(f"  {Color.GREEN}✓ EXCELLENT - Bot is production-ready{Color.RESET}")
    elif pass_rate >= 0.7:
        print(f"  {Color.YELLOW}⚠ GOOD - Minor fixes recommended{Color.RESET}")
    else:
        print(f"  {Color.RED}✗ NEEDS IMPROVEMENT - Major issues found{Color.RESET}")
    
    print(f"\n{Color.BOLD}SAMPLE MESSAGES ({len(TEST_RESULTS['message_samples'])}):{Color.RESET}")
    for i, sample in enumerate(TEST_RESULTS["message_samples"][:3], 1):
        print(f"\n  [{i}] {sample.get('trigger', 'N/A')}")
        print(f"      Message: \"{sample.get('body', '')[:80]}...\"")
        print(f"      CTA: {sample.get('cta', 'N/A')}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run all tests"""
    print(f"\n{Color.BOLD}{Color.BLUE}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║         VERA BOT - COMPREHENSIVE END-TO-END TEST SUITE          ║")
    print("║            Against magicpin AI Challenge Specification           ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{Color.RESET}\n")
    
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            requests.get(f"{BASE_URL}/v1/healthz", timeout=2)
            print(f"{Color.GREEN}✓ Server is ready{Color.RESET}\n")
            break
        except:
            if attempt == max_attempts - 1:
                print(f"{Color.RED}✗ Server not responding after {max_attempts} attempts{Color.RESET}")
                print(f"Make sure server is running at {BASE_URL}")
                sys.exit(1)
            time.sleep(1)
    
    # Run all tests
    test_healthz()
    test_metadata()
    test_context_endpoint()
    test_tick_endpoint()
    test_reply_endpoint()
    test_message_quality()
    
    # Print final report
    print_final_report()
    
    # Exit with appropriate code
    sys.exit(0 if TEST_RESULTS["failed"] == 0 else 1)

if __name__ == "__main__":
    main()
