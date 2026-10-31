"""Generate the labeled 1000-query orchestrator eval set (deterministic, seeded).

Labels come from the template's category, never from a router, so scoring is not circular.

Usage: python scripts/generate_eval_queries.py [--seed 42] [--out data/orchestrator_eval_1000.json]
"""

import argparse
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

MERCHANTS = [
    "Sri Lakshmi Stores", "Annapoorna Mess", "Gupta Kirana", "Royal Mobiles", "Fresh Mart",
    "Kaveri Medicals", "Balaji Hardware", "Hotel Saravana", "Style Hub Salon", "Patel Dairy",
    "Om Sai Bakery", "City Fuel Point", "Green Leaf Cafe", "Metro Electronics", "Ganesh Tea Stall",
]
CUSTOMERS = ["Ravi Kumar", "Priya Sharma", "Arjun Nair", "Fatima Shaikh", "Suresh Reddy",
             "Anita Das", "Vikram Singh", "Meena Iyer", "Rahul Verma", "Kavya Rao"]
AREAS = ["Bellandur", "HSR Layout", "Koramangala", "Whitefield", "Indiranagar", "Electronic City"]

SINGLE: dict[str, list[str]] = {
    "support": [
        "Resolve support ticket #{tid} for {m}",
        "{m} says the refund of Rs {amt} is not credited yet, please sort it out",
        "Settlement for {m} is delayed by 2 days, the owner is upset",
        "{m} ka payment settle nahi hua, ticket #{tid} check karo",
        "Chargeback dispute on order {tid} at {m}, help resolve it",
        "QR code payments are failing at {m}, they called the helpline",
        "Escalated issue #{tid}: KYC update is stuck for {m}",
        "A customer was charged twice at {m}, need a fix",
        "Why is ticket {tid} for {m} still open?",
        "{m} ki complaint hai ki UPI payment fail ho raha hai",
        "Close the grievance raised by {m} about a wrong MDR deduction",
        "The Soundbox at {m} stopped announcing payments",
    ],
    "collections": [
        "Check loan health and overdue recovery for {m}",
        "{m} missed their EMI of Rs {amt}, what's the next step?",
        "How many days past due is the loan of {m}?",
        "Has {m} turned into an NPA?",
        "Send a repayment reminder to {m}",
        "{m} ka EMI bounce hua hai, recovery plan batao",
        "Pending instalment of Rs {amt} from {m}, follow up on it",
        "Classify the merchant loan of {m}: healthy, watch or at risk?",
        "What is the collection status for {m} this month?",
        "{m} wants to restructure their outstanding dues of Rs {amt}",
        "Prepare the daily collections brief for {area}",
        "Target vs collected for {area} this month, how far behind are we?",
    ],
    "kam": [
        "Show the merchant portfolio metrics for {m}",
        "Any upsell opportunity for key account {m}?",
        "{m} is a key account, prepare a quarterly business review",
        "Pitch the card machine upgrade to {m}",
        "GMV trend for key account {m} over the last 3 months",
        "{m} ke liye cross-sell options kya hain?",
        "Renew the enterprise pricing agreement with {m}",
        "Is {m} at risk of churning to a competitor?",
        "Offer {m} a lower MDR plan to retain their business",
        "Which products can we bundle for our large chain account {m}?",
    ],
    "fse": [
        "Mark daily store visit done for {area} FSE",
        "Send today's daily plan to the FSE in {area}",
        "Mark task {n} as done",
        "Soundbox installation at {m} is completed",
        "Reschedule my visit to {m} to tomorrow",
        "Aaj ke tasks dikhao, {area} beat",
        "Onboarding of {m} finished on site, documents collected",
        "Log evening feedback: {m} wants a second QR standee",
        "Which merchants should I visit first today in {area}?",
        "The field visit at {m} is still pending, add it to tomorrow",
        "Remind the FSEs to carry Aadhaar and ID card tomorrow",
        "done {n}",
    ],
    "copilot": [
        "Give me the team lead overview of {area}",
        "How is my team doing today? Include tickets and visits",
        "Team status: which FSEs missed their targets in {area}?",
        "Summarize today's performance across all my agents",
        "Copilot, what should I discuss in tomorrow's team huddle?",
        "Mere team ka aaj ka status batao",
        "Overview of open tickets, overdue loans and visits for my team",
        "Which team member needs coaching this week?",
        "Leaderboard of FSEs by completed visits in {area}",
    ],
    # Individuals (consumers), not merchants.
    "customer_support": [
        "I was charged twice for my UPI payment of Rs {amt}, raise a dispute",
        "My refund of Rs {amt} for the cancelled order hasn't come back yet",
        "Customer {c} wants to file a chargeback for transaction {tid}",
        "Mera paisa kat gaya par recharge nahi hua",
        "Money got debited from my bank but the payment failed",
        "{c} reports an unauthorised transaction on their wallet",
        "My wallet KYC is stuck, please help",
        "What is the status of my complaint #{tid}?",
        "{c} says Rs {amt} was sent to the wrong UPI ID by mistake",
        "Bill payment of Rs {amt} failed but amount deducted, {c} is asking for it back",
    ],
    "customer_collections": [
        "What is the outstanding amount on my Paytm personal loan?",
        "When is my next EMI due?",
        "{c} wants a statement of their personal loan",
        "Mera loan ka EMI kitna baaki hai?",
        "I want to foreclose my loan, what's the payoff amount?",
        "I missed this month's EMI of Rs {amt}, what happens now?",
        "Customer {c} missed 2 payments on their Postpaid dues, check status",
        "Show my loan repayment schedule",
        "{c} is asking how much interest is left on their Rs {amt} loan",
        "Can {c} get the EMI date moved to the 10th?",
    ],
    "out_of_scope": [
        "Hi",
        "What's the weather in Bengaluru?",
        "Tell me a joke",
        "Who are you?",
        "asdfgh",
        "Book a cab to the office",
        "Thanks!",
        "Namaste",
    ],
}

# Clauses that can be joined into one multi-intent request.
CLAUSES: dict[str, list[str]] = {
    "support": [
        "resolve ticket #{tid}",
        "fix the refund complaint",
        "close the settlement delay complaint",
    ],
    "collections": [
        "check their overdue EMI",
        "send a repayment reminder",
        "check their loan health",
    ],
    "kam": [
        "look for upsell opportunities",
        "pull their portfolio metrics",
        "pitch the EDC upgrade",
    ],
    "fse": [
        "schedule an FSE visit",
        "add a store visit to tomorrow's plan",
        "mark the Soundbox install task done",
    ],
}
JOINERS = [
    "{a} for {m}, and also {b}",
    "For {m}: {a}, then {b}",
    "{a} aur {b} for {m}",
    "Please {b} after you {a} for {m}",
]

# Chat-style noise so the same template yields distinct queries.
PREFIXES = ["", "", "Urgent: ", "Quick one: ", "Note: ", "Hey, "]
SUFFIXES = ["", "", " asap", " please", " today", " - thanks", "!!"]
MAX_ATTEMPTS = 200

EXECUTION_ORDER = ["support", "collections", "kam", "fse"]

# Categories whose label differs from their name.
CATEGORY_LABEL = {
    "customer_support": "support",
    "customer_collections": "collections",
    "out_of_scope": "copilot",
}

# (category, count). Multi-intent categories are "a+b".
PLAN: list[tuple[str, int]] = [
    ("support", 120),
    ("customer_support", 50),
    ("collections", 120),
    ("customer_collections", 50),
    ("kam", 140),
    ("fse", 170),
    ("copilot", 130),
    ("out_of_scope", 40),
    ("support+collections", 60),
    ("support+kam", 30),
    ("support+fse", 30),
    ("collections+fse", 30),
    ("kam+fse", 30),
]


def _fill(template: str, rng: random.Random, m: str, c: str) -> str:
    return template.format(
        m=m,
        c=c,
        tid=rng.randint(1000, 99999),
        amt=rng.choice([450, 1200, 2500, 5000, 12000]),
        area=rng.choice(AREAS),
        n=rng.randint(1, 5),
    )


def _party(category: str, query: str, m: str, rng: random.Random) -> dict[str, str] | None:
    """Customer cases are about a customer; others name a merchant or nobody."""
    if category.startswith("customer_"):
        return {"type": "customer", "id": f"C_{rng.randint(1000, 9999)}"}
    merchant_id = f"M_{rng.randint(100, 999)}"
    return {"type": "merchant", "id": merchant_id} if m in query else None


def make_case(case_id: int, category: str, rng: random.Random) -> dict[str, object]:
    """Build one labeled case for a category such as "support" or "support+collections"."""
    m, c = rng.choice(MERCHANTS), rng.choice(CUSTOMERS)
    if "+" in category:
        a, b = category.split("+")
        joiner = rng.choice(JOINERS)
        query = joiner.format(
            a=_fill(rng.choice(CLAUSES[a]), rng, m, c),
            b=_fill(rng.choice(CLAUSES[b]), rng, m, c),
            m=m,
        )
        expected = [wf for wf in EXECUTION_ORDER if wf in (a, b)]
    else:
        query = _fill(rng.choice(SINGLE[category]), rng, m, c)
        expected = [CATEGORY_LABEL.get(category, category)]
    query = rng.choice(PREFIXES) + query[0].upper() + query[1:] + rng.choice(SUFFIXES)
    return {
        "id": case_id,
        "category": category,
        "query": query,
        "party": _party(category, query, m, rng),
        "expected": expected,
    }


def _unique_case(
    case_id: int, category: str, rng: random.Random, seen: set[str]
) -> dict[str, object]:
    """Draw cases until the query is new; raise if the templates are exhausted."""
    for _ in range(MAX_ATTEMPTS):
        case = make_case(case_id, category, rng)
        key = str(case["query"]).casefold()
        if key not in seen:
            seen.add(key)
            return case
    raise RuntimeError(f"Could not draw a unique {category!r} query; add templates")


def generate(seed: int = 42) -> list[dict[str, object]]:
    """Return all cases, each query unique, in a seeded shuffled order with ids 1..N."""
    rng = random.Random(seed)
    categories = [cat for cat, count in PLAN for _ in range(count)]
    rng.shuffle(categories)
    seen: set[str] = set()
    return [_unique_case(i, cat, rng, seen) for i, cat in enumerate(categories, start=1)]


def main(argv: list[str] | None = None) -> None:
    """CLI: write the eval set to --out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/orchestrator_eval_1000.json"))
    args = parser.parse_args(argv)
    cases = generate(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %d cases to %s", len(cases), args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
