"""Generate a fresh FSE/copilot holdout set (deterministic, seeded).

Wording, merchants, areas, names and chat noise are all new: nothing here appears in
data/orchestrator_eval_1000.json or in the prompt examples, so a prompt tuned on the
1000-query set can be checked on queries it has never seen. Labels come from the
template's category.

Usage: python scripts/generate_holdout_queries.py [--seed 7] [--out data/...]
"""

import argparse
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

MERCHANTS = [
    "Shree Ganesh Provisions", "Nandini Milk Parlour", "Udupi Grand", "Tirupati Textiles",
    "Lucky Footwear", "Sagar Sweets", "Vinayaka Xerox", "Coastal Fish Market",
    "Durga Pharma", "Ambika Stationery", "New Delhi Chaat", "Moti Mahal Biryani",
]
AREAS = ["Marathahalli", "BTM Layout", "Hebbal", "Yelahanka", "Banashankari", "Malleshwaram"]
FSE_NAMES = ["Kiran", "Deepa", "Imran", "Lakshmi", "Manoj", "Sowmya", "Prakash", "Nisha"]

# Sub-categories make the boundary visible in results: a manager DOING field work is
# "fse"; a manager ANALYSING the team is "copilot".
TEMPLATES: dict[str, list[str]] = {
    "fse/terse_update": [
        "visit {n} finished",
        "completed task number {n}",
        "{n} ho gaya",
        "task {n} khatam",
        "#{n} done",
        "{n} complete ✅",
        "kaam {n} poora hua",
    ],
    "fse/own_day": [
        "What's on my route for today in {area}?",
        "मेरे आज के काम क्या हैं?",
        "Which shop should I cover after {m}?",
        "I reached {m}, starting the onboarding now",
        "Installed 2 Soundboxes at {m}, update my checklist",
        "{m} was shut when I went, move it to Friday",
        "QR standee placed at {m}, tick it off",
        "End of day note: {m} wants another standee for the second counter",
        "How many stops are left on my list?",
        "Kal {area} mein kitne visits hain mere?",
    ],
    "fse/manager_action": [
        "Push tomorrow's beat plan to {fse} in {area}",
        "Share the visit list with all field executives in {area}",
        "Ask the {area} field team to bring their driving licence tomorrow",
        "Add {m} to {fse}'s visits for Thursday",
        "Move {fse}'s {m} appointment to next week",
        "{fse} ko aaj ki list bhej do",
        "Assign the {m} Soundbox install to {fse}",
    ],
    "copilot/team_analysis": [
        "Who on my team has the lowest visit completion this week?",
        "Compare the {area} and {area2} teams on onboarding numbers",
        "Give me a 3-line summary of how the field team did yesterday",
        "Which executives haven't logged any visits since Monday?",
        "Rank my FSEs by Soundbox installs this month",
        "Is the {area} team on track for the monthly target?",
        "Mere FSEs mein se kaun peeche chal raha hai?",
        "How many tasks did my team complete today versus plan?",
        "Spot any FSE whose completion rate dropped compared to last week",
        "Is {fse} improving or slipping on daily targets?",
    ],
    "copilot/team_management": [
        "Draft talking points for Monday's team meeting",
        "Who should I pair with the new joiner for training?",
        "Weekly review of visits, tickets and collections for my whole team",
        "What should I praise {fse} for in today's stand-up?",
        "Suggest how to split {area} between {fse} and {fse2}",
        "Team ki is hafte ki performance ka summary do",
    ],
}

# (category, count)
PLAN: list[tuple[str, int]] = [
    ("fse/terse_update", 35),
    ("fse/own_day", 35),
    ("fse/manager_action", 30),
    ("copilot/team_analysis", 60),
    ("copilot/team_management", 40),
]

# Noise deliberately different from the 1000-query set.
PREFIXES = ["", "", "", "Pls ", "Boss, ", "FYI ", "Hi, "]
SUFFIXES = ["", "", "", " ?", " pls", " - urgent", " 🙏", " ok"]
MAX_ATTEMPTS = 300


def _fill(template: str, rng: random.Random, m: str) -> str:
    area, area2 = rng.sample(AREAS, 2)
    fse, fse2 = rng.sample(FSE_NAMES, 2)
    return template.format(
        m=m, n=rng.randint(1, 9), area=area, area2=area2, fse=fse, fse2=fse2
    )


def make_case(case_id: int, category: str, rng: random.Random) -> dict[str, object]:
    """Build one labeled case; the label is the part of the category before '/'."""
    m = rng.choice(MERCHANTS)
    query = rng.choice(PREFIXES) + _fill(rng.choice(TEMPLATES[category]), rng, m)
    query += rng.choice(SUFFIXES)
    party = {"type": "merchant", "id": f"M_{rng.randint(1000, 1999)}"} if m in query else None
    return {
        "id": case_id,
        "category": category,
        "query": query,
        "party": party,
        "expected": [category.split("/")[0]],
    }


def generate(seed: int = 7) -> list[dict[str, object]]:
    """Return all cases, each query unique, in a seeded shuffled order with ids 1..N."""
    rng = random.Random(seed)
    categories = [cat for cat, n in PLAN for _ in range(n)]
    rng.shuffle(categories)
    seen: set[str] = set()
    cases = []
    for case_id, category in enumerate(categories, start=1):
        for _ in range(MAX_ATTEMPTS):
            case = make_case(case_id, category, rng)
            key = str(case["query"]).casefold()
            if key not in seen:
                seen.add(key)
                cases.append(case)
                break
        else:
            raise RuntimeError(f"Could not draw a unique {category!r} query; add templates")
    return cases


def main(argv: list[str] | None = None) -> None:
    """CLI: write the holdout set to --out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out", type=Path, default=Path("data/orchestrator_holdout_fse_copilot.json")
    )
    args = parser.parse_args(argv)
    cases = generate(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %d cases to %s", len(cases), args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
