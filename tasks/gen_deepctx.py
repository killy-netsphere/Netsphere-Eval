#!/usr/bin/env python3
"""
gen_deepctx.py -- netterm-eval v2 "deepctx" long-context category.

Design contract (see netterm-eval v2 inviolable rules):
  * NO judge model. Every gold is a bare string that the existing scorer can match
    with answers_match(), and every gold is COMPUTED BY CODE from the emitted
    haystack -- never hand-asserted.
  * Stdlib only, Python 3, fully deterministic under (seed, n_items, ctx_tokens).
  * Exactly one defensible answer per item; the precedence rule that resolves all
    duplicate records is stated verbatim in every prompt.

What this generates
-------------------
A synthetic "campus operations ledger": one record per line, six record types
(PROJECT / STAFF / OFFICE / BUILDING / ACCESS / TASK) plus inert NOTE chatter.
Each item is a STANDALONE prompt containing its own haystack, so depth placement
is controlled per item. At ctx_tokens=128000 one item is ~128K tokens; a 12-item
run therefore costs ~1.5M prompt tokens.

Task kinds (cycled so a small n_items still covers all five):
  hop5          5-hop chain: project -> lead -> office -> building -> access -> custodian.
                THREE hops on the path (PROJECT, STAFF, OFFICE) each carry a second
                record for the same key (superseded by as_of, or an as_of tie broken
                by source=AUDIT); each losing record chains to a DIFFERENT final
                answer, so precedence must be applied repeatedly, not once.
  aggregate     sum / count-distinct / argmax over a small (5-7 record) qualifying
                set. MEMBERSHIP is the hard part: the department is never named --
                it is "the department of the staff member who leads project X", and
                that lead's own STAFF record is superseded (stale record shows the
                twin department), so the filter itself needs a precedence lookup.
                Some qualifying TASK records are superseded too (hours flip,
                billable->closed), so the summed set needs per-record resolution.
  superseded    same key appears twice with different as_of; latest wins (recency trap)
  absent        the requested fact provably does not exist -> gold is "NOT IN CONTEXT"
  contradiction two records tie on as_of; the stated source/rev tie-break resolves them

Every item records meta["depth"] in {0.1, 0.5, 0.9}: the fraction of the haystack at
which the CRITICAL record sits (the record that carries the gold, or -- for absent
items -- the near-miss decoy pair the model is expected to reject). All other needle
placements are in meta["needle_depths"]. Report accuracy by meta["depth"] to expose
lost-in-the-middle.

Anti-shallow-matching: every needle identifier has near-miss twins present in the
haystack (digit transpositions, one-off digits, confusable building names such as
KESTREL/KESTRAL, confusable department names such as Photonics/Phononics), so
grep-like retrieval lands on a wrong record.

Scorer notes (v1 pitfall "8.4 cords" != "8.4"): every prompt states the required
answer form explicitly, and each item also ships "answer_any" with the plausible
harmless decorations (comma grouping, trailing .0, unit suffix).

Public API:
    gen(seed: int, n_items: int, ctx_tokens: int) -> list[dict]
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Dense id/punctuation-heavy text tokenizes worse than prose. Empirical-ish
# constant used only to size the haystack; meta["est_tokens"] reports the estimate.
CHARS_PER_TOKEN = 3.1

KINDS = ["hop5", "aggregate", "superseded", "absent", "contradiction"]
DEPTHS = [0.1, 0.5, 0.9]

FIRST = [
    "Marisol", "Idris", "Yevgenia", "Tobias", "Anneke", "Rafael", "Sunniva", "Kwame",
    "Delphine", "Bartholomew", "Ines", "Osric", "Ludmila", "Hakim", "Rosalind", "Emeka",
    "Solveig", "Casimir", "Beatriz", "Nikolai", "Thandiwe", "Ambrose", "Xiulan", "Fergus",
    "Ottoline", "Piotr", "Mirembe", "Lucien", "Halvard", "Junia", "Otieno", "Freya",
    "Cormac", "Sanaa", "Bjorn", "Perpetua", "Ravi", "Elke", "Malachy", "Noor",
]

LAST = [
    "Verhoeven", "Nakamura", "Oyelaran", "Castellanos", "Bergqvist", "Adeyemi", "Kowalczyk",
    "Fitzmaurice", "Vandenberg", "Okonkwo", "Ravensworth", "Szabo", "Delacroix", "Mbeki",
    "Thorvaldsen", "Anantharaman", "Lindqvist", "Petrosyan", "Marchetti", "Obuya",
    "Haugland", "Kirchner", "Bassinger", "Nwachukwu", "Dalgleish", "Yamashiro",
    "Prendergast", "Volkov", "Achterberg", "Sundstrom", "Balogun", "Ferreira",
    "Winterbourne", "Hadjiev", "Osterman", "Rukavina", "Belanger", "Tsakiris",
]

# Confusable building-name clusters. Every needle building is drawn together with
# its twin so that a fuzzy match lands on the wrong BUILDING record.
BUILDING_TWINS = [
    ("KESTREL", "KESTRAL"), ("HERON", "HERRON"), ("ALDER", "ALDEN"),
    ("MERLIN", "MERLYN"), ("VESPER", "VESPERS"), ("BRAMBLE", "BRAMBEL"),
    ("CINDER", "CINDAR"), ("HOLLOW", "HALLOW"), ("MARROW", "MARLOW"),
    ("SABLE", "SABEL"), ("QUILL", "QUILLE"), ("TERN", "TERNE"),
]

# Departments that only ever appear on needle cohorts, each paired with a
# high-frequency confusable twin that filler staff use constantly.
DEPT_TWINS = [
    ("Photonics", "Phononics"), ("Metrology", "Meteorology"),
    ("Cytometry", "Cytogenetics"), ("Hydrology", "Hydraulics"),
    ("Rheology", "Radiology"), ("Bioinformatics", "Biomechanics"),
]

COMMON_DEPTS = [
    "Logistics", "Procurement", "Facilities", "Payroll", "Networking",
    "Archives", "Custodial", "Fabrication", "Safety", "Legal",
]

TASK_STATUS = ["billable", "internal", "blocked", "closed", "deferred"]

NOTE_TEXT = [
    "quarterly filter replacement completed in the north wing",
    "badge reader firmware rolled back after intermittent read failures",
    "loading dock resurfacing pushed to the next maintenance window",
    "spare cryostat gaskets relocated to the basement cage",
    "fire panel supervisory alarm cleared, no action required",
    "elevator 3 recertified, placard updated",
    "night custodial route shortened by one corridor",
    "roof anemometer replaced under warranty",
    "chilled water loop topped off, no leak found",
    "visitor parking stripe repainting deferred to spring",
    "dust monitoring cartridge swapped on schedule",
    "emergency lighting battery test passed on all floors",
]

WINGS = ["north", "south", "east", "west", "annex", "mezzanine"]
CYCLES = ["FY29", "FY30", "FY31", "FY32"]
PROJ_STATUS = ["active", "paused", "closing", "provisional"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def _date(rng: random.Random, y_lo: int = 2029, y_hi: int = 2032) -> str:
    y = rng.randint(y_lo, y_hi)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _later(rng: random.Random, iso: str) -> str:
    """A date strictly later than `iso`, same coarse era."""
    y, m, d = (int(x) for x in iso.split("-"))
    y += rng.randint(0, 1)
    m += rng.randint(1, 6)
    while m > 12:
        m -= 12
        y += 1
    d = rng.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _earlier(rng: random.Random, iso: str) -> str:
    """A date strictly earlier than `iso` (month/year strictly lower)."""
    y, m, d = (int(x) for x in iso.split("-"))
    y -= rng.randint(0, 1)
    m -= rng.randint(1, 6)
    while m < 1:
        m += 12
        y -= 1
    d = rng.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _mutate_surname(rng: random.Random, s: str) -> str:
    """Produce a near-miss surname: one edit, still pronounceable."""
    cands = []
    for i, ch in enumerate(s):
        if ch.lower() in "aeiou":
            for r in "aeiou":
                if r != ch.lower():
                    cands.append(s[:i] + (r.upper() if ch.isupper() else r) + s[i + 1:])
        else:
            cands.append(s[:i] + ch + s[i:])          # double a consonant
    for i in range(1, len(s) - 1):
        cands.append(s[:i] + s[i + 1] + s[i] + s[i + 2:])  # transpose
    cands = [c for c in cands if c != s]
    return rng.choice(cands)


def _building_variants(key: str, rng: random.Random, k: int) -> List[str]:
    """Near-miss variants of a building key like BLD-KESTREL-3: twin spelling and +/-1 tag."""
    _, name, tag = key.split("-", 2)
    out = []
    for base, twin in BUILDING_TWINS:
        if name in (base, twin):
            other = twin if name == base else base
            out.append(f"BLD-{other}-{tag}")
    try:
        t = int(tag)
        for nt in (t - 1, t + 1):
            if 1 <= nt <= 9:
                out.append(f"BLD-{name}-{nt}")
    except ValueError:
        pass
    seen, uniq = set(), []
    for o in out:
        if o != key and o not in seen:
            seen.add(o)
            uniq.append(o)
    rng.shuffle(uniq)
    return uniq[:k]


def _digit_variants(key: str, rng: random.Random, k: int) -> List[str]:
    """Near-miss variants of an id like STF-30712: transpositions and +/-1 digits.

    Delegates to _building_variants for non-numeric BLD-* keys."""
    prefix, num = key.split("-", 1)
    if not num.isdigit():
        return _building_variants(key, rng, k)
    out = []
    for i in range(len(num) - 1):
        if num[i] != num[i + 1]:
            out.append(f"{prefix}-{num[:i]}{num[i+1]}{num[i]}{num[i+2:]}")
    for i in range(len(num)):
        for delta in (-1, 1):
            nd = (int(num[i]) + delta) % 10
            out.append(f"{prefix}-{num[:i]}{nd}{num[i+1:]}")
    out = [o for o in out if o != key]
    rng.shuffle(out)
    seen, uniq = set(), []
    for o in out:
        if o not in seen:
            seen.add(o)
            uniq.append(o)
    return uniq[:k]


# ---------------------------------------------------------------------------
# Ledger record rendering + parsing (the parser is the gold oracle)
# ---------------------------------------------------------------------------

FIELD_SEP = " :: "


def _render(rtype: str, key: str, fields: List[Tuple[str, str]]) -> str:
    body = FIELD_SEP.join(f"{k}={v}" for k, v in fields)
    return f"{rtype} {key}{FIELD_SEP}{body}"


REC_TYPES = ("PROJECT", "STAFF", "OFFICE", "BUILDING", "ACCESS", "TASK")


def parse_ledger(text: str) -> Dict[str, List[Dict[str, str]]]:
    """Parse an emitted ledger back into records. Used to COMPUTE every gold."""
    recs: Dict[str, List[Dict[str, str]]] = {t: [] for t in REC_TYPES}
    for line in text.splitlines():
        body = line.split("|", 1)[1].strip() if "|" in line else line.strip()
        parts = body.split(FIELD_SEP)
        head = parts[0].split(" ")
        if len(head) != 2 or head[0] not in recs:
            continue
        d: Dict[str, str] = {"_type": head[0], "_key": head[1]}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                d[k] = v
        recs[head[0]].append(d)
    return recs


def _prec_key(r: Dict[str, str]) -> Tuple[str, int, int]:
    srank = {"AUDIT": 2, "SELF-REPORT": 1}.get(r.get("source", ""), 0)
    try:
        rev = int(r.get("rev", "0"))
    except ValueError:
        rev = 0
    return (r.get("as_of", "0000-00-00"), srank, rev)


def resolve(recs: Dict[str, List[Dict[str, str]]], rtype: str, key: str
            ) -> Optional[Dict[str, str]]:
    """Authoritative record for `key`, per the precedence rule stated in the prompt."""
    cands = [r for r in recs[rtype] if r["_key"] == key]
    if not cands:
        return None
    best = max(cands, key=_prec_key)
    return best


def resolution_is_unambiguous(recs: Dict[str, List[Dict[str, str]]]) -> Tuple[bool, str]:
    """No key may have two records tied on the full precedence tuple."""
    for rtype in REC_TYPES:
        by_key: Dict[str, List[Dict[str, str]]] = {}
        for r in recs[rtype]:
            by_key.setdefault(r["_key"], []).append(r)
        for key, group in by_key.items():
            if len(group) == 1:
                continue
            keys = sorted((_prec_key(r) for r in group), reverse=True)
            if keys[0] == keys[1]:
                return False, f"{rtype} {key} has a precedence tie {keys[0]}"
            if keys[0][0] == keys[1][0]:
                # tied on as_of -> both must carry an explicit source for the
                # stated rule to apply
                for r in group:
                    if r.get("as_of") == keys[0][0] and "source" not in r:
                        return False, f"{rtype} {key} ties on as_of without source"
    return True, ""


# ---------------------------------------------------------------------------
# World: id minting with global uniqueness + reserved (never-emitted) keys
# ---------------------------------------------------------------------------

class World:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.taken: Dict[str, set] = {t: set() for t in REC_TYPES}
        self.forbidden: Dict[str, set] = {t: set() for t in REC_TYPES}  # must NEVER be emitted
        self.names: set = set()

    def mint(self, rtype: str, prefix: str, width: int) -> str:
        while True:
            n = self.rng.randint(10 ** (width - 1), 10 ** width - 1)
            key = f"{prefix}-{n}"
            if key not in self.taken[rtype] and key not in self.forbidden[rtype]:
                self.taken[rtype].add(key)
                return key

    def claim(self, rtype: str, key: str) -> bool:
        if key in self.taken[rtype] or key in self.forbidden[rtype]:
            return False
        self.taken[rtype].add(key)
        return True

    def forbid(self, rtype: str, key: str) -> None:
        self.forbidden[rtype].add(key)

    def near_ids(self, rtype: str, key: str, k: int) -> List[str]:
        out = []
        for cand in _digit_variants(key, self.rng, k * 4):
            if len(out) >= k:
                break
            if self.claim(rtype, cand):
                out.append(cand)
        return out

    def person(self) -> str:
        # Bounded rejection sampling: FIRST x LAST is only 1520 names and a
        # 128K haystack consumes ~1400 of them, so an unbounded loop can spin
        # for minutes (or forever) near exhaustion. After 200 misses, extend
        # the space with a middle initial (40 x 26 x 38 = 39520 names).
        for _ in range(200):
            nm = f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}"
            if nm not in self.names:
                self.names.add(nm)
                return nm
        while True:
            nm = (f"{self.rng.choice(FIRST)} {chr(65 + self.rng.randint(0, 25))}. "
                  f"{self.rng.choice(LAST)}")
            if nm not in self.names:
                self.names.add(nm)
                return nm

    def near_person(self, name: str) -> str:
        first, last = name.split(" ", 1)
        for _ in range(40):
            nm = f"{first} {_mutate_surname(self.rng, last)}"
            if nm not in self.names:
                self.names.add(nm)
                return nm
        return self.person()


# ---------------------------------------------------------------------------
# Needle bundle: text + placement fraction + role
# ---------------------------------------------------------------------------

class Needles:
    def __init__(self, rng: random.Random, depth: float):
        self.rng = rng
        self.depth = depth
        self.items: List[Dict[str, Any]] = []
        # scatter pool, deliberately avoiding the critical depth band
        pool = [0.02, 0.07, 0.14, 0.21, 0.29, 0.36, 0.44, 0.52, 0.60, 0.67,
                0.74, 0.81, 0.88, 0.94, 0.98]
        pool = [f for f in pool if abs(f - depth) > 0.06]
        rng.shuffle(pool)
        self._pool = pool
        self._i = 0

    def scatter_frac(self) -> float:
        f = self._pool[self._i % len(self._pool)]
        self._i += 1
        # tiny deterministic jitter so items don't share a template layout
        return min(0.99, max(0.005, f + self.rng.uniform(-0.012, 0.012)))

    def add(self, text: str, role: str, frac: Optional[float] = None,
            critical: bool = False) -> None:
        if frac is None:
            frac = self.depth if critical else self.scatter_frac()
        self.items.append({"text": text, "role": role, "frac": frac,
                           "critical": critical})


# ---------------------------------------------------------------------------
# Filler generation (the distractor haystack)
# ---------------------------------------------------------------------------

def _filler_lines(w: World, rng: random.Random, char_budget: int,
                  twin_dept: str, twin_buildings: List[str]) -> List[str]:
    """Plausible near-miss chatter. Never uses rare/target departments, so the
    size of an aggregation's qualifying set stays bounded as ctx grows."""
    lines: List[str] = []
    used = 0
    staff_pool: List[Tuple[str, str]] = []   # (sid, dept)
    proj_pool: List[str] = []
    office_pool: List[str] = []
    bld_pool: List[str] = list(twin_buildings)
    ac_pool: List[str] = []
    depts = COMMON_DEPTS + [twin_dept] * 3   # twin dept is deliberately frequent

    while used < char_budget:
        roll = rng.random()
        if roll < 0.30 or not staff_pool:
            sid = w.mint("STAFF", "STF", 5)
            dept = rng.choice(depts)
            office = (rng.choice(office_pool) if office_pool and rng.random() < 0.5
                      else w.mint("OFFICE", "RM", 4))
            as_of = _date(rng)
            line = _render("STAFF", sid, [("name", w.person()), ("dept", dept),
                                          ("office", office), ("as_of", as_of)])
            lines.append(line)
            used += len(line) + 12
            staff_pool.append((sid, dept))
            if rng.random() < 0.10:
                # superseded filler staff -- dept only ever moves inside the
                # common pool, never into a rare/target dept
                line2 = _render("STAFF", sid, [
                    ("name", w.person()), ("dept", rng.choice(COMMON_DEPTS)),
                    ("office", w.mint("OFFICE", "RM", 4)), ("as_of", _later(rng, as_of))])
                lines.append(line2)
                used += len(line2) + 12
        elif roll < 0.55:
            tid = w.mint("TASK", "TSK", 5)
            owner = rng.choice(staff_pool)[0]
            proj = rng.choice(proj_pool) if proj_pool else w.mint("PROJECT", "PRJ", 4)
            as_of = _date(rng)
            hrs = rng.randint(2, 96)
            line = _render("TASK", tid, [("project", proj), ("owner", owner),
                                         ("status", rng.choice(TASK_STATUS)),
                                         ("hours", str(hrs)), ("as_of", as_of)])
            lines.append(line)
            used += len(line) + 12
            if rng.random() < 0.07:
                line2 = _render("TASK", tid, [("project", proj), ("owner", owner),
                                              ("status", rng.choice(TASK_STATUS)),
                                              ("hours", str(rng.randint(2, 96))),
                                              ("as_of", _later(rng, as_of))])
                lines.append(line2)
                used += len(line2) + 12
        elif roll < 0.68:
            pid = w.mint("PROJECT", "PRJ", 4)
            lead = rng.choice(staff_pool)[0]
            line = _render("PROJECT", pid, [("lead", lead),
                                            ("cycle", rng.choice(CYCLES)),
                                            ("status", rng.choice(PROJ_STATUS)),
                                            ("as_of", _date(rng))])
            lines.append(line)
            used += len(line) + 12
            proj_pool.append(pid)
        elif roll < 0.80:
            rid = w.mint("OFFICE", "RM", 4)
            bld = (rng.choice(bld_pool) if bld_pool and rng.random() < 0.7
                   else self_mint_building(w, rng))
            line = _render("OFFICE", rid, [("building", bld),
                                           ("floor", str(rng.randint(1, 9))),
                                           ("as_of", _date(rng))])
            lines.append(line)
            used += len(line) + 12
            office_pool.append(rid)
        elif roll < 0.88:
            bld = self_mint_building(w, rng)
            ac = w.mint("ACCESS", "AC", 4)
            line = _render("BUILDING", bld, [("access_code", ac),
                                             ("wing", rng.choice(WINGS)),
                                             ("as_of", _date(rng))])
            lines.append(line)
            used += len(line) + 12
            bld_pool.append(bld)
            ac_pool.append(ac)
        elif roll < 0.94:
            ac = rng.choice(ac_pool) if ac_pool else w.mint("ACCESS", "AC", 4)
            if ac in w.taken["ACCESS"] and any(f" {ac}{FIELD_SEP}" in l for l in lines[-40:]):
                ac = w.mint("ACCESS", "AC", 4)
            if not w.claim("ACCESS", ac):
                ac = w.mint("ACCESS", "AC", 4)
            line = _render("ACCESS", ac, [("custodian", w.person()),
                                          ("renewed", _date(rng)),
                                          ("as_of", _date(rng))])
            lines.append(line)
            used += len(line) + 12
        else:
            line = f"NOTE{FIELD_SEP}{_date(rng)}{FIELD_SEP}{rng.choice(NOTE_TEXT)}"
            lines.append(line)
            used += len(line) + 12
    return lines


def self_mint_building(w: World, rng: random.Random) -> str:
    for _ in range(60):
        base, twin = rng.choice(BUILDING_TWINS)
        cand = f"BLD-{rng.choice((base, twin))}-{rng.randint(1, 9)}"
        if w.claim("BUILDING", cand):
            return cand
    return f"BLD-SPUR-{w.rng.randint(1000, 9999)}"


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def _assemble(filler: List[str], needles: Needles) -> Tuple[str, Dict[str, float]]:
    """Interleave needles into filler at their target fractions.

    COLLISION-FREE BY CONSTRUCTION: each needle claims the nearest FREE slot to
    its target index via a bidirectional expanding search over the full index
    space, so a needle can never overwrite another and the filler list is
    consumed exactly (no IndexError is possible). The critical needle claims
    its slot first, so meta['depth'] stays honest.
    """
    ordered = sorted(needles.items, key=lambda n: (not n["critical"], n["frac"]))
    total = len(filler) + len(ordered)
    assert len(ordered) <= total, "more needles than slots"
    slots: Dict[int, Dict[str, Any]] = {}
    for n in ordered:
        want = int(round(n["frac"] * (total - 1)))
        want = max(0, min(total - 1, want))
        pos: Optional[int] = None
        if want not in slots:
            pos = want
        else:
            for d in range(1, total):
                lo, hi = want - d, want + d
                if lo >= 0 and lo not in slots:
                    pos = lo
                    break
                if hi <= total - 1 and hi not in slots:
                    pos = hi
                    break
        assert pos is not None and pos not in slots, \
            "needle placement failed to find a free slot"
        slots[pos] = n

    # A dropped/overwritten needle is structurally impossible above; assert anyway.
    assert len(slots) == len(ordered), \
        f"needle dropped during placement: {len(slots)} != {len(ordered)}"

    out: List[str] = []
    fi = 0
    actual: Dict[str, float] = {}
    for i in range(total):
        if i in slots:
            n = slots[i]
            out.append(n["text"])
            actual[n["role"]] = round(i / (total - 1), 4)
        else:
            out.append(filler[fi])
            fi += 1
    assert fi == len(filler), "filler under/over-consumed"
    assert len(actual) == len(ordered), "duplicate needle role collapsed the depth map"
    numbered = [f"R{i:06d} | {t}" for i, t in enumerate(out)]
    return "\n".join(numbered), actual


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

PREAMBLE = """You are reading an excerpt from the WESTFIELD CAMPUS OPERATIONS LEDGER.
Each line is one record, in the form:

  R000123 | TYPE KEY :: field=value :: field=value

Record types: PROJECT, STAFF, OFFICE, BUILDING, ACCESS, TASK. NOTE lines are
commentary and carry no fields.

PRECEDENCE RULE (this matters): the same KEY may appear on more than one record.
When it does, exactly one record is authoritative, chosen in this order:
  1. the record with the LATEST as_of date wins;
  2. if the as_of dates are equal, source=AUDIT beats source=SELF-REPORT;
  3. if still tied, the higher rev= number wins.
Position in the file is IRRELEVANT to precedence: a superseded record may appear
anywhere, including after the record that supersedes it.

Identifiers and names in this ledger are deliberately similar to one another.
Match them character by character. A near-match is a wrong match."""

CLOSER = """If -- and only if -- the ledger does not contain what is needed to answer,
reply with exactly: NOT IN CONTEXT

Work through it however you like, then end your reply with a single final line of
exactly this form, containing the answer and nothing else:
FINAL ANSWER: <answer>"""


def _build_prompt(question: str, fmt: str, ledger: str) -> str:
    return (
        f"{PREAMBLE}\n\n"
        f"QUESTION: {question}\n\n{fmt}\n\n{CLOSER}\n\n"
        f"=== LEDGER BEGINS ===\n{ledger}\n=== LEDGER ENDS ===\n\n"
        f"QUESTION (repeated): {question}\n\n{fmt}\n\n"
        f"If the ledger does not contain what is needed, answer exactly NOT IN CONTEXT.\n"
        f"End with exactly one line: FINAL ANSWER: <answer>"
    )


FMT_NAME = ("Answer with the person's full name exactly as it is written in the record "
            "-- first name and surname, nothing else. No identifiers, no titles, no "
            "explanation on that line.")
FMT_INT = ("Answer with a bare integer: digits only. No units, no word 'hours', no "
           "comma or other thousands separator, no decimal point.")
FMT_COUNT = ("Answer with a bare integer count: digits only. No units, no words -- "
             "not '7 projects', just 7. No comma or other thousands separator, no "
             "decimal point.")
FMT_ID = ("Answer with the identifier exactly as it is written in the record (for "
          "example BLD-HERON-4), and nothing else.")
FMT_DATE = ("Answer with the date in YYYY-MM-DD form and nothing else.")


def _int_variants(n: int, unit: Optional[str] = None) -> List[str]:
    """Type-aware harmless decorations. Hour-sum items accept hour-suffixed
    forms; COUNT items accept BARE-integer forms only (an 'N hours' answer to a
    count question is a wrong answer and must NOT score)."""
    out = [str(n), f"{n:,}", f"{n}.0"]
    if unit == "hours":
        out += [f"{n} hours", f"{n} h"]
    return out


def _date_variants(iso: str) -> List[str]:
    """Zero-padded gold plus the non-zero-padded ISO spellings."""
    y, m, d = (int(x) for x in iso.split("-"))
    forms = [iso, f"{y}-{m}-{d}", f"{y}-{m:02d}-{d}", f"{y}-{m}-{d:02d}"]
    return list(dict.fromkeys(forms))


# ---------------------------------------------------------------------------
# Item builders. Each returns (needles, question, fmt, gold_fn, meta_extra)
# gold_fn(recs) computes the gold from the PARSED final ledger.
# ---------------------------------------------------------------------------

def _chain(w: World, rng: random.Random, nd: Needles, *,
           break_at: Optional[str] = None,
           superseded_office: bool = False,
           contradiction_building: bool = False,
           superseded_project: bool = False,
           contested_office: bool = False,
           critical_at: str = "access") -> Dict[str, Any]:
    """Build a project->lead->office->building->access chain plus near-miss twins.

    break_at: record type to omit entirely (its key is forbidden) -> absent needle.
    superseded_project: an OLDER PROJECT record for the same pid hands the lead
        to a decoy staffer whose chain fully resolves to a DIFFERENT final answer.
        Stale record placed EARLY, current LATE (defeats first-mention-wins).
    superseded_office: the lead's STAFF record is superseded; stale record (with
        a fully-resolvable different chain) placed LATE (defeats last-mention-wins).
    contested_office: two OFFICE records tie on as_of; AUDIT (correct building)
        beats the later-placed, higher-rev SELF-REPORT (twin building).
    critical_at: which gold-bearing record carries critical=True ('access' or
        'office') so meta['depth'] tracks a record the answer depends on.
    """
    info: Dict[str, Any] = {}

    custodian = w.person()
    lead_name = w.person()

    pid = w.mint("PROJECT", "PRJ", 4)
    sid = w.mint("STAFF", "STF", 5)
    rid = w.mint("OFFICE", "RM", 4)
    base, twin = rng.choice(BUILDING_TWINS)
    tag = rng.randint(1, 9)
    bld, bld_twin = f"BLD-{base}-{tag}", f"BLD-{twin}-{tag}"
    w.claim("BUILDING", bld)
    w.claim("BUILDING", bld_twin)
    ac = w.mint("ACCESS", "AC", 4)

    info.update(pid=pid, sid=sid, rid=rid, bld=bld, ac=ac,
                custodian=custodian, lead_name=lead_name)

    # --- PROJECT -> lead
    p_as = _date(rng, 2030, 2031)
    nd.add(_render("PROJECT", pid, [("lead", sid), ("cycle", rng.choice(CYCLES)),
                                    ("status", "active"), ("as_of", p_as)]),
           "project",
           frac=(rng.uniform(0.60, 0.92) if superseded_project else None))
    if superseded_project:
        # CONTESTED HOP 1: older PROJECT record (same key, earlier as_of) hands
        # the lead to a decoy staffer. Stale placed EARLY, current LATE, so
        # "first mention wins" fails here.
        d_sid = w.mint("STAFF", "STF", 5)
        d_rid = w.mint("OFFICE", "RM", 4)
        d_bld = self_mint_building(w, rng)
        d_ac = w.mint("ACCESS", "AC", 4)
        nd.add(_render("PROJECT", pid, [("lead", d_sid), ("cycle", rng.choice(CYCLES)),
                                        ("status", "provisional"),
                                        ("as_of", _earlier(rng, p_as))]),
               "project_stale", frac=rng.uniform(0.02, 0.18))
        # ... and the decoy lead's chain fully resolves to a DIFFERENT answer.
        nd.add(_render("STAFF", d_sid, [("name", w.near_person(lead_name)),
                                        ("dept", rng.choice(COMMON_DEPTS)),
                                        ("office", d_rid), ("as_of", _date(rng))]),
               "stale_lead_staff")
        nd.add(_render("OFFICE", d_rid, [("building", d_bld),
                                         ("floor", str(rng.randint(1, 9))),
                                         ("as_of", _date(rng))]),
               "stale_lead_office")
        nd.add(_render("BUILDING", d_bld, [("access_code", d_ac),
                                           ("wing", rng.choice(WINGS)),
                                           ("as_of", _date(rng))]),
               "stale_lead_building")
        nd.add(_render("ACCESS", d_ac, [("custodian", w.near_person(custodian)),
                                        ("renewed", _date(rng, 2029, 2030)),
                                        ("as_of", _date(rng))]),
               "stale_lead_access")
    # near-miss projects pointing at decoy leads
    for npid in w.near_ids("PROJECT", pid, 2):
        nd.add(_render("PROJECT", npid, [("lead", w.mint("STAFF", "STF", 5)),
                                         ("cycle", rng.choice(CYCLES)),
                                         ("status", rng.choice(PROJ_STATUS)),
                                         ("as_of", _date(rng))]),
               f"decoy_project_{npid}")

    # --- STAFF -> office (optionally superseded)
    s_as = _date(rng, 2029, 2030)
    if superseded_office:
        old_rid = w.mint("OFFICE", "RM", 4)
        while True:   # old building must be a NEW key (never collide with bld/bld_twin)
            obase, otwin = rng.choice(BUILDING_TWINS)
            otag = rng.randint(1, 9)
            old_bld = f"BLD-{obase}-{otag}"
            if w.claim("BUILDING", old_bld):
                break
        s_as_new = _later(rng, s_as)
        # CONTESTED HOP 2: OLD record (stale) placed LATE in the file to bait
        # "last mention wins"; current record placed EARLY.
        nd.add(_render("STAFF", sid, [("name", lead_name), ("dept", rng.choice(COMMON_DEPTS)),
                                      ("office", old_rid), ("as_of", s_as)]),
               "staff_stale", frac=rng.uniform(0.72, 0.97))
        nd.add(_render("STAFF", sid, [("name", lead_name), ("dept", rng.choice(COMMON_DEPTS)),
                                      ("office", rid), ("as_of", s_as_new)]),
               "staff_current", frac=rng.uniform(0.03, 0.45))
        nd.add(_render("OFFICE", old_rid, [("building", old_bld),
                                           ("floor", str(rng.randint(1, 9))),
                                           ("as_of", _date(rng))]),
               "office_stale")
        oac = w.mint("ACCESS", "AC", 4)
        nd.add(_render("BUILDING", old_bld, [("access_code", oac),
                                             ("wing", rng.choice(WINGS)),
                                             ("as_of", _date(rng))]),
               "building_stale")
        nd.add(_render("ACCESS", oac, [("custodian", w.near_person(custodian)),
                                       ("renewed", _date(rng, 2029, 2030)),
                                       ("as_of", _date(rng))]),
               "access_stale")
        info["old_bld"] = old_bld
    else:
        nd.add(_render("STAFF", sid, [("name", lead_name), ("dept", rng.choice(COMMON_DEPTS)),
                                      ("office", rid), ("as_of", s_as)]),
               "staff")

    for nsid in w.near_ids("STAFF", sid, 2):
        nd.add(_render("STAFF", nsid, [("name", w.near_person(lead_name)),
                                       ("dept", rng.choice(COMMON_DEPTS)),
                                       ("office", w.mint("OFFICE", "RM", 4)),
                                       ("as_of", _date(rng))]),
               f"decoy_staff_{nsid}")

    # --- OFFICE -> building
    o_as = _date(rng)
    if contested_office:
        # CONTESTED HOP 3: two OFFICE records tie on as_of; AUDIT (-> bld) beats
        # the later-placed, higher-rev SELF-REPORT (-> twin building, whose
        # chain resolves to a DIFFERENT custodian/renewed).
        nd.add(_render("OFFICE", rid, [("building", bld), ("floor", str(rng.randint(1, 9))),
                                       ("as_of", o_as), ("source", "AUDIT"),
                                       ("rev", "3")]),
               "office", critical=(critical_at == "office"))
        nd.add(_render("OFFICE", rid, [("building", bld_twin), ("floor", str(rng.randint(1, 9))),
                                       ("as_of", o_as), ("source", "SELF-REPORT"),
                                       ("rev", "8")]),
               "office_selfreport", frac=rng.uniform(0.55, 0.95))
    else:
        nd.add(_render("OFFICE", rid, [("building", bld), ("floor", str(rng.randint(1, 9))),
                                       ("as_of", o_as)]),
               "office", critical=(critical_at == "office"))
    for nrid in w.near_ids("OFFICE", rid, 2):
        nd.add(_render("OFFICE", nrid, [("building", bld_twin),
                                        ("floor", str(rng.randint(1, 9))),
                                        ("as_of", _date(rng))]),
               f"decoy_office_{nrid}")

    # --- BUILDING -> access_code
    twin_ac = w.mint("ACCESS", "AC", 4)
    nd.add(_render("BUILDING", bld_twin, [("access_code", twin_ac),
                                          ("wing", rng.choice(WINGS)),
                                          ("as_of", _date(rng))]),
           "building_twin")
    nd.add(_render("ACCESS", twin_ac, [("custodian", w.near_person(custodian)),
                                       ("renewed", _date(rng, 2029, 2030)),
                                       ("as_of", _date(rng))]),
           "access_twin")

    if break_at == "BUILDING":
        w.forbid("BUILDING", bld)          # provably no BUILDING record for bld
        info["missing"] = ("BUILDING", bld)
        return info

    b_as = _date(rng)
    if contradiction_building:
        bad_ac = w.mint("ACCESS", "AC", 4)
        # tie on as_of; AUDIT wins per rule. The SELF-REPORT bait has the higher
        # rev AND is placed later in the file, so both naive heuristics lose.
        nd.add(_render("BUILDING", bld, [("access_code", ac), ("wing", rng.choice(WINGS)),
                                         ("as_of", b_as), ("source", "AUDIT"),
                                         ("rev", "2")]),
               "building_audit", critical=False)
        nd.add(_render("BUILDING", bld, [("access_code", bad_ac), ("wing", rng.choice(WINGS)),
                                         ("as_of", b_as), ("source", "SELF-REPORT"),
                                         ("rev", "9")]),
               "building_selfreport",
               frac=min(0.985, max(nd.depth + 0.25, 0.6)))
        nd.add(_render("ACCESS", bad_ac, [("custodian", w.near_person(custodian)),
                                          ("renewed", _date(rng)), ("as_of", _date(rng))]),
               "access_bait")
        info["bad_ac"] = bad_ac
    else:
        nd.add(_render("BUILDING", bld, [("access_code", ac), ("wing", rng.choice(WINGS)),
                                         ("as_of", b_as)]),
               "building")

    if break_at == "ACCESS":
        w.forbid("ACCESS", ac)
        info["missing"] = ("ACCESS", ac)
        return info

    renewed = _date(rng, 2031, 2032)
    info["renewed"] = renewed
    nd.add(_render("ACCESS", ac, [("custodian", custodian), ("renewed", renewed),
                                  ("as_of", _date(rng))]),
           "access", critical=(critical_at == "access"))
    for nac in w.near_ids("ACCESS", ac, 2):
        nd.add(_render("ACCESS", nac, [("custodian", w.near_person(custodian)),
                                       ("renewed", _date(rng)), ("as_of", _date(rng))]),
               f"decoy_access_{nac}")
    return info


def _walk_chain(recs, pid) -> Optional[Dict[str, str]]:
    """Resolve project -> lead -> office -> building -> access, honouring precedence."""
    p = resolve(recs, "PROJECT", pid)
    if not p:
        return None
    s = resolve(recs, "STAFF", p["lead"])
    if not s:
        return None
    o = resolve(recs, "OFFICE", s["office"])
    if not o:
        return None
    b = resolve(recs, "BUILDING", o["building"])
    if not b:
        return None
    a = resolve(recs, "ACCESS", b["access_code"])
    return a


def _chain_finish(recs, level: str, key: str, field: str) -> Optional[str]:
    """Resolve the chain tail from `level` downward, authoritatively."""
    if level == "STAFF":
        s = resolve(recs, "STAFF", key)
        return _chain_finish(recs, "OFFICE", s.get("office", ""), field) if s else None
    if level == "OFFICE":
        o = resolve(recs, "OFFICE", key)
        return _chain_finish(recs, "BUILDING", o.get("building", ""), field) if o else None
    if level == "BUILDING":
        b = resolve(recs, "BUILDING", key)
        return _chain_finish(recs, "ACCESS", b.get("access_code", ""), field) if b else None
    a = resolve(recs, "ACCESS", key)
    return a.get(field) if a else None


def distractor_answers(recs, pid: str, field: str) -> Dict[str, Optional[str]]:
    """For every contested hop on the resolution path, the final answer a model
    reaches if it picks the LOSING record at that hop (and resolves the rest of
    the chain correctly). Every value MUST differ from the gold -- asserted at
    build time and in the validators."""
    out: Dict[str, Optional[str]] = {}
    pcands = [r for r in recs["PROJECT"] if r["_key"] == pid]
    if len(pcands) > 1:
        lose = min(pcands, key=_prec_key)
        out["project_hop"] = _chain_finish(recs, "STAFF", lose.get("lead", ""), field)
    p = resolve(recs, "PROJECT", pid)
    if not p:
        return out
    scands = [r for r in recs["STAFF"] if r["_key"] == p.get("lead", "")]
    if len(scands) > 1:
        lose = min(scands, key=_prec_key)
        out["staff_hop"] = _chain_finish(recs, "OFFICE", lose.get("office", ""), field)
    s = resolve(recs, "STAFF", p.get("lead", ""))
    if not s:
        return out
    ocands = [r for r in recs["OFFICE"] if r["_key"] == s.get("office", "")]
    if len(ocands) > 1:
        lose = min(ocands, key=_prec_key)
        out["office_hop"] = _chain_finish(recs, "BUILDING", lose.get("building", ""), field)
    o = resolve(recs, "OFFICE", s.get("office", ""))
    if not o:
        return out
    bcands = [r for r in recs["BUILDING"] if r["_key"] == o.get("building", "")]
    if len(bcands) > 1:
        lose = min(bcands, key=_prec_key)
        out["building_hop"] = _chain_finish(recs, "ACCESS", lose.get("access_code", ""), field)
    return out


# ---- aggregation cohort ----------------------------------------------------

def _agg_cohort(w: World, rng: random.Random, nd: Needles, sub: str,
                target: str, twin: str) -> Dict[str, Any]:
    """Membership-first aggregation cohort. The department is NEVER named in the
    question: it is 'the department of the staff member who leads project
    <anchor>', and the anchor lead's own STAFF record is superseded (the stale
    record shows the TWIN department, placed LATE in the file), so resolving
    the filter is itself a precedence lookup -- and picking the stale record
    swaps the entire cohort and yields a different answer.

    The qualifying set is kept SMALL (5-7 records for sum): membership, not
    stamina, is the task. Traps:
      - anchor lead's dept superseded twin -> target (stale placed LATE)
      - near-miss anchor PROJECT ids led by twin-dept staff
      - leaver: target -> twin (later as_of)  => must be EXCLUDED
      - joiner: twin -> target (later as_of)  => must be INCLUDED
      - near-miss member ids in the twin dept with fat billable tasks (poison)
      - flip task: hours superseded (authoritative NEW hours placed EARLY,
        stale old hours placed LATE)
      - drop task: billable superseded to closed => excluded from the sum
      - block task: blocked superseded to internal => excluded from the count
    The largest qualifying task is the CRITICAL needle, placed at meta['depth'].
    """
    members: List[Tuple[str, str]] = []       # (sid, name)
    projects: List[str] = [w.mint("PROJECT", "PRJ", 4)
                           for _ in range(rng.randint(4, 5))]

    n_core = 4
    for _ in range(n_core):
        members.append((w.mint("STAFF", "STF", 5), w.person()))

    # ---- anchor: the project whose lead DEFINES the department --------------
    anchor_sid, anchor_nm = members[0]
    anchor_pid = w.mint("PROJECT", "PRJ", 4)
    a_as = _date(rng, 2029, 2030)
    a_as_new = _later(rng, a_as)
    nd.add(_render("STAFF", anchor_sid, [("name", anchor_nm), ("dept", target),
                                         ("office", w.mint("OFFICE", "RM", 4)),
                                         ("as_of", a_as_new)]),
           "anchor_staff_current", frac=rng.uniform(0.03, 0.40))
    # stale record shows the TWIN dept, placed LATE to bait last-mention-wins
    nd.add(_render("STAFF", anchor_sid, [("name", anchor_nm), ("dept", twin),
                                         ("office", w.mint("OFFICE", "RM", 4)),
                                         ("as_of", a_as)]),
           "anchor_staff_stale", frac=rng.uniform(0.70, 0.97))
    nd.add(_render("PROJECT", anchor_pid, [("lead", anchor_sid),
                                           ("cycle", rng.choice(CYCLES)),
                                           ("status", "active"),
                                           ("as_of", _date(rng))]),
           "anchor_project")
    # near-miss anchor project ids, led by TWIN-dept staff (grep poison)
    for npid in w.near_ids("PROJECT", anchor_pid, 2):
        t_sid = w.mint("STAFF", "STF", 5)
        nd.add(_render("STAFF", t_sid, [("name", w.person()), ("dept", twin),
                                        ("office", w.mint("OFFICE", "RM", 4)),
                                        ("as_of", _date(rng))]),
               f"twinstaff_{t_sid}")
        nd.add(_render("PROJECT", npid, [("lead", t_sid),
                                         ("cycle", rng.choice(CYCLES)),
                                         ("status", rng.choice(PROJ_STATUS)),
                                         ("as_of", _date(rng))]),
               f"decoy_anchor_{npid}")

    # ---- remaining core members (current records, target dept) --------------
    for sid, nm in members[1:]:
        nd.add(_render("STAFF", sid, [("name", nm), ("dept", target),
                                      ("office", w.mint("OFFICE", "RM", 4)),
                                      ("as_of", _date(rng))]),
               f"member_{sid}")
    # near-miss twin-dept staffers with confusable names + poison billable tasks
    for sid, nm in members[1:3]:
        nsid = w.near_ids("STAFF", sid, 1)
        if nsid:
            nd.add(_render("STAFF", nsid[0], [("name", w.near_person(nm)),
                                              ("dept", twin),
                                              ("office", w.mint("OFFICE", "RM", 4)),
                                              ("as_of", _date(rng))]),
                   f"nearmember_{nsid[0]}")
            tid = w.mint("TASK", "TSK", 5)
            nd.add(_render("TASK", tid, [("project", rng.choice(projects)),
                                         ("owner", nsid[0]), ("status", "billable"),
                                         ("hours", str(rng.randint(40, 99))),
                                         ("as_of", _date(rng))]),
                   f"poisontask_{tid}")

    # ---- membership recency traps -------------------------------------------
    leaver = w.mint("STAFF", "STF", 5)
    l_nm = w.person()
    l_as = _date(rng, 2029, 2030)
    nd.add(_render("STAFF", leaver, [("name", l_nm), ("dept", target),
                                     ("office", w.mint("OFFICE", "RM", 4)),
                                     ("as_of", l_as)]), "leaver_old")
    nd.add(_render("STAFF", leaver, [("name", l_nm), ("dept", twin),
                                     ("office", w.mint("OFFICE", "RM", 4)),
                                     ("as_of", _later(rng, l_as))]), "leaver_new")
    joiner = w.mint("STAFF", "STF", 5)
    j_nm = w.person()
    j_as = _date(rng, 2029, 2030)
    nd.add(_render("STAFF", joiner, [("name", j_nm), ("dept", twin),
                                     ("office", w.mint("OFFICE", "RM", 4)),
                                     ("as_of", j_as)]), "joiner_old")
    nd.add(_render("STAFF", joiner, [("name", j_nm), ("dept", target),
                                     ("office", w.mint("OFFICE", "RM", 4)),
                                     ("as_of", _later(rng, j_as))]), "joiner_new")

    core_sids = [s for s, _ in members]

    # leaver's billable task: naive membership wrongly includes it
    lt = w.mint("TASK", "TSK", 5)
    nd.add(_render("TASK", lt, [("project", rng.choice(projects)),
                                ("owner", leaver), ("status", "billable"),
                                ("hours", str(rng.randint(40, 90))),
                                ("as_of", _date(rng))]), f"leavertask_{lt}")
    # leaver's blocked task on its own project: poisons the naive count too
    lb = w.mint("TASK", "TSK", 5)
    nd.add(_render("TASK", lb, [("project", rng.choice(projects)),
                                ("owner", leaver), ("status", "blocked"),
                                ("hours", str(rng.randint(5, 40))),
                                ("as_of", _date(rng))]), f"leaverblock_{lb}")
    # joiner's billable task: QUALIFIES
    jt = w.mint("TASK", "TSK", 5)
    j_hrs = rng.randint(11, 77)
    nd.add(_render("TASK", jt, [("project", rng.choice(projects)),
                                ("owner", joiner), ("status", "billable"),
                                ("hours", str(j_hrs)), ("as_of", _date(rng))]),
           f"joinertask_{jt}")

    # ---- superseded TASK records: precedence bites INSIDE the summed set ----
    # flip task: hours superseded; authoritative NEW hours EARLY, stale LATE
    f_tid = w.mint("TASK", "TSK", 5)
    f_owner = rng.choice(core_sids)
    f_as = _date(rng, 2029, 2030)
    h_old = rng.randint(10, 60)
    h_new = h_old + rng.randint(5, 25)
    f_proj = rng.choice(projects)
    nd.add(_render("TASK", f_tid, [("project", f_proj), ("owner", f_owner),
                                   ("status", "billable"), ("hours", str(h_new)),
                                   ("as_of", _later(rng, f_as))]),
           "flip_task_new", frac=rng.uniform(0.03, 0.40))
    nd.add(_render("TASK", f_tid, [("project", f_proj), ("owner", f_owner),
                                   ("status", "billable"), ("hours", str(h_old)),
                                   ("as_of", f_as)]),
           "flip_task_stale", frac=rng.uniform(0.70, 0.97))
    # drop task: was billable, superseded to closed => excluded from the sum
    d_tid = w.mint("TASK", "TSK", 5)
    d_owner = rng.choice(core_sids)
    d_as = _date(rng, 2029, 2030)
    d_proj = rng.choice(projects)
    d_hrs = rng.randint(30, 80)
    nd.add(_render("TASK", d_tid, [("project", d_proj), ("owner", d_owner),
                                   ("status", "billable"), ("hours", str(d_hrs)),
                                   ("as_of", d_as)]), "drop_task_stale")
    nd.add(_render("TASK", d_tid, [("project", d_proj), ("owner", d_owner),
                                   ("status", "closed"), ("hours", str(d_hrs)),
                                   ("as_of", _later(rng, d_as))]), "drop_task_new")
    # block task: was blocked, superseded to internal => excluded from the count
    b_tid = w.mint("TASK", "TSK", 5)
    b_owner = rng.choice(core_sids)
    b_as = _date(rng, 2029, 2030)
    b_proj = rng.choice(projects)
    nd.add(_render("TASK", b_tid, [("project", b_proj), ("owner", b_owner),
                                   ("status", "blocked"), ("hours", str(rng.randint(5, 30))),
                                   ("as_of", b_as)]), "block_task_stale")
    nd.add(_render("TASK", b_tid, [("project", b_proj), ("owner", b_owner),
                                   ("status", "internal"), ("hours", str(rng.randint(5, 30))),
                                   ("as_of", _later(rng, b_as))]), "block_task_new")

    # ---- plain qualifying + chaff tasks: qualifying set totals 5-7 ----------
    q_total = rng.randint(5, 7)   # critical + joiner + flip + (q_total-3) plain
    for _ in range(q_total - 3):
        tid = w.mint("TASK", "TSK", 5)
        nd.add(_render("TASK", tid, [("project", rng.choice(projects)),
                                     ("owner", rng.choice(core_sids)),
                                     ("status", "billable"),
                                     ("hours", str(rng.randint(7, 79))),
                                     ("as_of", _date(rng))]), f"memtask_{tid}")
    # blocked tasks by members (the count metric) and non-billable chaff
    for _ in range(rng.randint(2, 3)):
        tid = w.mint("TASK", "TSK", 5)
        nd.add(_render("TASK", tid, [("project", rng.choice(projects)),
                                     ("owner", rng.choice(core_sids)),
                                     ("status", "blocked"),
                                     ("hours", str(rng.randint(3, 40))),
                                     ("as_of", _date(rng))]), f"blocktask_{tid}")
    for _ in range(rng.randint(2, 4)):
        tid = w.mint("TASK", "TSK", 5)
        nd.add(_render("TASK", tid, [("project", rng.choice(projects)),
                                     ("owner", rng.choice(core_sids)),
                                     ("status", rng.choice(["internal", "closed", "deferred"])),
                                     ("hours", str(rng.randint(3, 60))),
                                     ("as_of", _date(rng))]), f"chafftask_{tid}")

    # the largest qualifying task, at the exact critical depth
    crit_owner = rng.choice(core_sids)
    crit_tid = w.mint("TASK", "TSK", 5)
    crit_hrs = rng.randint(85, 99)
    nd.add(_render("TASK", crit_tid, [("project", rng.choice(projects)),
                                      ("owner", crit_owner), ("status", "billable"),
                                      ("hours", str(crit_hrs)), ("as_of", _date(rng))]),
           "critical_task", critical=True)

    # a task owned by a transposition of a real member's id (resolves elsewhere)
    ghost = w.near_ids("STAFF", crit_owner, 1)
    if ghost:
        nd.add(_render("STAFF", ghost[0], [("name", w.person()),
                                           ("dept", rng.choice(COMMON_DEPTS)),
                                           ("office", w.mint("OFFICE", "RM", 4)),
                                           ("as_of", _date(rng))]),
               f"ghoststaff_{ghost[0]}")
        gt = w.mint("TASK", "TSK", 5)
        nd.add(_render("TASK", gt, [("project", rng.choice(projects)),
                                    ("owner", ghost[0]), ("status", "billable"),
                                    ("hours", str(rng.randint(50, 99))),
                                    ("as_of", _date(rng))]),
               f"ghosttask_{gt}")

    # projects referenced by the cohort
    all_owners = core_sids + [leaver, joiner]
    for pid in projects:
        nd.add(_render("PROJECT", pid, [("lead", rng.choice(all_owners)),
                                        ("cycle", rng.choice(CYCLES)),
                                        ("status", rng.choice(PROJ_STATUS)),
                                        ("as_of", _date(rng))]),
               f"aggproj_{pid}")

    return {"target": target, "twin": twin, "members": members,
            "anchor_pid": anchor_pid, "anchor_sid": anchor_sid,
            "leaver": leaver, "joiner": joiner, "crit_hours": crit_hrs,
            "q_total": q_total}


def _agg_gold_for_dept(recs, target: str, sub: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Aggregate over a GIVEN department, resolving every key authoritatively."""
    staff_keys = {r["_key"] for r in recs["STAFF"]}
    dept_of = {}
    name_of = {}
    for k in staff_keys:
        r = resolve(recs, "STAFF", k)
        dept_of[k] = r.get("dept")
        name_of[k] = r.get("name")
    in_dept = {k for k, d in dept_of.items() if d == target}

    task_keys = {r["_key"] for r in recs["TASK"]}
    rows = []
    for k in task_keys:
        r = resolve(recs, "TASK", k)
        if r.get("owner") in in_dept:
            rows.append(r)

    if sub == "sum":
        qual = [r for r in rows if r.get("status") == "billable"]
        total = sum(int(r["hours"]) for r in qual)
        return str(total), {"n_qualifying": len(qual), "n_in_dept": len(in_dept)}
    if sub == "count":
        qual = [r for r in rows if r.get("status") == "blocked"]
        projs = {r.get("project") for r in qual}
        return str(len(projs)), {"n_qualifying": len(qual), "n_in_dept": len(in_dept)}
    # argmax
    tot: Dict[str, int] = {}
    for r in rows:
        if r.get("status") == "billable":
            tot[r["owner"]] = tot.get(r["owner"], 0) + int(r["hours"])
    if not tot:
        return None, {"n_in_dept": len(in_dept)}
    ranked = sorted(tot.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None, {"tie": True}
    return name_of[ranked[0][0]], {"top_hours": ranked[0][1],
                                   "runner_up": ranked[1][1] if len(ranked) > 1 else None,
                                   "n_in_dept": len(in_dept)}


def _agg_gold(recs, anchor_pid: str, sub: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Compute the aggregation gold from the ANCHOR PROJECT: resolve the
    project's lead, the lead's authoritative department (a superseded lookup),
    then aggregate over that department."""
    p = resolve(recs, "PROJECT", anchor_pid)
    if not p:
        return None, {}
    lead = resolve(recs, "STAFF", p.get("lead", ""))
    if not lead:
        return None, {}
    target = lead.get("dept")
    gold, aux = _agg_gold_for_dept(recs, target, sub)
    aux["resolved_dept"] = target
    return gold, aux


# ---------------------------------------------------------------------------
# Per-item construction
# ---------------------------------------------------------------------------

def _make_item(seed: int, idx: int, kind: str, depth: float, ctx_tokens: int,
               attempt: int) -> Optional[Dict[str, Any]]:
    rng = random.Random((seed * 1_000_003) ^ (idx * 7919) ^ (attempt * 104_729))
    w = World(rng)
    nd = Needles(rng, depth)

    dept_target, dept_twin = rng.choice(DEPT_TWINS)
    agg_sub = ["sum", "count", "max"][(idx // len(KINDS)) % 3]
    variant = idx % 2

    meta_extra: Dict[str, Any] = {}
    gold: Optional[str] = None
    answer_any: List[str] = []

    if kind == "hop5":
        info = _chain(w, rng, nd, superseded_project=True, superseded_office=True,
                      contested_office=True)
        if variant == 0:
            question = ("Who is the custodian of the access code assigned to the building "
                        f"that contains the office of the staff member who leads project {info['pid']}?")
            fmt = FMT_NAME
            gold_fn = lambda recs: (_walk_chain(recs, info["pid"]) or {}).get("custodian")
        else:
            question = ("On what date was the access code renewed for the building that "
                        "contains the office of the staff member who leads project "
                        f"{info['pid']}? (the 'renewed' field of that ACCESS record)")
            fmt = FMT_DATE
            gold_fn = lambda recs: (_walk_chain(recs, info["pid"]) or {}).get("renewed")
        meta_extra["chain"] = [info["pid"], info["sid"], info["rid"], info["bld"], info["ac"]]
        meta_extra["contested_hops"] = ["project", "staff", "office"]
        meta_extra["trap"] = ("three hops on the path (PROJECT recency, STAFF recency, "
                              "OFFICE as_of-tie broken by source=AUDIT) each carry a second "
                              "record for the same key; every losing record chains to a "
                              "DIFFERENT final answer, and the stale records are split "
                              "early/late so neither first- nor last-mention-wins survives")

    elif kind == "superseded":
        # critical_at='office': the gold-bearing OFFICE record (the one whose
        # building= field IS the answer) sits at meta['depth'], so the declared
        # depth controls a record the answer actually depends on.
        info = _chain(w, rng, nd, superseded_office=True, critical_at="office")
        question = ("Using the authoritative record for each key, which building contains "
                    "the office of the staff member who leads project "
                    f"{info['pid']}?")
        fmt = FMT_ID
        def gold_fn(recs, pid=info["pid"]):
            p = resolve(recs, "PROJECT", pid)
            s = resolve(recs, "STAFF", p["lead"]) if p else None
            o = resolve(recs, "OFFICE", s["office"]) if s else None
            return o.get("building") if o else None
        meta_extra["stale_building"] = info.get("old_bld")
        meta_extra["trap"] = "stale STAFF record is placed LATER in the file than the current one"

    elif kind == "contradiction":
        info = _chain(w, rng, nd, contradiction_building=True)
        question = ("Two records give different access codes for the same building. Apply "
                    "the precedence rule stated above, then answer: who is the custodian of "
                    "the access code for the building that contains the office of the staff "
                    f"member who leads project {info['pid']}?")
        fmt = FMT_NAME
        gold_fn = lambda recs: (_walk_chain(recs, info["pid"]) or {}).get("custodian")
        meta_extra["bait_access_code"] = info.get("bad_ac")
        meta_extra["trap"] = ("the SELF-REPORT record has the higher rev= AND appears later "
                              "in the file; the AUDIT record wins")

    elif kind == "absent":
        if variant == 0:
            info = _chain(w, rng, nd, break_at="BUILDING")
            question = ("Which access code covers the building that contains the office of "
                        f"the staff member who leads project {info['pid']}?")
            fmt = ("Answer with the access code exactly as written (for example AC-1234) "
                   "and nothing else.")
            def gold_fn(recs, pid=info["pid"]):
                a = _walk_chain(recs, pid)
                return "NOT IN CONTEXT" if a is None else a.get("_key")
        else:
            info = _chain(w, rng, nd, break_at="ACCESS")
            question = ("Who is the custodian of the access code assigned to the building "
                        "that contains the office of the staff member who leads project "
                        f"{info['pid']}?")
            fmt = FMT_NAME
            def gold_fn(recs, pid=info["pid"]):
                a = _walk_chain(recs, pid)
                return "NOT IN CONTEXT" if a is None else a.get("custodian")
        # the decoy pair (near-miss ids bracketing the missing key) sits at the
        # critical depth: that is what the model must look at and reject.
        mtype, mkey = info["missing"]
        decoys = _digit_variants(mkey, rng, 8)
        placed_keys: List[str] = []
        for dk in decoys:
            if len(placed_keys) >= 2 or not w.claim(mtype, dk):
                continue
            if mtype == "BUILDING":
                text = _render("BUILDING", dk, [("access_code", w.mint("ACCESS", "AC", 4)),
                                                ("wing", rng.choice(WINGS)),
                                                ("as_of", _date(rng))])
            else:
                text = _render("ACCESS", dk, [("custodian", w.person()),
                                              ("renewed", _date(rng)),
                                              ("as_of", _date(rng))])
            nd.add(text, f"decoy_missing_{dk}", critical=(len(placed_keys) == 0))
            placed_keys.append(dk)
        if len(placed_keys) < 2:
            return None      # insist on >=2 near-miss decoys or rebuild
        meta_extra["missing_key"] = f"{mtype} {mkey}"
        meta_extra["near_miss_decoys"] = placed_keys
        meta_extra["trap"] = ("the chain resolves for 3-4 hops and then dead-ends; near-miss "
                              "keys for the missing record DO exist")

    elif kind == "aggregate":
        cohort = _agg_cohort(w, rng, nd, agg_sub, dept_target, dept_twin)
        anchor_pid = cohort["anchor_pid"]
        if agg_sub == "sum":
            question = (f"Sum the hours field over every TASK record whose status is "
                        f"billable and whose owner is a staff member in the same "
                        f"department as the staff member who leads project {anchor_pid}. "
                        f"Use the authoritative record for every key.")
            fmt = FMT_INT
        elif agg_sub == "count":
            question = (f"How many DISTINCT projects have at least one TASK record with "
                        f"status=blocked whose owner is a staff member in the same "
                        f"department as the staff member who leads project {anchor_pid}? "
                        f"Use the authoritative record for every key.")
            fmt = FMT_COUNT
        else:
            question = (f"Among staff members in the same department as the staff member "
                        f"who leads project {anchor_pid}, which one has the greatest "
                        f"total hours across their billable TASK records? Use the "
                        f"authoritative record for every key.")
            fmt = FMT_NAME
        gold_fn = lambda recs: _agg_gold(recs, anchor_pid, agg_sub)[0]
        meta_extra["agg_sub"] = agg_sub
        meta_extra["anchor_project"] = anchor_pid
        meta_extra["anchor_lead"] = cohort["anchor_sid"]
        meta_extra["dept"] = dept_target
        meta_extra["dept_twin"] = dept_twin
        meta_extra["leaver"] = cohort["leaver"]
        meta_extra["joiner"] = cohort["joiner"]
        meta_extra["trap"] = ("the department is never named: it must be resolved from the "
                              "anchor project's lead, whose own STAFF record is superseded "
                              "(the stale record shows the twin department); qualifying "
                              "TASK records are superseded too (hours flip, billable->closed)")
    else:
        raise ValueError(kind)

    # ---- size the haystack and assemble -----------------------------------
    needle_chars = sum(len(n["text"]) + 12 for n in nd.items)
    overhead = len(PREAMBLE) + len(CLOSER) + 2 * len(question) + 2 * len(fmt) + 400
    budget = int(ctx_tokens * CHARS_PER_TOKEN) - needle_chars - overhead
    if budget < 2000:
        budget = 2000
    twin_bldgs = [f"BLD-{t}-{i}" for _, t in BUILDING_TWINS[:4] for i in (1, 2)]
    filler = _filler_lines(w, rng, budget, dept_twin, twin_bldgs)

    ledger, actual = _assemble(filler, nd)

    # ---- compute the gold FROM the emitted ledger --------------------------
    recs = parse_ledger(ledger)
    ok, why = resolution_is_unambiguous(recs)
    if not ok:
        return None
    gold = gold_fn(recs)
    if gold is None:
        return None
    if kind != "absent" and gold == "NOT IN CONTEXT":
        return None

    # ---- distractor-path guarantees (rebuild if any distractor ties gold) ---
    if kind == "hop5":
        field = "custodian" if variant == 0 else "renewed"
        d_ans = distractor_answers(recs, info["pid"], field)
        if len(d_ans) < 3:                      # all three contested hops must exist
            return None
        if any(v is None or v == gold for v in d_ans.values()):
            return None
        meta_extra["distractor_answers"] = d_ans
    elif kind == "superseded":
        p = resolve(recs, "PROJECT", info["pid"])
        scands = [r for r in recs["STAFF"] if r["_key"] == p.get("lead", "")]
        if len(scands) < 2:
            return None
        lose = min(scands, key=_prec_key)
        o_stale = resolve(recs, "OFFICE", lose.get("office", ""))
        stale_bld = o_stale.get("building") if o_stale else None
        if stale_bld is None or stale_bld == gold:
            return None
        meta_extra["distractor_answers"] = {"staff_hop": stale_bld}
    elif kind == "contradiction":
        d_ans = distractor_answers(recs, info["pid"], "custodian")
        if "building_hop" not in d_ans:
            return None
        if d_ans["building_hop"] is None or d_ans["building_hop"] == gold:
            return None
        meta_extra["distractor_answers"] = {"building_hop": d_ans["building_hop"]}
    elif kind == "aggregate":
        gold_v, aux = _agg_gold(recs, anchor_pid, agg_sub)
        if aux.get("resolved_dept") != dept_target:
            return None
        # stale-dept distractor: resolving the anchor lead's dept from the
        # LOSING (older) STAFF record must yield a DIFFERENT final answer
        lead_cands = [r for r in recs["STAFF"] if r["_key"] == cohort["anchor_sid"]]
        if len(lead_cands) < 2:
            return None
        stale_dept = min(lead_cands, key=_prec_key).get("dept")
        if stale_dept == dept_target:
            return None
        stale_ans = _agg_gold_for_dept(recs, stale_dept, agg_sub)[0]
        if stale_ans is None or stale_ans == gold:
            return None
        meta_extra["stale_dept"] = stale_dept
        meta_extra["distractor_answers"] = {"stale_dept": stale_ans}
        if agg_sub == "sum":
            # membership must be the hard part, not stamina: 5-7 summed records
            if not (5 <= aux.get("n_qualifying", 0) <= 7):
                return None
            # the naive unfiltered sum (every billable task, no membership
            # filter) must differ from the gold
            unf = 0
            for k in {r["_key"] for r in recs["TASK"]}:
                t = resolve(recs, "TASK", k)
                if t.get("status") == "billable":
                    unf += int(t.get("hours", "0"))
            if unf == int(gold):
                return None
            meta_extra["naive_unfiltered_sum"] = unf

    if fmt is FMT_INT:
        answer_any = _int_variants(int(gold), unit="hours")
    elif fmt is FMT_COUNT:
        answer_any = _int_variants(int(gold), unit=None)
    elif gold == "NOT IN CONTEXT":
        answer_any = ["NOT IN CONTEXT", "NOT_IN_CONTEXT", "NOT-IN-CONTEXT",
                      "NOTINCONTEXT", "not in context"]
    elif fmt is FMT_DATE:
        answer_any = _date_variants(gold)
    else:
        answer_any = [gold]

    critical = [n for n in nd.items if n["critical"]]
    crit_role = critical[0]["role"] if critical else None
    meta = {
        "kind": kind,
        "depth": depth,
        "depth_mode": ("critical_contributor" if kind == "aggregate" else
                       "decoy_for_missing_record" if kind == "absent" else
                       "gold_bearing_record"),
        "critical_role": crit_role,
        "critical_actual_frac": actual.get(crit_role) if crit_role else None,
        "needle_depths": actual,
        "n_needles": len(nd.items),
        "n_lines": ledger.count("\n") + 1,
        "est_tokens": _est_tokens(ledger),
        "answer_format": ("integer" if fmt in (FMT_INT, FMT_COUNT) else
                          "date" if fmt is FMT_DATE else
                          "identifier" if fmt is FMT_ID else "full_name"),
    }
    meta.update(meta_extra)
    if kind == "aggregate":
        meta.update(_agg_gold(recs, anchor_pid, agg_sub)[1])

    return {
        "id": f"deepctx-{kind}-{idx:03d}",
        "prompt": _build_prompt(question, fmt, ledger),
        "answer": gold,
        "answer_any": answer_any,
        "meta": meta,
        # not part of the public schema, but handy for the validator
        "_ledger": ledger,
        "_question": question,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def gen(seed: int, n_items: int, ctx_tokens: int) -> List[Dict[str, Any]]:
    """Deterministically generate `n_items` deep long-context items.

    Kinds cycle over KINDS and depths over DEPTHS; since len(KINDS)=5 and
    len(DEPTHS)=3 are coprime, 15 consecutive items cover every (kind, depth)
    cell exactly once.
    """
    items: List[Dict[str, Any]] = []
    for i in range(n_items):
        kind = KINDS[i % len(KINDS)]
        depth = DEPTHS[i % len(DEPTHS)]
        item = None
        for attempt in range(24):
            item = _make_item(seed, i, kind, depth, ctx_tokens, attempt)
            if item is not None:
                break
        if item is None:
            raise RuntimeError(f"could not build item {i} ({kind}) after 24 attempts")
        items.append(item)
    return items


if __name__ == "__main__":  # pragma: no cover
    import json
    for it in gen(7, 12, 8000):
        pub = {k: v for k, v in it.items() if not k.startswith("_")}
        print(json.dumps(pub)[:400])
