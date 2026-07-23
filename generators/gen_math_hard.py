"""gen_math_hard.py -- netterm-eval v2 hard multi-step math generator.

Design rules honoured:
  * stdlib only (random, fractions, math, itertools), Python 3.
  * deterministic: everything derives from random.Random(seed).
  * every gold answer is COMPUTED, never hand-asserted.
  * every prompt states the exact required answer format (bare integer /
    reduced fraction a/b), so the scorer's numeric coercion never sees a
    unit suffix like "8.4 cords".

Public API:
    gen(seed:int, n:int) -> list[dict]  with keys id, prompt, answer,
                                        answer_any (optional list of variants)

Difficulty target: a strong frontier model ~40-75%. Every family is a chain
where a single arithmetic slip cascades into a wrong final integer/fraction.

Scorer-tolerance safety (fatal-defect fix): the authoritative scorer
(netterm-eval harness.answers_match) numerically coerces "a/b" answers and
accepts |got - gold| <= max(1e-9, 1e-6*|gold|).  Every fraction-emitting
family (contfrac, telescope, mobius) therefore REJECTS any instance whose
canonical near-miss wrong answers -- evaluation one term/step short, one step
long, dropped last term, numerator+-1, denominator+-1 -- land within 10x that
tolerance of gold (see _frac_gold_safe).  Rejected instances are regenerated
deterministically from the same rng stream, so same seed -> same items holds.
"""

from fractions import Fraction
from math import gcd, factorial
import itertools
import random

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _egcd(a, b):
    if b == 0:
        return (a, 1, 0)
    g, x, y = _egcd(b, a % b)
    return (g, y, x - (a // b) * y)


def _inv(a, m):
    g, x, _ = _egcd(a % m, m)
    if g != 1:
        raise ValueError("not invertible")
    return x % m


def _crt(residues, moduli):
    """Combined residue and modulus. Moduli must be pairwise coprime."""
    r, m = 0, 1
    for ri, mi in zip(residues, moduli):
        # solve r + m*t == ri (mod mi)
        t = ((ri - r) % mi) * _inv(m % mi, mi) % mi
        r = r + m * t
        m *= mi
    return r % m, m


def _mat_mul(A, B, mod):
    k = len(A)
    return [
        [sum(A[i][t] * B[t][j] for t in range(k)) % mod for j in range(k)]
        for i in range(k)
    ]


def _mat_pow(A, e, mod):
    k = len(A)
    R = [[1 if i == j else 0 for j in range(k)] for i in range(k)]
    while e:
        if e & 1:
            R = _mat_mul(R, A, mod)
        A = _mat_mul(A, A, mod)
        e >>= 1
    return R


def _det_fraction(M):
    """Exact determinant by Fraction Gaussian elimination."""
    n = len(M)
    A = [[Fraction(x) for x in row] for row in M]
    det = Fraction(1)
    for c in range(n):
        piv = None
        for r in range(c, n):
            if A[r][c] != 0:
                piv = r
                break
        if piv is None:
            return 0
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            det = -det
        det *= A[c][c]
        inv = Fraction(1) / A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] * inv
            if f:
                for k in range(c, n):
                    A[r][k] -= f * A[c][k]
    assert det.denominator == 1
    return int(det)


def _fmt_frac(f: Fraction):
    """(gold_string, answer_any_list) for an exact rational answer.

    For negative values the variants include a U+2212 (typographic minus)
    spelling: the prompts forbid it, but belt-and-braces we accept it."""
    if f.denominator == 1:
        s = str(f.numerator)
        variants = [s, "%d/1" % f.numerator]
        if f.numerator < 0:
            variants += [v.replace("-", "−") for v in variants]
        return s, variants
    s = "%d/%d" % (f.numerator, f.denominator)
    variants = [s]
    if f.numerator < 0:
        variants.append(s.replace("-", "−"))
    return s, variants


def _neg_variants(ans):
    """answer_any list for a (possibly negative) integer gold."""
    s = str(ans)
    if ans < 0:
        return [s, s.replace("-", "−")]
    return None


# --------------------------------------------------------------------------
# scorer-tolerance safety for fraction golds
# --------------------------------------------------------------------------


def _sci_round(f, sig):
    """Exact Fraction value of float(f) rounded to `sig` significant digits."""
    if f == 0:
        return Fraction(0)
    mant, exp = ("%.*e" % (sig - 1, float(f))).split("e")
    return Fraction(mant) * Fraction(10) ** int(exp)


def _scorer_tol(gold):
    """The harness numeric acceptance radius around gold, exact Fraction:
    answers_match accepts got when |got - gold| <= max(1e-9, 1e-6*|gold|)."""
    t = abs(gold) * Fraction(1, 10 ** 6)
    floor = Fraction(1, 10 ** 9)
    return t if t > floor else floor


def _frac_gold_safe(gold, near):
    """True iff every plausible wrong answer sits safely OUTSIDE the scorer's
    numeric tolerance around gold.

    `near` holds the family-specific wrong evaluations (one term/step short,
    one step long, dropped last term), as Fractions.  Generic numerator+-1 /
    denominator+-1 off-by-ones are added here.  Bar: strictly more than 10x
    the harness tolerance (i.e. >1e-5 relative in the relative regime).
    Note num+-1 passing implies |numerator| < 1e5 and den+-1 passing implies
    denominator < 1e5, so the bound subsumes any denominator cap.

    Decimal renderings of gold itself are deliberately NOT screened: a 6+
    significant-digit decimal of the correct value is numerically CORRECT --
    the harness matcher's whole design is to accept numerically-equal
    variants -- and screening it is impossible in general anyway (for any
    gold with mantissa >= ~3.3 the 6-sig rounding error is always inside
    tolerance, so such a gate silently extinguishes those instances; proven
    for 9-sig on ALL golds). Only genuinely WRONG values are screened.
    """
    n, d = gold.numerator, gold.denominator
    cands = list(near) + [Fraction(n + 1, d), Fraction(n - 1, d),
                          Fraction(n, d + 1)]
    if d > 1:
        cands.append(Fraction(n, d - 1))
    bar = 10 * _scorer_tol(gold)
    for v in cands:
        if v != gold and abs(v - gold) <= bar:
            return False
    return True


_INT_FMT = ("Answer with a bare integer: digits only, no commas, no units, "
            "no words, no LaTeX.")
_NEGINT_FMT = ("Answer with a bare integer (it may be negative): an optional "
               "minus sign then digits only, no commas, no units, no words, "
               "no LaTeX. If the answer is negative, use the ASCII "
               "hyphen-minus character '-' (U+002D) as the minus sign, not a "
               "typographic minus.")
_FRAC_FMT = ("Answer with the exact value as a reduced fraction in the form "
             "a/b with b positive and gcd(a,b)=1 (if the value is an integer, "
             "give just that integer). No decimals, no units, no LaTeX. If "
             "the value is negative, put the sign on the numerator and use "
             "the ASCII hyphen-minus character '-' (U+002D) as the minus "
             "sign, not a typographic minus.")
_TAIL = ("\nEnd your reply with a line of exactly this form:\n"
         "FINAL ANSWER: <answer>")


# --------------------------------------------------------------------------
# family 1: CRT with 4-5 pairwise-coprime moduli and coefficient congruences
# --------------------------------------------------------------------------

_CRT_POOL = [7, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37, 41, 43,
             47, 49, 53, 59, 61, 121, 125, 169]


def _pairwise_coprime_sample(rng, k):
    while True:
        cand = rng.sample(_CRT_POOL, k)
        ok = all(gcd(a, b) == 1 for a, b in itertools.combinations(cand, 2))
        if ok:
            prod = 1
            for c in cand:
                prod *= c
            if 2 * 10 ** 5 <= prod <= 4 * 10 ** 7:
                return cand, prod


def _f_crt(rng, idx):
    k = rng.choice([4, 4, 5])
    mods, M = _pairwise_coprime_sample(rng, k)
    lines = []
    residues = []
    for m in mods:
        # a*x = b (mod m) with gcd(a,m)=1 -> forces a modular inverse step
        while True:
            a = rng.randrange(2, m)
            if gcd(a, m) == 1:
                break
        b = rng.randrange(1, m)
        residues.append(b * _inv(a, m) % m)
        lines.append("  %d*x = %d  (mod %d)" % (a, b, m))
    r, _M = _crt(residues, mods)
    assert _M == M
    style = rng.choice(["smallest", "above"])
    if style == "smallest":
        ans = r if r != 0 else M
        q = ("Find the smallest strictly positive integer x satisfying all of "
             "these simultaneous congruences.")
    else:
        L = rng.randrange(M, 6 * M)
        # smallest x > L with x = r (mod M)
        ans = r + M * ((L - r) // M + 1)
        while ans <= L:
            ans += M
        q = ("Find the smallest integer x with x > %d satisfying all of these "
             "simultaneous congruences." % L)
    prompt = (
        "Solve this system of linear congruences over the integers.\n\n"
        + "\n".join(lines)
        + "\n\nThe moduli are pairwise coprime. " + q
        + "\n\n" + _INT_FMT + _TAIL
    )
    return prompt, str(ans), None


# --------------------------------------------------------------------------
# family 2: continued fraction -> exact reduced fraction
# --------------------------------------------------------------------------


def _cf_eval(ts):
    v = Fraction(ts[-1])
    for t in reversed(ts[:-1]):
        v = t + Fraction(1) / v
    return v


def _f_contfrac(rng, idx):
    # Scorer-tolerance safety forces SMALL convergent denominators: the CF
    # truncated one term short differs from the full value by exactly
    # 1/(q_k*q_{k-1}), so q_k*q_{k-1} must stay low enough that this classic
    # slip lands >10x outside the harness tolerance.  Terms are drawn small
    # and _frac_gold_safe arbitrates every instance.
    for _attempt in range(200000):
        k = rng.choice([5, 6, 7])
        terms = ([rng.randrange(1, 10)] +
                 [rng.choice([1, 1, 1, 2, 2, 3, 4]) for _ in range(k - 1)] +
                 [rng.choice([2, 2, 3, 4, 5])])    # last term != 1: canonical
        val = _cf_eval(terms)
        prev = _cf_eval(terms[:-1])       # truncated one term short
        prev2 = _cf_eval(terms[:-2])      # truncated two terms short
        twist = rng.choice(["plain", "recip", "minus"])
        extra = ""
        if twist == "plain":
            target = val
            near = [prev, prev2]
        elif twist == "recip":
            target = Fraction(1) / val
            extra = "\nReport the reciprocal 1/x of that value, not x itself."
            near = [Fraction(1) / prev, Fraction(1) / prev2, val, prev]
        else:
            num = rng.randrange(2, 12)
            den = rng.randrange(2, 12)
            target = val - Fraction(num, den)
            extra = ("\nThen subtract %d/%d from that value and report the "
                     "result." % (num, den))
            near = [prev - Fraction(num, den), prev2 - Fraction(num, den),
                    val, prev]
        if _frac_gold_safe(target, near):
            break
    else:
        raise RuntimeError("contfrac generation failed")
    body = "x = " + str(terms[0]) + " + 1/(" + \
        " + 1/(".join(str(t) for t in terms[1:]) + ")" * k
    gold, variants = _fmt_frac(target)
    notation = str(terms[0]) + "; " + ", ".join(str(t) for t in terms[1:])
    prompt = (
        "Evaluate this finite continued fraction exactly, using exact rational "
        "arithmetic (no decimal approximation at any step):\n\n"
        "  x = [%s]\n\nwhich means\n\n  %s\n%s\n\n%s%s"
        % (notation, body, extra, _FRAC_FMT, _TAIL)
    )
    return prompt, gold, variants


# --------------------------------------------------------------------------
# family 3: telescoping / partial-fraction exact sums
# --------------------------------------------------------------------------


def _f_telescope(rng, idx):
    # No branch here has a memorized single-boundary-term closed form.
    # 'asym' and 'mixed' force a real 3-coefficient partial-fraction solve
    # whose telescoping leaves residual harmonic head/tail blocks; 'kd' with
    # d in {3,4} leaves d head terms and d tail terms to book-keep.
    # Scorer-tolerance safety: the reduced denominator (and numerator) of the
    # gold must stay well under 1e5 or numerator/denominator off-by-one wrong
    # answers fall inside the harness numeric tolerance; that caps N for the
    # wide-spread kinds.  _frac_gold_safe arbitrates every instance, including
    # the dropped-last-term sum S(N-1) and the extra-term sum S(N+1).
    for _attempt in range(200000):
        kind = rng.choice(["asym", "mixed", "kd"])
        if kind == "asym":
            a = rng.choice([1, 2, 3])
            b = a + rng.choice([2, 3, 4])
            N = rng.randrange(6, 41)

            def term(k, a=a, b=b):
                return Fraction(1, k * (k + a) * (k + b))
            expr = "1/(k*(k+%d)*(k+%d))" % (a, b)
        elif kind == "mixed":
            while True:
                u = rng.randrange(2, 7)
                v = rng.randrange(1, 10)
                # v==u   -> numerator u*(k+1) cancels: collapses to u/(k*(k+2))
                # v==2*u -> the 1/(k+2) coefficient vanishes: easier split
                if v != u and v != 2 * u:
                    break
            N = rng.randrange(12, 47)

            def term(k, u=u, v=v):
                return Fraction(u * k + v, k * (k + 1) * (k + 2))
            expr = "(%d*k + %d)/(k*(k+1)*(k+2))" % (u, v)
        else:
            d = rng.choice([3, 4])
            N = rng.randrange(12, 47)

            def term(k, d=d):
                return Fraction(1, k * (k + d))
            expr = "1/(k*(k+%d))" % d
        total = sum(term(k) for k in range(1, N + 1))
        near = [total - term(N),        # dropped last term  (= S(N-1))
                total + term(N + 1)]    # one extra term     (= S(N+1))
        if _frac_gold_safe(total, near):
            break
    else:
        raise RuntimeError("telescope generation failed")
    gold, variants = _fmt_frac(total)
    prompt = (
        "Compute the exact value of the finite sum\n\n"
        "  S = sum_{k=1}^{%d} %s\n\n"
        "Use partial fractions / telescoping and exact rational arithmetic. "
        "Do not approximate.\n\n%s%s" % (N, expr, _FRAC_FMT, _TAIL)
    )
    return prompt, gold, variants


# --------------------------------------------------------------------------
# family 4: linear Diophantine with a minimality constraint
# --------------------------------------------------------------------------


def _f_diophantine(rng, idx):
    while True:
        a = rng.randrange(137, 1400)
        b = rng.randrange(137, 1400)
        g = gcd(a, b)
        if g in (1, 2, 3, 5, 7) and a != b:
            break
    c = g * rng.randrange(200, 4000)
    g2, x0, y0 = _egcd(a, b)
    if g2 < 0:
        g2, x0, y0 = -g2, -x0, -y0
    x0 *= c // g2
    y0 *= c // g2
    stepx = b // g2
    stepy = a // g2
    # smallest strictly positive x
    t = -((x0 - 1) // stepx)
    x = x0 + stepx * t
    while x <= 0:
        x += stepx
        t += 1
    while x - stepx > 0:
        x -= stepx
        t -= 1
    y = y0 - stepy * t
    assert a * x + b * y == c
    which = rng.choice(["y", "x"])
    if which == "y":
        ans, ask = y, ("report the corresponding value of y")
    else:
        ans, ask = x, ("report that value of x")
    prompt = (
        "Consider the linear Diophantine equation\n\n"
        "  %d*x + %d*y = %d\n\n"
        "over the integers. Among all integer solutions (x, y), take the one "
        "with the smallest strictly positive x, and %s.\n\n%s%s"
        % (a, b, c, ask, _NEGINT_FMT, _TAIL)
    )
    return prompt, str(ans), _neg_variants(ans)


# --------------------------------------------------------------------------
# family 5: inclusion-exclusion combinatorics
# --------------------------------------------------------------------------


def _bounded_compositions(total, caps):
    """#solutions of x1+..+xm = total, 0 <= xi <= caps[i]  (exact, IE)."""
    m = len(caps)
    res = 0
    for r in range(m + 1):
        for sub in itertools.combinations(range(m), r):
            s = total - sum(caps[i] + 1 for i in sub)
            if s < 0:
                continue
            # C(s + m - 1, m - 1)
            num = 1
            for j in range(m - 1):
                num = num * (s + m - 1 - j)
            res += (-1) ** r * num // factorial(m - 1)
    return res


def _surjections(n, k):
    return sum((-1) ** i * (factorial(k) // (factorial(i) * factorial(k - i)))
               * (k - i) ** n for i in range(k + 1))


def _no_adjacent_equal(counts):
    """Exact count of arrangements of a multiset with no two equal adjacent.
    Computed by DP over (remaining counts, last symbol) -- independent of any
    closed form."""
    from functools import lru_cache
    m = len(counts)

    @lru_cache(maxsize=None)
    def go(state, last):
        if sum(state) == 0:
            return 1
        tot = 0
        for i in range(m):
            if state[i] and i != last:
                nxt = list(state)
                nxt[i] -= 1
                tot += go(tuple(nxt), i)
        return tot

    return go(tuple(counts), -1)


def _f_combinatorics(rng, idx):
    for _attempt in range(300):
        out = _f_combinatorics_once(rng, idx)
        if out is not None:
            return out
    raise RuntimeError("combinatorics generation failed")


def _count_forbidden_perms(n, forb):
    """#permutations sigma of 1..n with sigma(i) not in forb[i], by bitmask DP
    over used values (position i+1 is decided at popcount(mask)==i)."""
    dp = [0] * (1 << n)
    dp[0] = 1
    for mask in range(1 << n):
        if not dp[mask]:
            continue
        i = bin(mask).count("1") + 1          # 1-indexed next position
        if i > n:
            continue
        bad = forb.get(i, ())
        for v in range(1, n + 1):
            bit = 1 << (v - 1)
            if (mask & bit) or v in bad:
                continue
            dp[mask | bit] += dp[mask]
    return dp[(1 << n) - 1]


def _f_combinatorics_once(rng, idx):
    kind = rng.choice(["bounded", "surj", "noadj", "forbidden"])
    if kind == "bounded":
        m = rng.choice([4, 5])
        caps = [rng.randrange(3, 12) for _ in range(m)]
        lo = sum(caps) // 3
        total = rng.randrange(lo + 2, sum(caps) - 1)
        ans = _bounded_compositions(total, caps)
        capstr = ", ".join("0 <= x%d <= %d" % (i + 1, c)
                           for i, c in enumerate(caps))
        prompt = (
            "Count the integer solutions of\n\n  %s = %d\n\nsubject to\n\n  %s"
            "\n\nUse inclusion-exclusion; give the exact count.\n\n%s%s"
            % (" + ".join("x%d" % (i + 1) for i in range(m)), total, capstr,
               _INT_FMT, _TAIL)
        )
    elif kind == "surj":
        # NOT the bare memorized alternating sum: the extra f(1) != f(2)
        # constraint needs the merge argument surj(n,k) - surj(n-1,k)
        # (or a two-layer inclusion-exclusion).
        k = rng.choice([4, 5])
        n = rng.randrange(k + 2, k + 5)
        ans = _surjections(n, k) - _surjections(n - 1, k)
        prompt = (
            "Label the domain elements 1, 2, ..., %d. How many functions from "
            "a set of %d distinct elements onto a set of %d distinct elements "
            "are surjective (every one of the %d targets is hit at least "
            "once) AND additionally satisfy f(1) != f(2), i.e. the two "
            "designated domain elements 1 and 2 receive different values?\n\n"
            "Give the exact count.\n\n%s%s" % (n, n, k, k, _INT_FMT, _TAIL)
        )
    elif kind == "noadj":
        letters = "ABCDE"
        m = rng.choice([3, 3, 4])
        counts = [rng.randrange(1, 4) for _ in range(m)]
        tot = sum(counts)
        # feasibility + non-triviality: max count must not exceed ceil(tot/2)
        if not (6 <= tot <= 9) or max(counts) > (tot + 1) // 2:
            return None
        ans = _no_adjacent_equal(counts)
        if ans < 20:
            return None
        word = "".join(letters[i] * counts[i] for i in range(m))
        desc = ", ".join("%d cop%s of %s"
                         % (counts[i], "y" if counts[i] == 1 else "ies",
                            letters[i]) for i in range(m))
        prompt = (
            "Consider the multiset of letters consisting of %s (a total of %d "
            "letters, e.g. the string %s). How many distinct arrangements of "
            "all %d letters in a row have NO two identical letters adjacent to "
            "each other?\n\nGive the exact count.\n\n%s%s"
            % (desc, tot, word, tot, _INT_FMT, _TAIL)
        )
    else:
        # forbidden-position permanent: inclusion-exclusion where the sets
        # are matchings on an IRREGULAR rook board, not divisibility classes.
        n = rng.randrange(7, 10)
        npairs = rng.randrange(n, n + 5)
        cells = set()
        for _try in range(600):
            if len(cells) >= npairs:
                break
            r = rng.randrange(1, n + 1)
            c = rng.randrange(1, n + 1)
            if (r, c) in cells:
                continue
            if sum(1 for (rr, _) in cells if rr == r) >= 2:
                continue
            if sum(1 for (_, cc) in cells if cc == c) >= 2:
                continue
            cells.add((r, c))
        if len(cells) < npairs:
            return None
        cells = sorted(cells)
        rowcount = {}
        for r, _ in cells:
            rowcount[r] = rowcount.get(r, 0) + 1
        # irregularity: >=2 positions carry two restrictions, >=1 carries none
        if sum(1 for v in rowcount.values() if v == 2) < 2:
            return None
        if len(rowcount) >= n:
            return None
        forb = {}
        for r, c in cells:
            forb.setdefault(r, set()).add(c)
        ans = _count_forbidden_perms(n, forb)
        prompt = (
            "Count the permutations sigma of {1, 2, ..., %d} (each of the %d "
            "values used exactly once) that satisfy ALL of these "
            "restrictions simultaneously:\n\n%s\n\nHere sigma(i) is the value "
            "placed at position i. Use inclusion-exclusion over the forbidden "
            "cells (rook-polynomial style); give the exact count.\n\n%s%s"
            % (n, n,
               "\n".join("  sigma(%d) != %d" % (r, c) for r, c in cells),
               _INT_FMT, _TAIL)
        )
    if ans < 20:            # no trivial / degenerate counts
        return None
    return prompt, str(ans), None


# --------------------------------------------------------------------------
# family 6: linear recurrence at astronomically large n under a small modulus
# --------------------------------------------------------------------------


def _seq_profile(coeffs, seeds, m, horizon=200000):
    """Distinct values and cycle length of the state sequence mod m."""
    order = len(coeffs)
    state = tuple(s % m for s in seeds)
    seen = {}
    vals = set(state)
    i = 0
    while state not in seen:
        seen[state] = i
        nxt = sum(coeffs[j] * state[order - 1 - j] for j in range(order)) % m
        state = state[1:] + (nxt,)
        vals.add(nxt)
        i += 1
        if i > horizon:
            break
    period = i - seen.get(state, 0)
    return len(vals), period


_REC_PRIMES = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
               163, 167, 173, 179, 181, 191, 193, 197, 199]


def _divisors(x):
    out = []
    i = 1
    while i * i <= x:
        if x % i == 0:
            out.append(i)
            if i != x // i:
                out.append(x // i)
        i += 1
    return sorted(out)


def _f_recurrence(rng, idx):
    # Order 2 only (order-3 made the period route infeasible and the tiny
    # modulus made lucky guessing ~1/m; see review). Modulus is a prime in
    # the low hundreds -> answer space 0..m-1 kills guessing. Eigenvalues r,s
    # are built with multiplicative order dividing chosen divisors of m-1, so
    # the state period = lcm(ord r, ord s) divides m-1 (<200): the period
    # route needs a careful <=200-step iteration and index reduction, and the
    # matrix-power route needs ~47 correct modular squarings.
    for _attempt in range(2000):
        m = rng.choice(_REC_PRIMES)
        divs = [d for d in _divisors(m - 1) if 20 <= d <= 200]
        if not divs:
            continue
        d1 = rng.choice(divs)
        d2 = rng.choice(divs)
        r = pow(rng.randrange(2, m), (m - 1) // d1, m)
        s = pow(rng.randrange(2, m), (m - 1) // d2, m)
        if r in (0, 1) or s in (0, 1) or r == s:
            continue
        c1 = (r + s) % m
        c2 = (-r * s) % m
        if c1 == 0 or c2 == 0:
            continue
        seeds = [rng.randrange(1, m) for _ in range(2)]
        nvals, period = _seq_profile([c1, c2], seeds, m, horizon=2000)
        if not (24 <= period <= 200) or nvals < 12:
            continue
        break
    else:
        raise RuntimeError("recurrence generation failed")
    N = rng.randrange(10 ** 11, 10 ** 15)
    # companion matrix power (computed, not asserted)
    A = [[c1 % m, c2 % m], [1, 0]]
    vec = [seeds[1] % m, seeds[0] % m]
    P = _mat_pow(A, N - 1, m)
    ans = sum(P[0][j] * vec[j] for j in range(2)) % m
    prompt = (
        "A sequence of integers is defined by\n\n"
        "  a(n) = %d*a(n-1) + %d*a(n-2)   for n >= 2\n\n"
        "with initial values a(0) = %d, a(1) = %d.\n\n"
        "Compute a(%d) mod %d. (The sequence is purely periodic modulo %d "
        "and its period is at most 200, so you can find the period by "
        "iterating the recurrence mod %d and then reduce the index, or use "
        "matrix exponentiation.) The answer is an integer in the range 0 to "
        "%d inclusive.\n\n%s%s"
        % (c1, c2, seeds[0], seeds[1], N, m, m, m, m - 1, _INT_FMT, _TAIL)
    )
    return prompt, str(ans), None


# --------------------------------------------------------------------------
# family 7: exact determinant of a structured 5x5 integer matrix
# --------------------------------------------------------------------------


def _f_determinant(rng, idx):
    # No 4x4 (materially easier for a strong model). Dense/rank1plus are 5x5;
    # banded is 6x6 sparse-structured.
    for _attempt in range(200):
        style = rng.choice(["dense", "banded", "rank1plus"])
        n = 6 if style == "banded" else 5
        if style == "dense":
            M = [[rng.randrange(-6, 10) for _ in range(n)] for _ in range(n)]
        elif style == "banded":
            M = [[0] * n for _ in range(n)]
            for i in range(n):
                M[i][i] = rng.randrange(2, 9)
                if i + 1 < n:
                    M[i][i + 1] = rng.randrange(-5, 6)
                    M[i + 1][i] = rng.randrange(-5, 6)
                if i + 2 < n:
                    M[i][i + 2] = rng.randrange(-3, 4)
        else:
            u = [rng.randrange(1, 5) for _ in range(n)]
            v = [rng.randrange(1, 5) for _ in range(n)]
            d = [rng.randrange(-7, 8) for _ in range(n)]
            if any(x == 0 for x in d):
                continue
            M = [[u[i] * v[j] + (d[i] if i == j else 0) for j in range(n)]
                 for i in range(n)]
        ans = _det_fraction(M)
        # reject singular (guessable "0") and reject any two identical rows
        rows_distinct = len({tuple(r) for r in M}) == n
        if ans != 0 and abs(ans) > 3 and rows_distinct:
            break
    rows = "\n".join("  [ " + "  ".join("%4d" % x for x in row) + " ]"
                     for row in M)
    prompt = (
        "Compute the exact determinant of this %dx%d integer matrix:\n\n%s\n\n"
        "Work exactly (integer or exact rational arithmetic only; no floating "
        "point rounding). The determinant is an integer.\n\n%s%s"
        % (n, n, rows, _NEGINT_FMT, _TAIL)
    )
    return prompt, str(ans), _neg_variants(ans)


# --------------------------------------------------------------------------
# family 8: multiplicative order + huge modular exponentiation
# --------------------------------------------------------------------------

_PRIMES = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
           167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233]


def _order(a, m):
    k, cur = 1, a % m
    while cur != 1:
        cur = cur * a % m
        k += 1
    return k


def _f_order(rng, idx):
    p = rng.choice(_PRIMES)
    a = rng.randrange(2, p - 1)
    d = _order(a, p)
    kind = rng.choice(["value", "order", "tower"])
    if kind == "order":
        # Hardened: order of an element modulo a COMPOSITE m = p*q. The
        # divisor-of-p-1 shortcut no longer applies directly; the model must
        # split via CRT (order mod p, order mod q) and combine with lcm.
        for _attempt in range(400):
            p1, p2 = rng.sample(_PRIMES, 2)
            mm = p1 * p2
            a = rng.randrange(2, mm - 1)
            if gcd(a, mm) != 1:
                continue
            d1 = _order(a, p1)
            d2 = _order(a, p2)
            if d1 > 8 and d2 > 8 and d1 != d2:
                break
        else:
            raise RuntimeError("order/composite generation failed")
        ans = _order(a, mm)          # naive loop mod m: computed, not asserted
        prompt = (
            "Let m = %d = %d * %d (a product of two distinct primes) and "
            "a = %d, with gcd(a, m) = 1. The multiplicative order of a "
            "modulo m is the smallest integer k >= 1 with a^k = 1 (mod m).\n\n"
            "Hint: find the order of a modulo %d (it divides %d) and the "
            "order of a modulo %d (it divides %d) separately, then combine "
            "the two orders correctly.\n\nCompute that order k.\n\n%s%s"
            % (mm, p1, p2, a, p1, p1 - 1, p2, p2 - 1, _INT_FMT, _TAIL)
        )
    elif kind == "value":
        N = rng.randrange(10 ** 12, 10 ** 16)
        ans = pow(a, N, p)
        prompt = (
            "Let p = %d (a prime) and a = %d. Compute\n\n  a^%d mod p\n\n"
            "Hint: first find the multiplicative order of a modulo p (it "
            "divides p-1 = %d), then reduce the exponent. The answer is an "
            "integer in the range 0 to %d inclusive.\n\n%s%s"
            % (p, a, N, p - 1, p - 1, _INT_FMT, _TAIL)
        )
    else:
        b = rng.randrange(2, 40)
        e = rng.randrange(20, 60)
        ans = pow(a, pow(b, e), p)
        prompt = (
            "Let p = %d (a prime) and a = %d. Compute the power tower value\n\n"
            "  a^(%d^%d) mod p\n\n"
            "Hint: reduce the exponent %d^%d modulo the multiplicative order "
            "of a modulo p (that order divides p-1 = %d). The answer is an "
            "integer in the range 0 to %d inclusive.\n\n%s%s"
            % (p, a, b, e, b, e, p - 1, p - 1, _INT_FMT, _TAIL)
        )
    return prompt, str(ans), None


# --------------------------------------------------------------------------
# family 9: iterated Mobius (linear-fractional) map in exact rationals
# --------------------------------------------------------------------------


def _f_mobius(rng, idx):
    # Scorer-tolerance safety: near an attracting fixed point the one-step-
    # short iterate x_{steps-1} (and the one-step-long x_{steps+1}) can fall
    # inside the harness tolerance of x_steps, and gold denominators past 1e5
    # make den+-1 wrong answers pass; _frac_gold_safe rejects those instances.
    for _attempt in range(200000):
        steps = rng.choice([5, 6, 7])
        a = rng.randrange(1, 7)
        b = rng.randrange(-6, 8)
        c = rng.randrange(1, 6)
        d = rng.randrange(-6, 8)
        p = rng.randrange(1, 12)
        q = rng.randrange(2, 12)
        if a * d - b * c == 0:          # degenerate: f is constant
            continue
        x = Fraction(p, q)
        seen = [x]
        ok = True
        for _ in range(steps):
            den = c * x + d
            if den == 0:
                ok = False
                break
            x = (a * x + b) / den
            seen.append(x)
        if not ok:
            continue
        # reject fixed points / short cycles (answer would be trivial) and
        # reject tiny results that could be guessed
        if len(set(seen)) < steps + 1:
            continue
        if abs(x.numerator) < 60 or x.denominator < 30:
            continue
        near = [seen[-2]]                    # one step short: x_{steps-1}
        nden = c * x + d
        if nden != 0:
            near.append((a * x + b) / nden)  # one step long:  x_{steps+1}
        if not _frac_gold_safe(x, near):
            continue
        break
    else:
        raise RuntimeError("mobius generation failed")
    gold, variants = _fmt_frac(x)
    def _lin(m, k):
        return "%d*t %s %d" % (m, "-" if k < 0 else "+", abs(k))

    prompt = (
        "Define the map f(t) = (%s) / (%s) on the rationals.\n\n"
        "Starting from x_0 = %d/%d, set x_{k+1} = f(x_k). Compute x_%d exactly, "
        "keeping every intermediate value as an exact fraction in lowest terms "
        "(never a decimal).\n\n%s%s"
        % (_lin(a, b), _lin(c, d), p, q, steps, _FRAC_FMT, _TAIL)
    )
    return prompt, gold, variants


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

_FAMILIES = [
    ("crt", _f_crt),
    ("contfrac", _f_contfrac),
    ("telescope", _f_telescope),
    ("dioph", _f_diophantine),
    ("comb", _f_combinatorics),
    ("recur", _f_recurrence),
    ("det", _f_determinant),
    ("order", _f_order),
    ("mobius", _f_mobius),
]


def gen(seed, n):
    """Return n deterministic hard-math items for the given seed."""
    rng = random.Random(seed * 1000003 + 17)
    order = []
    while len(order) < n:
        block = list(range(len(_FAMILIES)))
        rng.shuffle(block)
        order.extend(block)
    order = order[:n]

    items = []
    seen_prompts = set()
    for i, fi in enumerate(order):
        name, fn = _FAMILIES[fi]
        base = rng.randrange(1 << 62)
        for bump in range(1000):
            # deterministic internal salt bump: regenerate on a within-seed
            # prompt collision (observed for comb) without disturbing the
            # master stream that seeds the other items
            sub = random.Random(base + bump)
            prompt, answer, variants = fn(sub, i)
            if prompt not in seen_prompts:
                break
        else:
            raise RuntimeError("prompt decollision failed for item %d" % i)
        seen_prompts.add(prompt)
        item = {
            "id": "math_hard_%02d_%s" % (i + 1, name),
            "prompt": prompt,
            "answer": answer,
        }
        if variants and len(variants) > 1:
            item["answer_any"] = variants
        items.append(item)
    return items


if __name__ == "__main__":
    import json
    import sys
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    print(json.dumps(gen(s, k), indent=2))
