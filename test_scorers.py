#!/usr/bin/env python3
"""Validate the harness scorers against known-good and known-bad responses.
Run:  python3 test_scorers.py   (exit 0 = all scorer paths behave)"""
import json
from pathlib import Path
from harness import (score_math, score_code, score_json, score_instruct,
                     answers_match, gen_longctx, strip_think, load_jsonl)

HERE = Path(__file__).parent
fails = []


def check(name, cond):
    if not cond:
        fails.append(name)
        print(f"FAIL {name}")
    else:
        print(f"ok   {name}")


R = lambda text: {"content": strip_think(text), "raw_content": text,
                  "latency": 0.1, "completion_tokens": 10, "finish": "stop"}

# ---- answers_match edge cases -------------------------------------------
check("num fraction vs decimal", answers_match("8500.80", "42504/5"))
check("dollar+comma", answers_match("$8,500.80", "42504/5"))
check("unreduced fraction", answers_match("103776/2598960", "2162/54145"))
check("trailing period", answers_match("649.", "649"))
check("bold markdown", answers_match("**649**", "649"))
check("case-insensitive string", answers_match("A3-101x", "a3-101X"))
check("wrong number rejected", not answers_match("650", "649"))
check("leading-zero code exact", answers_match("043501", "43501"))  # numeric path

# ---- math scorer ---------------------------------------------------------
math_tasks = load_jsonl(HERE / "tasks/math.jsonl")
m01 = next(t for t in math_tasks if t["id"] == "m01")
check("math pass", score_math(m01, R("thinking...\nFINAL ANSWER: 649"))[0])
check("math pass w/ think block",
      score_math(m01, R("<think>7^4=401...</think>FINAL ANSWER: 649"))[0])
check("math fail wrong", not score_math(m01, R("FINAL ANSWER: 343"))[0])
check("math fail no line", not score_math(m01, R("The answer is 649"))[0])
m20 = next(t for t in math_tasks if t["id"] == "m20")
check("math fraction/decimal equiv", score_math(m20, R("FINAL ANSWER: $8500.80"))[0])

# ---- code scorer ---------------------------------------------------------
code_tasks = load_jsonl(HERE / "tasks/code.jsonl")
c01 = next(t for t in code_tasks if t["id"] == "c01")
good = ("Here you go:\n```python\n"
        "def rle(s):\n"
        "    if not s: return ''\n"
        "    out=[]; cur=s[0]; n=1\n"
        "    for ch in s[1:]:\n"
        "        if ch==cur: n+=1\n"
        "        else: out.append(cur+str(n)); cur=ch; n=1\n"
        "    out.append(cur+str(n))\n"
        "    return ''.join(out)\n```")
bad = "```python\ndef rle(s):\n    return s\n```"
check("code pass", score_code(c01, R(good))[0])
check("code fail", not score_code(c01, R(bad))[0])
check("code fail no entry", not score_code(c01, R("```python\nx=1\n```"))[0])
hang = "```python\ndef rle(s):\n    while True: pass\n```"
ok, why = score_code(c01, R(hang))
check("code timeout handled", (not ok) and why == "timeout")

# ---- json scorer ---------------------------------------------------------
json_tasks = load_jsonl(HERE / "tasks/json_tool.jsonl")
j01 = next(t for t in json_tasks if t["id"] == "j01")
good_j = json.dumps({"tool": "schedule_snapshot",
                     "args": {"vm": "pg-primary", "retain_days": 14, "quiesce": True}})
check("json pass", score_json(j01, R(good_j))[0])
check("json pass fenced", score_json(j01, R("```json\n" + good_j + "\n```"))[0])
bad_type = json.dumps({"tool": "schedule_snapshot",
                       "args": {"vm": "pg-primary", "retain_days": "14",
                                "quiesce": True}})
check("json fail str-int", not score_json(j01, R(bad_type))[0])
check("json fail prose", not score_json(j01, R("Sure! " + good_j))[0])
j05 = next(t for t in json_tasks if t["id"] == "j05")
good5 = json.dumps({"message": 'He said "run it" twice.\nThen stopped.'})
check("json escaped newline+quotes", score_json(j05, R(good5))[0])
j11 = next(t for t in json_tasks if t["id"] == "j11")
extra_key = json.dumps({"53": 2809, "59": 3481, "61": 3721, "67": 4489})
check("json keys_exactly rejects extra", not score_json(j11, R(extra_key))[0])
bool_trap = json.dumps({"tool": "schedule_snapshot",
                        "args": {"vm": "pg-primary", "retain_days": True,
                                 "quiesce": True}})
check("json bool-not-int", not score_json(j01, R(bool_trap))[0])

# ---- instruct scorer -----------------------------------------------------
inst = load_jsonl(HERE / "tasks/instruct.jsonl")
g = lambda i: next(t for t in inst if t["id"] == i)
check("i03 pass", score_instruct(g("i03"), R("Au"))[0])
check("i03 fail", not score_instruct(g("i03"), R("Gold is Au"))[0])
check("i05 pass", score_instruct(g("i05"), R("stable stable stable stable stable"))[0])
check("i05 fail count", not score_instruct(
    g("i05"), R("stable stable stable stable"))[0])
check("i08 pass", score_instruct(
    g("i08"), R("A crimson orb sinks low, kissing far hills. Night winds hum "
                "soft songs of vanishing light."))[0])
check("i08 fail letter-e", not score_instruct(
    g("i08"), R("The sun sets slowly. Evening arrives quietly."))[0])
check("i10 pass", score_instruct(g("i10"), R("150000"))[0])
check("i10 fail units", not score_instruct(g("i10"), R("150000 ms"))[0])
check("i13 pass", score_instruct(g("i13"), R("Callisto,Europa,Ganymede,Io"))[0])
check("i13 fail spaces", not score_instruct(
    g("i13"), R("Callisto, Europa, Ganymede, Io"))[0])
check("i14 pass", score_instruct(g("i14"), R("BEGIN\nEND"))[0])
check("i01 pass", score_instruct(
    g("i01"), R("NVMe attaches flash over PCIe lanes. It cuts protocol overhead "
                "compared with SATA. Parallel queues keep many operations in "
                "flight."))[0])
check("i01 fail 4 sentences", not score_instruct(
    g("i01"), R("One. Two. Three. Four."))[0])
check("i04 pass", score_instruct(
    g("i04"), R("A hypervisor is software that creates and runs virtual machines "
                "by abstracting physical hardware into shared pools of compute, "
                "memory, storage, and networking. It schedules guest access to "
                "real resources, enforces isolation between workloads, and lets "
                "one physical server safely host many independent operating "
                "systems at the same time today."))[0])

# ---- longctx generation sanity ------------------------------------------
lt = gen_longctx(seed=7, n_items=12, ctx_tokens=2000)
check("longctx count", len(lt) == 12)
check("longctx unique ids", len({t["id"] for t in lt}) == 12)
for t in lt:
    check(f"longctx answer-in-haystack {t['id']}", t["answer"] in t["prompt"])
lt2 = gen_longctx(seed=7, n_items=12, ctx_tokens=2000)
check("longctx deterministic", json.dumps(lt) == json.dumps(lt2))
cnt = [t for t in lt if t["id"].startswith("l_cnt")][0]
hay = cnt["prompt"].split("=== DATABASE ===")[1].split("=== END ===")[0]
occ = hay.count("dept Cryogenics")
check("longctx count-answer correct", str(occ) == cnt["answer"])

print()
if fails:
    print(f"{len(fails)} FAILURES: {fails}")
    raise SystemExit(1)
print("ALL SCORER TESTS PASS")

# ---- new categories: bizarre + grounded ----------------------------------
from harness import score_bizarre
biz = load_jsonl(HERE / "tasks/bizarre.jsonl")
gb = lambda i: next(t for t in biz if t["id"] == i)
check("b07 pass '2nd'", score_bizarre(gb("b07"), R("FINAL ANSWER: 2nd"))[0])
check("b07 pass 'second'", score_bizarre(gb("b07"), R("FINAL ANSWER: Second"))[0])
check("b07 fail 'first'", not score_bizarre(gb("b07"), R("FINAL ANSWER: first"))[0])
check("b10 microwave pass", score_bizarre(gb("b10"), R("beep beep beep"))[0])
check("b10 microwave fail", not score_bizarre(gb("b10"), R("Beep! Beep! Beep!"))[0])
check("b13 decimal pass", score_bizarre(gb("b13"), R("FINAL ANSWER: 8.40"))[0])
check("b18 strawberry", gb("b18")["answer"] == "9")
check("b19 parrot r-count", gb("b19")["answer"] == str("purple parrot territory".count("r")))
check("b20 reversed pass", score_bizarre(
    gb("b20"), R("FINAL ANSWER: money me owes moon the"))[0])
check("b11 palindrome", gb("b11")["answer"] == "4994")

grd = load_jsonl(HERE / "tasks/grounded.jsonl")
gg = lambda i: next(t for t in grd if t["id"] == i)
check("g04 pass", score_math(gg("g04"), R("FINAL ANSWER: 7719"))[0])
check("g03 trap pass", score_math(gg("g03"), R("FINAL ANSWER: NOT IN CONTEXT"))[0])
check("g03 trap pass lc", score_math(gg("g03"), R("FINAL ANSWER: not in context"))[0])
check("g03 trap fail hallucination",
      not score_math(gg("g03"), R("FINAL ANSWER: 18.2 GHz"))[0])
check("g14 ratio alt", score_math(gg("g14"), R("FINAL ANSWER: 1 to 9"))[0])
check("grounded traps count", sum(1 for t in grd
      if (t.get("answer_any") or [""])[0] == "not in context") == 5)

print()
if fails:
    print(f"{len(fails)} FAILURES: {fails}")
    raise SystemExit(1)
print("ALL NEW-CATEGORY TESTS PASS")

# ---- nonsense: unscored by construction ----------------------------------
from harness import score_none, UNSCORED
non = load_jsonl(HERE / "tasks/nonsense.jsonl")
check("nonsense 15 items", len(non) == 15)
check("nonsense unique ids", len({t["id"] for t in non}) == 15)
check("nonsense has no answers", all("answer" not in t and "answer_any" not in t
                                     and "checks" not in t for t in non))
check("nonsense unscored", score_none(non[0], R("Tuesday feels chartreuse."))[0] is None)
check("nonsense in UNSCORED", "nonsense" in UNSCORED)
print("NONSENSE TESTS DONE")

# ---- thinking detection + autoscale --------------------------------------
from harness import detect_thinking
check("detect none", detect_thinking({}) == (None, 0))
check("detect off", detect_thinking(
    {"chat_template_kwargs": {"enable_thinking": False}}) == ("off", 0))
check("detect on", detect_thinking(
    {"chat_template_kwargs": {"enable_thinking": True}}) == ("on", 0))
check("detect max nested", detect_thinking(
    {"chat_template_kwargs": {"reasoning_effort": "max"}}) == ("max", 0))
check("detect high", detect_thinking({"reasoning_effort": "HIGH"}) == ("high", 0))
check("detect budget", detect_thinking(
    {"thinking": {"type": "enabled", "budget_tokens": 20000}}) == ("on", 20000))
print("THINKING DETECTION TESTS DONE")
