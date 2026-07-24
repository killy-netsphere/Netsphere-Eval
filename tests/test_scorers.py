"""Pytest test suite for Netsphere-Eval scorers and utilities."""
import json
import sys
from pathlib import Path

import pytest

# Ensure we can import harness from the project root
HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from harness import (
    UNSCORED,
    _to_number,
    _typecheck,
    answers_match,
    detect_thinking,
    extract_code,
    extract_final,
    gen_longctx,
    load_jsonl,
    score_bizarre,
    score_code,
    score_instruct,
    score_json,
    score_math,
    score_none,
    strip_fences,
    strip_think,
    walk_path,
)

R = lambda text: {
    "content": strip_think(text),
    "raw_content": text,
    "latency": 0.1,
    "completion_tokens": 10,
    "finish": "stop",
}


# ---- harness utilities ----------------------------------------------------

class TestStripThink:
    def test_removes_think_block(self):
        assert strip_think("<think>reasoning</think>answer") == "answer"

    def test_removes_multiline_think(self):
        assert strip_think("<think>\nstep 1\nstep 2\n</think>done") == "done"

    def test_no_think_block(self):
        assert strip_think("plain answer") == "plain answer"

    def test_empty_input(self):
        assert strip_think("") == ""
        assert strip_think(None) == ""


class TestStripFences:
    def test_python_fence(self):
        assert strip_fences("```python\nprint(1)\n```") == "print(1)"

    def test_json_fence(self):
        assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_no_fence(self):
        assert strip_fences("hello") == "hello"

    def test_language_suffix(self):
        assert strip_fences("```py\nx=1\n```") == "x=1"


class TestExtractFinal:
    def test_basic(self):
        assert extract_final("FINAL ANSWER: 42") == "42"

    def test_with_context(self):
        text = "Working... FINAL ANSWER: 649\nThe answer is 649"
        assert extract_final(text) == "649"

    def test_no_final_answer(self):
        assert extract_final("no match") is None

    def test_case_insensitive(self):
        assert extract_final("Final Answer: yes") == "yes"

    def test_last_takes_precedence(self):
        assert extract_final("FINAL ANSWER: 1\nFINAL ANSWER: 2") == "2"


class TestExtractCode:
    def test_python_block(self):
        text = "```python\ndef f(): pass\n```"
        assert "def f(): pass" in extract_code(text)

    def test_last_block_wins(self):
        text = "```python\na=1\n```\n```python\na=2\n```"
        assert extract_code(text).rstrip() == "a=2"

    def test_no_block_returns_text(self):
        assert extract_code("plain") == "plain"


class TestToNumber:
    def test_integer(self):
        assert _to_number("42") == 42.0

    def test_fraction(self):
        assert _to_number("3/4") == 0.75

    def test_decimal(self):
        assert _to_number("3.14") == 3.14

    def test_dollar_sign(self):
        assert _to_number("$100") == 100.0

    def test_comma(self):
        assert _to_number("8,500.80") == 8500.8

    def test_invalid(self):
        assert _to_number("abc") is None

    def test_trailing_period(self):
        assert _to_number("649.") == 649.0


class TestWalkPath:
    def test_root(self):
        assert walk_path({"a": 1}, ".") == {"a": 1}

    def test_simple_key(self):
        assert walk_path({"a": {"b": 2}}, "a.b") == 2

    def test_list_index(self):
        assert walk_path({"a": [10, 20]}, "a.1") == 20

    def test_missing_key(self):
        with pytest.raises(KeyError):
            walk_path({"a": 1}, "b")


class TestTypecheck:
    def test_int(self):
        assert _typecheck(42, "int")
        assert not _typecheck(True, "int")
        assert not _typecheck("42", "int")

    def test_number(self):
        assert _typecheck(42, "number")
        assert _typecheck(3.14, "number")
        assert not _typecheck("3", "number")

    def test_str(self):
        assert _typecheck("hello", "str")
        assert not _typecheck(42, "str")

    def test_bool(self):
        assert _typecheck(True, "bool")
        assert not _typecheck(1, "bool")

    def test_array(self):
        assert _typecheck([1, 2], "array")
        assert not _typecheck("list", "array")

    def test_object(self):
        assert _typecheck({"a": 1}, "object")
        assert not _typecheck([], "object")

    def test_unknown_type(self):
        assert not _typecheck(42, "foobar")


class TestDetectThinking:
    def test_no_data(self):
        assert detect_thinking({}) == (None, 0)

    def test_disabled(self):
        assert detect_thinking({"chat_template_kwargs": {"enable_thinking": False}}) == ("off", 0)

    def test_enabled(self):
        assert detect_thinking({"chat_template_kwargs": {"enable_thinking": True}}) == ("on", 0)

    def test_max_reasoning(self):
        assert detect_thinking({"chat_template_kwargs": {"reasoning_effort": "max"}}) == ("max", 0)

    def test_high_effort(self):
        assert detect_thinking({"reasoning_effort": "HIGH"}) == ("high", 0)

    def test_with_budget(self):
        assert detect_thinking({"thinking": {"type": "enabled", "budget_tokens": 20000}}) == ("on", 20000)


# ---- answers_match ---------------------------------------------------------

class TestAnswersMatch:
    def test_exact_match(self):
        assert answers_match("42", "42")

    def test_case_insensitive(self):
        assert answers_match("Hello", "HELLO")

    def test_numeric_vs_fraction(self):
        assert answers_match("8500.80", "42504/5")

    def test_dollar_comma(self):
        assert answers_match("$8,500.80", "42504/5")

    def test_unreduced_fraction(self):
        assert answers_match("103776/2598960", "2162/54145")

    def test_trailing_period(self):
        assert answers_match("649.", "649")

    def test_bold_markdown(self):
        assert answers_match("**649**", "649")

    def test_wrong_number_rejected(self):
        assert not answers_match("650", "649")

    def test_leading_zero_code(self):
        assert answers_match("043501", "43501")


# ---- math scorer ------------------------------------------------------------

class TestScoreMath:
    @pytest.fixture
    def tasks(self):
        return load_jsonl(HERE / "tasks/math.jsonl")

    def test_math_pass(self, tasks):
        m01 = next(t for t in tasks if t["id"] == "m01")
        ok, why = score_math(m01, R("thinking...\nFINAL ANSWER: 649"))
        assert ok, why

    def test_math_pass_with_think_block(self, tasks):
        m01 = next(t for t in tasks if t["id"] == "m01")
        ok, why = score_math(m01, R("<think>7^4=401...</think>FINAL ANSWER: 649"))
        assert ok, why

    def test_math_fail_wrong(self, tasks):
        m01 = next(t for t in tasks if t["id"] == "m01")
        ok, why = score_math(m01, R("FINAL ANSWER: 343"))
        assert not ok
        assert "got=" in why

    def test_math_fail_no_final_line(self, tasks):
        m01 = next(t for t in tasks if t["id"] == "m01")
        ok, why = score_math(m01, R("The answer is 649"))
        assert not ok
        assert "no_final_answer_line" in why

    def test_math_fraction_decimal_equiv(self, tasks):
        m20 = next(t for t in tasks if t["id"] == "m20")
        ok, why = score_math(m20, R("FINAL ANSWER: $8500.80"))
        assert ok, why

    def test_mock_pass(self):
        ok, why = score_math({}, {"_mock_pass": True})
        assert ok and why == "mock"

    def test_mock_fail(self):
        ok, why = score_math({}, {"_mock_pass": False})
        assert not ok


# ---- code scorer ------------------------------------------------------------

class TestScoreCode:
    @pytest.fixture
    def tasks(self):
        return load_jsonl(HERE / "tasks/code.jsonl")

    def test_code_pass(self, tasks):
        c01 = next(t for t in tasks if t["id"] == "c01")
        good = (
            "Here you go:\n```python\n"
            "def rle(s):\n"
            "    if not s: return ''\n"
            "    out=[]; cur=s[0]; n=1\n"
            "    for ch in s[1:]:\n"
            "        if ch==cur: n+=1\n"
            "        else: out.append(cur+str(n)); cur=ch; n=1\n"
            "    out.append(cur+str(n))\n"
            "    return ''.join(out)\n```"
        )
        ok, why = score_code(c01, R(good))
        assert ok, why

    def test_code_fail(self, tasks):
        c01 = next(t for t in tasks if t["id"] == "c01")
        bad = "```python\ndef rle(s):\n    return s\n```"
        ok, why = score_code(c01, R(bad))
        assert not ok

    def test_code_fail_no_entry(self, tasks):
        c01 = next(t for t in tasks if t["id"] == "c01")
        ok, why = score_code(c01, R("```python\nx=1\n```"))
        assert not ok
        assert "entry_point_missing" in why

    def test_code_timeout(self, tasks):
        c01 = next(t for t in tasks if t["id"] == "c01")
        hang = "```python\ndef rle(s):\n    while True: pass\n```"
        ok, why = score_code(c01, R(hang))
        assert not ok
        assert why == "timeout"


# ---- JSON/tool scorer -------------------------------------------------------

class TestScoreJson:
    @pytest.fixture
    def tasks(self):
        return load_jsonl(HERE / "tasks/json_tool.jsonl")

    def test_json_pass(self, tasks):
        j01 = next(t for t in tasks if t["id"] == "j01")
        good_j = json.dumps({
            "tool": "schedule_snapshot",
            "args": {"vm": "pg-primary", "retain_days": 14, "quiesce": True},
        })
        ok, why = score_json(j01, R(good_j))
        assert ok, why

    def test_json_pass_fenced(self, tasks):
        j01 = next(t for t in tasks if t["id"] == "j01")
        good_j = json.dumps({
            "tool": "schedule_snapshot",
            "args": {"vm": "pg-primary", "retain_days": 14, "quiesce": True},
        })
        ok, why = score_json(j01, R("```json\n" + good_j + "\n```"))
        assert ok, why

    def test_json_fail_str_int(self, tasks):
        j01 = next(t for t in tasks if t["id"] == "j01")
        bad = json.dumps({
            "tool": "schedule_snapshot",
            "args": {"vm": "pg-primary", "retain_days": "14", "quiesce": True},
        })
        ok, why = score_json(j01, R(bad))
        assert not ok

    def test_json_fail_prose(self, tasks):
        j01 = next(t for t in tasks if t["id"] == "j01")
        ok, why = score_json(j01, R("Sure! " + json.dumps({
            "tool": "schedule_snapshot",
            "args": {"vm": "pg-primary", "retain_days": 14, "quiesce": True},
        })))
        assert not ok

    def test_json_escaped_newline_quotes(self, tasks):
        j05 = next(t for t in tasks if t["id"] == "j05")
        good5 = json.dumps({"message": 'He said "run it" twice.\nThen stopped.'})
        ok, why = score_json(j05, R(good5))
        assert ok, why

    def test_json_extra_keys_rejected(self, tasks):
        j11 = next(t for t in tasks if t["id"] == "j11")
        extra = json.dumps({53: 2809, 59: 3481, 61: 3721, 67: 4489})
        ok, why = score_json(j11, R(extra))
        assert not ok

    def test_json_bool_not_int(self, tasks):
        j01 = next(t for t in tasks if t["id"] == "j01")
        bad = json.dumps({
            "tool": "schedule_snapshot",
            "args": {"vm": "pg-primary", "retain_days": True, "quiesce": True},
        })
        ok, why = score_json(j01, R(bad))
        assert not ok


# ---- instruct scorer --------------------------------------------------------

class TestScoreInstruct:
    @pytest.fixture
    def tasks(self):
        return load_jsonl(HERE / "tasks/instruct.jsonl")

    def test_i03_pass(self, tasks):
        i = lambda tid: next(t for t in tasks if t["id"] == tid)
        ok, why = score_instruct(i("i03"), R("Au"))
        assert ok, why

    def test_i03_fail(self, tasks):
        i = lambda tid: next(t for t in tasks if t["id"] == tid)
        ok, why = score_instruct(i("i03"), R("Gold is Au"))
        assert not ok

    def test_i05_pass(self, tasks):
        i = lambda tid: next(t for t in tasks if t["id"] == tid)
        ok, why = score_instruct(i("i05"), R("stable stable stable stable stable"))
        assert ok, why

    def test_i05_fail_count(self, tasks):
        i = lambda tid: next(t for t in tasks if t["id"] == tid)
        ok, why = score_instruct(i("i05"), R("stable stable stable stable"))
        assert not ok

    def test_i08_pass_lipogram(self, tasks):
        i = lambda tid: next(t for t in tasks if t["id"] == tid)
        ok, why = score_instruct(
            i("i08"),
            R("A crimson orb sinks low, kissing far hills. Night winds hum "
              "soft songs of vanishing light."),
        )
        assert ok, why

    def test_i08_fail_lipogram(self, tasks):
        i = lambda tid: next(t for t in tasks if t["id"] == tid)
        ok, why = score_instruct(
            i("i08"),
            R("The sun sets slowly. Evening arrives quietly."),
        )
        assert not ok

    def test_i10_pass(self, tasks):
        i = lambda tid: next(t for t in tasks if t["id"] == tid)
        ok, why = score_instruct(i("i10"), R("150000"))
        assert ok, why

    def test_i10_fail_units(self, tasks):
        i = lambda tid: next(t for t in tasks if t["id"] == tid)
        ok, why = score_instruct(i("i10"), R("150000 ms"))
        assert not ok

    def test_i13_pass(self, tasks):
        i = lambda tid: next(t for t in tasks if t["id"] == tid)
        ok, why = score_instruct(i("i13"), R("Callisto,Europa,Ganymede,Io"))
        assert ok, why

    def test_i13_fail_spaces(self, tasks):
        i = lambda tid: next(t for t in tasks if t["id"] == tid)
        ok, why = score_instruct(i("i13"), R("Callisto, Europa, Ganymede, Io"))
        assert not ok

    def test_i14_pass(self, tasks):
        i = lambda tid: next(t for t in tasks if t["id"] == tid)
        ok, why = score_instruct(i("i14"), R("BEGIN\nEND"))
        assert ok, why

    def test_i01_pass_word_count(self, tasks):
        i = lambda tid: next(t for t in tasks if t["id"] == tid)
        ok, why = score_instruct(
            i("i01"),
            R("NVMe attaches flash over PCIe lanes. It cuts protocol overhead "
              "compared with SATA. Parallel queues keep many operations in flight."),
        )
        assert ok, why

    def test_i01_fail_word_count(self, tasks):
        i = lambda tid: next(t for t in tasks if t["id"] == tid)
        ok, why = score_instruct(i("i01"), R("One. Two. Three. Four."))
        assert not ok

    def test_i04_pass_sentence_count(self, tasks):
        i = lambda tid: next(t for t in tasks if t["id"] == tid)
        ok, why = score_instruct(
            i("i04"),
            R("A hypervisor is software that creates and runs virtual machines "
              "by abstracting physical hardware into shared pools of compute, "
              "memory, storage, and networking. It schedules guest access to "
              "real resources, enforces isolation between workloads, and lets "
              "one physical server safely host many independent operating "
              "systems at the same time today."),
        )
        assert ok, why


# ---- bizarre scorer ---------------------------------------------------------

class TestScoreBizarre:
    @pytest.fixture
    def tasks(self):
        return load_jsonl(HERE / "tasks/bizarre.jsonl")

    def test_b07_pass_2nd(self, tasks):
        b07 = next(t for t in tasks if t["id"] == "b07")
        ok, why = score_bizarre(b07, R("FINAL ANSWER: 2nd"))
        assert ok, why

    def test_b07_pass_second(self, tasks):
        b07 = next(t for t in tasks if t["id"] == "b07")
        ok, why = score_bizarre(b07, R("FINAL ANSWER: Second"))
        assert ok, why

    def test_b07_fail_first(self, tasks):
        b07 = next(t for t in tasks if t["id"] == "b07")
        ok, why = score_bizarre(b07, R("FINAL ANSWER: first"))
        assert not ok

    def test_b10_microwave_pass(self, tasks):
        b10 = next(t for t in tasks if t["id"] == "b10")
        ok, why = score_bizarre(b10, R("beep beep beep"))
        assert ok, why

    def test_b10_microwave_fail(self, tasks):
        b10 = next(t for t in tasks if t["id"] == "b10")
        ok, why = score_bizarre(b10, R("Beep! Beep! Beep!"))
        assert not ok

    def test_b13_decimal_pass(self, tasks):
        b13 = next(t for t in tasks if t["id"] == "b13")
        ok, why = score_bizarre(b13, R("FINAL ANSWER: 8.40"))
        assert ok, why

    def test_b18_strawberry(self, tasks):
        b18 = next(t for t in tasks if t["id"] == "b18")
        assert b18["answer"] == "9"

    def test_b19_parrot_count(self, tasks):
        b19 = next(t for t in tasks if t["id"] == "b19")
        assert b19["answer"] == str("purple parrot territory".count("r"))

    def test_b20_pass_reversed(self, tasks):
        b20 = next(t for t in tasks if t["id"] == "b20")
        ok, why = score_bizarre(b20, R("FINAL ANSWER: money me owes moon the"))
        assert ok, why

    def test_b11_palindrome(self, tasks):
        b11 = next(t for t in tasks if t["id"] == "b11")
        assert b11["answer"] == "4994"


# ---- grounded scorer (uses score_math) --------------------------------------

class TestScoreGrounded:
    @pytest.fixture
    def tasks(self):
        return load_jsonl(HERE / "tasks/grounded.jsonl")

    def test_g04_pass(self, tasks):
        g04 = next(t for t in tasks if t["id"] == "g04")
        ok, why = score_math(g04, R("FINAL ANSWER: 7719"))
        assert ok, why

    def test_g03_trap_pass(self, tasks):
        g03 = next(t for t in tasks if t["id"] == "g03")
        ok, why = score_math(g03, R("FINAL ANSWER: NOT IN CONTEXT"))
        assert ok, why

    def test_g03_trap_pass_lowercase(self, tasks):
        g03 = next(t for t in tasks if t["id"] == "g03")
        ok, why = score_math(g03, R("FINAL ANSWER: not in context"))
        assert ok, why

    def test_g03_trap_fail_hallucination(self, tasks):
        g03 = next(t for t in tasks if t["id"] == "g03")
        ok, why = score_math(g03, R("FINAL ANSWER: 18.2 GHz"))
        assert not ok

    def test_g14_ratio_alt(self, tasks):
        g14 = next(t for t in tasks if t["id"] == "g14")
        ok, why = score_math(g14, R("FINAL ANSWER: 1 to 9"))
        assert ok, why


# ---- nonsense (unscored) ----------------------------------------------------

class TestNonsense:
    @pytest.fixture
    def tasks(self):
        return load_jsonl(HERE / "tasks/nonsense.jsonl")

    def test_nonsense_20_items(self, tasks):
        assert len(tasks) == 20
        assert len({t["id"] for t in tasks}) == 20

    def test_nonsense_no_answers(self, tasks):
        for t in tasks:
            assert "answer" not in t
            assert "answer_any" not in t
            assert "checks" not in t

    def test_nonsense_unscored(self, tasks):
        ok, why = score_none(tasks[0], R("Tuesday feels chartreuse."))
        assert ok is None

    def test_nonsense_in_unscored(self):
        assert "nonsense" in UNSCORED


# ---- longctx generation -----------------------------------------------------

class TestGenLongctx:
    def test_count(self):
        lt = gen_longctx(seed=7, n_items=12, ctx_tokens=2000)
        assert len(lt) == 12

    def test_unique_ids(self):
        lt = gen_longctx(seed=7, n_items=12, ctx_tokens=2000)
        assert len({t["id"] for t in lt}) == 12

    def test_answer_in_haystack(self):
        lt = gen_longctx(seed=7, n_items=12, ctx_tokens=2000)
        for t in lt:
            assert t["answer"] in t["prompt"]

    def test_deterministic(self):
        lt1 = gen_longctx(seed=7, n_items=12, ctx_tokens=2000)
        lt2 = gen_longctx(seed=7, n_items=12, ctx_tokens=2000)
        assert json.dumps(lt1) == json.dumps(lt2)

    def test_count_answer_correct(self):
        lt = gen_longctx(seed=7, n_items=12, ctx_tokens=2000)
        cnt = next(t for t in lt if t["id"].startswith("l_cnt"))
        hay = cnt["prompt"].split("=== DATABASE ===")[1].split("=== END ===")[0]
        occ = hay.count("dept Cryogenics")
        assert str(occ) == cnt["answer"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
