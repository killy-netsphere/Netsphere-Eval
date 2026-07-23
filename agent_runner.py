#!/usr/bin/env python3
"""netterm-eval v2 -- multi-turn agentic tool-loop runner and trace scorer.

Plugs into the existing harness. A task is a dict:
    {"id": str,
     "prompt": str,                  # user instruction; should require tool use
     "tools": [ ...openai tool schemas... ],
     "impl": {name: callable},       # deterministic pure-python implementations
     "answer": <gold> | "answer_any": [<gold>, ...],
     "trace_checks": [ {...}, ... ]} # programmatic predicates over the call trace

Scoring = correct final answer AND all trace_checks satisfied. Both are objective;
no judge model is involved, consistent with the v1 design rules.

Verified working against DeepSeek-V4-Flash (vLLM, deepseek_v4 tool parser) and
GLM-5.2 Q3 (llama.cpp --jinja) on 2026-07-22.
"""
import json
import re

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FINAL_RE = re.compile(r"FINAL ANSWER\s*:\s*(.+)", re.IGNORECASE)

MAX_TURNS = 18   # T05 full exploration = 14 turns at one call/turn; headroom above that
MAX_TOOL_CALLS = 30


def strip_think(text):
    return THINK_RE.sub("", text or "").strip()


def extract_final(text):
    hits = FINAL_RE.findall(text or "")
    return hits[-1].strip() if hits else None


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def run_tool_loop(chat_fn, task, seed=7, max_turns=MAX_TURNS):
    """chat_fn(messages, tools, seed) -> parsed OpenAI 'message' dict.

    Returns a record with the final text, the full ordered tool-call trace, and
    the message list (kept for the transcript so failures are auditable).
    """
    messages = [{"role": "user", "content": task["prompt"]}]
    tools = task.get("tools") or []
    impl = task.get("impl") or {}
    trace = []
    final_text = None
    stop_reason = "max_turns"
    turns = 0

    for turns in range(1, max_turns + 1):
        msg = chat_fn(messages, tools, seed)
        calls = msg.get("tool_calls") or []

        if not calls:
            final_text = strip_think(msg.get("content") or "")
            stop_reason = "final"
            break

        # record the assistant turn verbatim so the model sees its own calls
        messages.append({"role": "assistant",
                         "content": msg.get("content") or None,
                         "tool_calls": calls})

        for tc in calls:
            if len(trace) >= MAX_TOOL_CALLS:
                stop_reason = "call_budget"
                break
            fn_block = tc.get("function") or {}
            name = fn_block.get("name") or "<missing>"
            raw = fn_block.get("arguments")
            bad_args = False
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                if not isinstance(args, dict):
                    args, bad_args = {}, True
            except (json.JSONDecodeError, TypeError):
                args, bad_args = {}, True

            fn = impl.get(name)
            if bad_args:
                result = {"error": "arguments were not a valid JSON object"}
            elif fn is None:
                result = {"error": "unknown tool: %s" % name}
            else:
                try:
                    result = fn(**args)
                except TypeError as e:
                    result = {"error": "bad arguments: %s" % e}
                except Exception as e:            # tool bugs must not kill the run
                    result = {"error": "%s: %s" % (type(e).__name__, e)}

            is_err = isinstance(result, dict) and "error" in result
            trace.append({"tool": name, "args": args,
                          "result": result, "error": is_err})
            messages.append({"role": "tool",
                             "tool_call_id": tc.get("id") or "call_%d" % len(trace),
                             "name": name,
                             "content": json.dumps(result, ensure_ascii=False)})
        if stop_reason == "call_budget":
            break

    return {"final": final_text, "trace": trace, "turns": turns,
            "stop_reason": stop_reason, "messages": messages}


# ---------------------------------------------------------------------------
# trace predicates
# ---------------------------------------------------------------------------

def check_trace(trace, checks):
    """All predicates must hold. Returns (ok, reason)."""
    names = [t["tool"] for t in trace]
    for c in checks or []:
        op = c.get("op")
        if op == "called":
            if c["tool"] not in names:
                return False, "never called %s" % c["tool"]
        elif op == "not_called":
            if c["tool"] in names:
                return False, "called forbidden tool %s" % c["tool"]
        elif op == "min_calls":
            if len(trace) < c["n"]:
                return False, "only %d tool calls, need >= %d" % (len(trace), c["n"])
        elif op == "max_calls":
            if len(trace) > c["n"]:
                return False, "%d tool calls, budget %d" % (len(trace), c["n"])
        elif op == "order":
            if c["before"] not in names or c["after"] not in names:
                return False, "order check: missing %s or %s" % (c["before"], c["after"])
            if names.index(c["before"]) > names.index(c["after"]):
                return False, "%s must precede %s" % (c["before"], c["after"])
        elif op == "recovered_from_error":
            tool = c["tool"]
            errs = [i for i, t in enumerate(trace) if t["tool"] == tool and t["error"]]
            oks = [i for i, t in enumerate(trace) if t["tool"] == tool and not t["error"]]
            if not errs:
                return False, "no error was ever triggered on %s" % tool
            if not any(o > e for e in errs for o in oks):
                return False, "never recovered after error on %s" % tool
        elif op == "called_with":
            hit = any(t["tool"] == c["tool"] and
                      all(t["args"].get(k) == v for k, v in c["args"].items())
                      for t in trace)
            if not hit:
                return False, "%s never called with %s" % (c["tool"], c["args"])
        elif op == "no_error_calls":
            if any(t["error"] for t in trace):
                return False, "trace contains failed tool calls"
        else:
            return False, "unknown trace op: %r" % op
    return True, ""


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_agent(task, rec, answers_match):
    """answers_match is injected from the harness so matching stays identical."""
    if rec["final"] is None:
        return False, "no final answer (stop=%s, turns=%d, calls=%d)" % (
            rec["stop_reason"], rec["turns"], len(rec["trace"]))

    ok_trace, why = check_trace(rec["trace"], task.get("trace_checks"))

    ans = extract_final(rec["final"])
    if ans is None:
        # tolerate a bare final answer when the model skipped the FINAL ANSWER line
        ans = rec["final"].strip().splitlines()[-1].strip() if rec["final"].strip() else ""
    golds = task.get("answer_any") or [task.get("answer")]
    ok_ans = any(answers_match(ans, g) for g in golds)

    if not ok_ans and not ok_trace:
        return False, "answer+trace: got=%r want=%r; %s" % (ans, golds, why)
    if not ok_ans:
        return False, "answer: got=%r want=%r" % (ans, golds)
    if not ok_trace:
        return False, "trace: %s" % why
    return True, ""
