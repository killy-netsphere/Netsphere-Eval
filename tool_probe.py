#!/usr/bin/env python3
"""Probe an OpenAI-compatible endpoint for multi-turn tool-calling support.

Usage: tool_probe.py <base_url_with_/v1> <model> [extra_json]
Exit 0 = full multi-turn loop works. Prints a verdict per stage.
"""
import json, sys, urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2]
EXTRA = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_crate_weight",
        "description": "Return the weight in kilograms of a warehouse crate.",
        "parameters": {
            "type": "object",
            "properties": {"crate_id": {"type": "string",
                                        "description": "Crate identifier, e.g. A7"}},
            "required": ["crate_id"],
        },
    },
}]


def post(body, timeout=600):
    req = urllib.request.Request(
        BASE + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    msgs = [{"role": "user",
             "content": "What is the weight of crate A7? Use the provided tool to look "
                        "it up, then state the weight."}]
    body = {"model": MODEL, "messages": msgs, "tools": TOOLS,
            "tool_choice": "auto", "temperature": 0, "max_tokens": 2048}
    body.update(EXTRA)

    print("=== stage 1: does the model emit tool_calls? ===")
    try:
        r = post(body)
    except Exception as e:
        print("FAIL: request error:", e)
        return 2
    msg = r["choices"][0]["message"]
    finish = r["choices"][0].get("finish_reason")
    tcs = msg.get("tool_calls") or []
    print("finish_reason:", finish, "| tool_calls:", len(tcs))
    if not tcs:
        print("content:", repr((msg.get("content") or "")[:300]))
        print("VERDICT: NO TOOL CALLS -- endpoint/template does not emit tool_calls")
        return 1
    call = tcs[0]
    fname = call["function"]["name"]
    fargs = call["function"].get("arguments")
    print("tool name:", fname, "| raw args:", repr(fargs)[:200])
    try:
        parsed = json.loads(fargs) if isinstance(fargs, str) else fargs
    except json.JSONDecodeError as e:
        print("VERDICT: tool args are not valid JSON:", e)
        return 1
    print("parsed args:", parsed, "| crate_id ok:", parsed.get("crate_id"))

    print("\n=== stage 2: feed tool result back, expect final answer ===")
    msgs.append({"role": "assistant", "content": msg.get("content") or None,
                 "tool_calls": tcs})
    msgs.append({"role": "tool", "tool_call_id": call.get("id", "call_0"),
                 "name": fname, "content": json.dumps({"crate_id": "A7", "weight_kg": 63.5})})
    body2 = dict(body); body2["messages"] = msgs
    try:
        r2 = post(body2)
    except Exception as e:
        print("FAIL: second-turn request error:", e)
        return 2
    m2 = r2["choices"][0]["message"]
    out = (m2.get("content") or "").strip()
    print("finish_reason:", r2["choices"][0].get("finish_reason"))
    print("final content:", repr(out[:400]))
    ok = "63.5" in out or "63,5" in out
    print("\nVERDICT:", "FULL MULTI-TURN TOOL LOOP WORKS" if ok
          else "loop completed but did not report the tool value (check template)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
