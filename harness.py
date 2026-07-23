#!/usr/bin/env python3
"""
netterm-eval harness: paired A/B evaluation of two OpenAI-compatible endpoints.

Zero third-party dependencies (stdlib only).

  python3 harness.py run \
      --a-name glm52-q3  --a-url http://127.0.0.1:8080/v1 --a-model glm-5.2 \
      --b-name v4-flash  --b-url http://127.0.0.1:8000/v1 --b-model deepseek-v4-flash \
      --out results/run1

  python3 harness.py report --out results/run1

Categories: math, code, json_tool, instruct, longctx (generated), rag (optional).
Scoring is objective everywhere except rag, which uses a configurable judge.
WARNING: the code category EXECUTES model-generated Python. Run in a container
or throwaway user account.
"""
import argparse, hashlib, json, math, os, random, re, statistics, string, subprocess
import shutil, sys, tempfile, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
FINAL_RE = re.compile(r"FINAL ANSWER\s*:\s*(.+)", re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# ---------------------------------------------------------------------------
# HTTP / model client
# ---------------------------------------------------------------------------

def http_post_json(url, payload, headers, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def http_get_json(url, timeout=60):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def strip_think(text):
    if not text:
        return ""
    t = THINK_RE.sub("", text)
    # orphaned closing tag: some reasoning parsers (poolside_v1) leave a stray
    # </think> at the start of content after extracting the reasoning block
    t = re.sub(r"^\s*</think>", "", t)
    return t.strip()


def strip_fences(text):
    t = text.strip()
    m = re.match(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```$", t, re.DOTALL)
    return m.group(1).strip() if m else t


class Endpoint:
    def __init__(self, name, url, model, temp=0.0, key=None, extra=None,
                 max_tokens=6144, timeout=900, mock=None):
        self.name, self.url, self.model = name, url.rstrip("/"), model
        self.temp, self.key = temp, key
        self.extra = extra or {}
        self.max_tokens, self.timeout = max_tokens, timeout
        self.mock = mock  # None, or float pass-probability for smoke tests

    def chat(self, prompt, seed=0, max_tokens=None, temp=None):
        if self.mock is not None:
            return self._mock(prompt, seed)
        body = {"model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temp if temp is None else temp,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                "seed": seed}
        body.update(self.extra)
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        t0 = time.time()
        resp = http_post_json(self.url + "/chat/completions", body, headers, self.timeout)
        dt = time.time() - t0
        ch = resp.get("choices", [{}])[0]
        msg = ch.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        if not reasoning:
            inl = re.findall(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
            reasoning = "\n".join(inl).strip()
        if not strip_think(content):
            content = msg.get("reasoning_content") or content
        usage = resp.get("usage", {}) or {}
        return {"content": strip_think(content),
                "raw_content": content,
                "reasoning": reasoning,
                "finish": ch.get("finish_reason"),
                "latency": dt,
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens")}

    def chat_message(self, messages, tools=None, seed=0, max_tokens=None, temp=None):
        """Multi-turn / tool-calling entry: returns the raw assistant message dict
        (tool_calls intact) plus finish/usage. Used by the agentic category."""
        body = {"model": self.model, "messages": messages,
                "temperature": self.temp if temp is None else temp,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                "seed": seed}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        body.update(self.extra)
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        resp = http_post_json(self.url + "/chat/completions", body, headers, self.timeout)
        ch = resp.get("choices", [{}])[0]
        msg = dict(ch.get("message", {}) or {})
        msg["_finish"] = ch.get("finish_reason")
        msg["_usage"] = resp.get("usage", {}) or {}
        return msg

    def _mock(self, prompt, seed):
        h = int(hashlib.sha256(f"{self.name}|{prompt[:200]}|{prompt[-200:]}|{seed}".encode()).hexdigest(), 16)
        ok = (h % 1000) / 1000.0 < self.mock
        return {"content": "", "raw_content": "", "finish": "stop", "latency": 0.01,
                "completion_tokens": 50, "prompt_tokens": 100, "_mock_pass": ok}


# ---------------------------------------------------------------------------
# Answer matching (math / longctx)
# ---------------------------------------------------------------------------

def _to_number(s):
    s = s.strip().rstrip(".").replace("$", "").replace(",", "").replace(" ", "")
    m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if m:
        d = int(m.group(2))
        if d != 0:
            return int(m.group(1)) / d
    try:
        return float(s)
    except ValueError:
        return None


def answers_match(got, gold):
    g1, g2 = got.strip(), str(gold).strip()
    g1 = re.sub(r"^[\*\s`\"']+|[\*\s`\"'.]+$", "", g1)
    if g1.casefold() == g2.casefold():
        return True
    n1, n2 = _to_number(g1), _to_number(g2)
    if n1 is not None and n2 is not None:
        if n2 == 0:
            return abs(n1) < 1e-9
        return abs(n1 - n2) <= max(1e-9, 1e-6 * abs(n2))
    return re.sub(r"\s+", "", g1).casefold() == re.sub(r"\s+", "", g2).casefold()


def extract_final(text):
    hits = FINAL_RE.findall(text)
    return hits[-1].strip() if hits else None


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def score_math(task, r):
    if "_mock_pass" in r:
        return r["_mock_pass"], "mock"
    ans = extract_final(r["content"])
    if ans is None:
        return False, "no_final_answer_line"
    golds = task.get("answer_any") or [task["answer"]]
    ok = any(answers_match(ans, g) for g in golds)
    return ok, ("" if ok else f"got={ans!r} want={golds!r}")


def score_bizarre(task, r):
    if "checks" in task:
        return score_instruct(task, r)
    return score_math(task, r)


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else text


def score_code(task, r):
    if "_mock_pass" in r:
        return r["_mock_pass"], "mock"
    code = extract_code(r["content"])
    if task["entry_point"] not in code:
        return False, "entry_point_missing"
    src = code + "\n\n" + task["tests"] + "\nprint('__PASS__')\n"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "cand.py")
        with open(p, "w") as f:
            f.write(src)
        try:
            pr = subprocess.run([sys.executable, "-I", p], capture_output=True,
                                text=True, timeout=25, cwd=td)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if pr.returncode == 0 and "__PASS__" in pr.stdout:
        return True, ""
    err = (pr.stderr or pr.stdout).strip().splitlines()
    return False, (err[-1][:200] if err else "nonzero_exit")


def walk_path(obj, path):
    if path == ".":
        return obj
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(path)
            cur = cur[part]
        elif isinstance(cur, list):
            cur = cur[int(part)]
        else:
            raise KeyError(path)
    return cur


def _typecheck(v, tname):
    if tname == "int":
        return isinstance(v, int) and not isinstance(v, bool)
    if tname == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if tname == "str":
        return isinstance(v, str)
    if tname == "bool":
        return isinstance(v, bool)
    if tname == "array":
        return isinstance(v, list)
    if tname == "object":
        return isinstance(v, dict)
    return False


def score_json(task, r):
    if "_mock_pass" in r:
        return r["_mock_pass"], "mock"
    text = strip_fences(r["content"])
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"invalid_json:{e.msg}"
    for chk in task["checks"]:
        op, val, path = chk["op"], chk.get("value"), chk.get("path", ".")
        try:
            v = walk_path(obj, path)
        except (KeyError, IndexError, ValueError, TypeError):
            return False, f"missing_path:{path}"
        if op == "eq" and v != val:
            return False, f"{path}: got={v!r} want={val!r}"
        if op == "type" and not _typecheck(v, val):
            return False, f"{path}: wrong type ({type(v).__name__}, want {val})"
        if op == "approx" and not (isinstance(v, (int, float))
                                   and abs(float(v) - val) <= 0.011):
            return False, f"{path}: got={v!r} !~ {val}"
        if op == "keys_exactly" and (not isinstance(v, dict)
                                     or sorted(v.keys()) != sorted(val)):
            return False, f"{path}: keys={sorted(v.keys()) if isinstance(v, dict) else '?'}"
        if op == "len_eq" and len(v) != val:
            return False, f"{path}: len={len(v)} want {val}"
        if op == "nonempty" and not v:
            return False, f"{path}: empty"
        if op == "in" and v not in val:
            return False, f"{path}: {v!r} not in {val}"
    return True, ""


def score_instruct(task, r):
    if "_mock_pass" in r:
        return r["_mock_pass"], "mock"
    text = r["content"].strip()
    lines = [l.rstrip() for l in text.splitlines()]
    for chk in task["checks"]:
        op, val = chk["op"], chk.get("value")
        if op == "exact_sentences":
            n = len(re.findall(r"[.!?]+(?=\s|$)", text))
            if n != val:
                return False, f"sentences={n} want {val}"
        elif op == "must_exclude_word":
            if re.search(rf"\b{re.escape(val)}\b", text, re.IGNORECASE):
                return False, f"contains forbidden word {val!r}"
        elif op == "must_include_i":
            if val.lower() not in text.lower():
                return False, f"missing {val!r}"
        elif op == "line_count":
            if len(lines) != val:
                return False, f"lines={len(lines)} want {val}"
        elif op == "line_regex_all":
            for l in lines:
                if not re.match(val, l):
                    return False, f"line fails regex: {l[:60]!r}"
        elif op == "regex":
            if not re.search(val, text):
                return False, f"regex miss: {val[:50]}"
        elif op == "max_words":
            if len(text.split()) > val:
                return False, f"words={len(text.split())} > {val}"
        elif op == "word_count_between":
            n = len(text.split())
            if not (val[0] <= n <= val[1]):
                return False, f"words={n} not in {val}"
        elif op == "no_letter":
            if val in text.lower():
                return False, f"contains letter {val!r}"
        elif op == "json_eq":
            try:
                if json.loads(strip_fences(text)) != val:
                    return False, "json != expected"
            except json.JSONDecodeError:
                return False, "invalid_json"
        elif op == "exact_text":
            if "\n".join(lines) != val:
                return False, "text mismatch"
    return True, ""


# ---------------------------------------------------------------------------
# Long-context task generation (seeded, self-verifying by construction)
# ---------------------------------------------------------------------------

FIRST = ["Avery","Blake","Casey","Devon","Ellis","Finley","Greer","Harper","Indra",
         "Jules","Kiran","Logan","Mercer","Noor","Orin","Palmer","Quinn","Rowan",
         "Sasha","Tatum","Uma","Vale","Wren","Xiomara","Yael","Zephyr","Marlow",
         "Sutton","Briar","Callum","Dara","Emrys","Fallon","Gideon","Halston"]
LAST = ["Adler","Boone","Calloway","Drummond","Everly","Fontaine","Garrick","Hale",
        "Ibarra","Jennings","Kessler","Lockhart","Mercado","Nakano","Oakes","Pruitt",
        "Quill","Rhodes","Sable","Thorne","Ursini","Vance","Whitlock","Xanthos",
        "Yates","Zimmer","Abbott","Beckett","Croft","Dunmore","Ellery","Falk"]
DEPTS = ["Fabrication","Logistics","Telemetry","Procurement","Diagnostics",
         "Compliance","Metallurgy","Archives","Calibration","Dispatch"]
RARE_DEPT = "Cryogenics"
PROJ_ADJ = ["Amber","Cobalt","Crimson","Onyx","Ivory","Jade","Umber","Slate",
            "Saffron","Indigo","Vermilion","Cedar"]
PROJ_NOUN = ["Falcon","Lantern","Compass","Anvil","Beacon","Harbor","Summit",
             "Glacier","Meridian","Foundry","Citadel","Orchard"]


def gen_longctx(seed, n_items, ctx_tokens):
    rng = random.Random(seed)
    names = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(names)
    projects = [f"{a} {n}" for a in PROJ_ADJ for n in PROJ_NOUN]
    rng.shuffle(projects)

    def badge():
        return f"{rng.choice(string.ascii_uppercase)}{rng.randint(10,99)}-" \
               f"{rng.randint(100,999)}{rng.choice(string.ascii_uppercase)}"

    def room():
        return f"{rng.choice('ABCDEFG')}{rng.randint(1,6)}{rng.randint(10,99)}"

    def code():
        return "".join(rng.choice("2345679ACDEFHJKMNPRTUVWXY") for _ in range(6))

    n_hop = max(2, n_items // 3)
    n_one = n_items - n_hop - max(1, n_items // 8)
    n_cnt = n_items - n_hop - n_one

    records, tasks, used_rooms = [], [], set()
    ptr = 0

    def fresh_room():
        while True:
            r = room()
            if r not in used_rooms:
                used_rooms.add(r)
                return r

    # multi-hop: project -> lead -> lead's room -> door code
    for i in range(n_hop):
        lead, proj = names[ptr], projects[i]; ptr += 1
        rm, dc = fresh_room(), code()
        records += [f"PROJECT {proj}: lead engineer is {lead}; status green.",
                    f"EMPLOYEE {lead}: badge {badge()}, dept {rng.choice(DEPTS)}, office {rm}.",
                    f"ROOM {rm}: door code {dc}, floor {rm[1]}."]
        tasks.append({"id": f"l_hop{i:02d}",
                      "question": f"What is the door code of the office of the lead "
                                  f"engineer of PROJECT {proj}?",
                      "answer": dc})

    # one-hop: badge lookup
    for i in range(n_one):
        who = names[ptr]; ptr += 1
        b = badge()
        records.append(f"EMPLOYEE {who}: badge {b}, dept {rng.choice(DEPTS)}, "
                       f"office {fresh_room()}.")
        tasks.append({"id": f"l_one{i:02d}",
                      "question": f"What is the badge of EMPLOYEE {who}?",
                      "answer": b})

    # count: rare dept appears exactly k times
    k = rng.randint(3, 7)
    for _ in range(k):
        who = names[ptr]; ptr += 1
        records.append(f"EMPLOYEE {who}: badge {badge()}, dept {RARE_DEPT}, "
                       f"office {fresh_room()}.")
    for i in range(n_cnt):
        tasks.append({"id": f"l_cnt{i:02d}",
                      "question": f"Exactly how many EMPLOYEE records list dept "
                                  f"{RARE_DEPT}? Answer with a single integer.",
                      "answer": str(k)})

    # distractor fill -- names disjoint from every needle employee
    reserved = ptr
    pool = len(names) - reserved
    target_chars = int(ctx_tokens * 3.6)
    while sum(len(x) + 1 for x in records) < target_chars:
        who = names[reserved + (ptr % pool)]; ptr += 1
        records.append(f"EMPLOYEE {who}#{ptr}: badge {badge()}, "
                       f"dept {rng.choice(DEPTS)}, office {room()}.")
    rng.shuffle(records)
    haystack = "\n".join(records)

    out = []
    for t in tasks:
        prompt = ("Below is a personnel database dump. Answer the question using ONLY "
                  "this data.\n\n=== DATABASE ===\n" + haystack + "\n=== END ===\n\n"
                  f"Question: {t['question']}\n\n"
                  "End your response with a line of the exact form:\nFINAL ANSWER: <answer>")
        out.append({"id": t["id"], "prompt": prompt, "answer": t["answer"]})
    return out


# ---------------------------------------------------------------------------
# RAG (optional)
# ---------------------------------------------------------------------------

RAG_ANSWER_TMPL = (
    "Answer the question using ONLY the context below. Do not use outside knowledge. "
    "If the context does not contain the answer, reply exactly: NOT IN CONTEXT\n\n"
    "=== CONTEXT ===\n{ctx}\n=== END CONTEXT ===\n\nQuestion: {q}"
)

JUDGE_TMPL = (
    "You are grading whether an ANSWER is faithful to a CONTEXT.\n"
    "faithful = every factual claim in the ANSWER is supported by the CONTEXT "
    "(or the ANSWER is exactly 'NOT IN CONTEXT' and the context truly lacks the answer).\n"
    "complete = the ANSWER actually addresses the question{gp_clause}.\n\n"
    "=== CONTEXT ===\n{ctx}\n=== END ===\n\nQUESTION: {q}\n\nANSWER:\n{a}\n\n{gp}"
    "Respond with ONLY a JSON object: {{\"faithful\": true|false, "
    "\"complete\": true|false, \"notes\": \"<short reason>\"}}"
)


def walk_star_path(obj, path):
    parts = path.split(".")

    def rec(cur, i):
        if i == len(parts):
            return [cur]
        p = parts[i]
        if p == "*":
            if isinstance(cur, list):
                out = []
                for item in cur:
                    out += rec(item, i + 1)
                return out
            return []
        if isinstance(cur, dict) and p in cur:
            return rec(cur[p], i + 1)
        if isinstance(cur, list) and p.isdigit():
            return rec(cur[int(p)], i + 1)
        return []
    return rec(obj, 0)


def fetch_context(args, question):
    url = args.vault_url_template.format(
        query=urllib.parse.quote(question), k=args.vault_k)
    data = http_get_json(url)
    texts = [t for t in walk_star_path(data, args.vault_text_path)
             if isinstance(t, str)]
    return "\n\n---\n\n".join(texts)


def judge_absolute(judge_ep, ctx, q, a, gold_points):
    gp = ""
    gp_clause = ""
    if gold_points:
        gp = "KEY POINTS the answer should cover:\n- " + "\n- ".join(gold_points) + "\n\n"
        gp_clause = " and covers the key points"
    prompt = JUDGE_TMPL.format(ctx=ctx, q=q, a=a, gp=gp, gp_clause=gp_clause)
    for attempt in range(2):
        r = judge_ep.chat(prompt, seed=1234 + attempt, max_tokens=1024, temp=0.0)
        try:
            v = json.loads(strip_fences(r["content"]))
            return bool(v.get("faithful")), bool(v.get("complete")), \
                str(v.get("notes", ""))[:200]
        except (json.JSONDecodeError, AttributeError):
            continue
    return False, False, "judge_unparseable"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def score_none(task, r):
    return None, "unscored"


UNSCORED = {"nonsense"}

SCORERS = {"math": score_math, "code": score_code,
           "json_tool": score_json, "instruct": score_instruct,
           "bizarre": score_bizarre, "grounded": score_math,
           "nonsense": score_none,
           "longctx": score_math,  # longctx/grounded reuse FINAL ANSWER matching
           # v2 suite
           "math_hard": score_math, "grounded_v2": score_math,
           "deepctx": score_math}

V2_CATS = ["math_hard", "grounded_v2", "deepctx", "agentic", "nonsense"]


def run_agentic(ep, tasks_dir, outdir, seed, limit, tdir):
    """Multi-turn tool-loop category. Tasks + tool impls come from
    tasks/agent_tasks.py; loop + trace scoring from agent_runner.py."""
    import importlib.util as ilu

    def load_mod(name, path):
        spec = ilu.spec_from_file_location(name, path)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    runner = load_mod("agent_runner", str(Path(tasks_dir) / "agent_runner.py"))
    atasks = load_mod("agent_tasks", str(Path(tasks_dir) / "agent_tasks.py"))
    tasks = atasks.AGENT_TASKS[:limit] if limit else atasks.AGENT_TASKS

    res_path = outdir / ep.name / "agentic.jsonl"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    def chat_fn(messages, tools, sd):
        return ep.chat_message(messages, tools=tools, seed=sd)

    for i, task in enumerate(tasks, 1):
        t0 = time.time()
        try:
            rec = runner.run_tool_loop(chat_fn, task, seed=seed)
            ok, why = runner.score_agent(task, rec, answers_match)
        except Exception as e:
            rec = {"final": None, "trace": [], "turns": 0,
                   "stop_reason": f"harness_error:{type(e).__name__}", "messages": []}
            ok, why = False, f"harness_error: {e}"
        dt = time.time() - t0
        with open(tdir / f"{ep.name}_agentic_{task['id']}.json", "w") as f:
            json.dump({"id": task["id"], "model": ep.name, "cat": "agentic",
                       "prompt": task["prompt"], "response": rec.get("final") or "",
                       "reasoning": "", "finish": rec.get("stop_reason"),
                       "trace": rec.get("trace"), "turns": rec.get("turns"),
                       "messages": [m for m in rec.get("messages", [])
                                    if m.get("role") != "user" or
                                    len(str(m.get("content", ""))) < 4000]},
                      f, ensure_ascii=False, indent=1, default=str)
        results.append({"id": task["id"], "pass": ok, "why": why, "latency": dt,
                        "ct": None, "finish": rec.get("stop_reason"),
                        "calls": len(rec.get("trace") or []),
                        "turns": rec.get("turns")})
        st = "PASS" if ok else "fail"
        print(f"    [{ep.name}/agentic] {i}/{len(tasks)} {st} "
              f"({dt:.1f}s, {results[-1]['calls']} calls)", flush=True)
    with open(res_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results

CATEGORY_MAXTOK = {"math": None, "code": None, "json_tool": None,
                   "instruct": None, "longctx": None}


def run_category(ep, cat, tasks, outdir, seed, concurrency, limit, tdir):
    res_path = outdir / ep.name / f"{cat}.jsonl"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = tasks[:limit] if limit else tasks
    scorer = SCORERS[cat]
    results = []

    def one(task):
        r = ep.chat(task["prompt"], seed=seed)
        ok, why = scorer(task, r)
        tr = {"id": task["id"], "model": ep.name, "cat": cat,
              "prompt": task["prompt"], "response": r.get("raw_content", ""),
              "reasoning": r.get("reasoning", ""),
              "finish": r.get("finish")}
        with open(tdir / f"{ep.name}_{cat}_{task['id']}.json", "w") as f:
            json.dump(tr, f, ensure_ascii=False, indent=1)
        row = {"id": task["id"], "pass": ok, "why": why,
               "latency": r["latency"], "ct": r.get("completion_tokens"),
               "finish": r.get("finish")}
        d = (task.get("meta") or {}).get("depth")
        if d is not None:
            row["depth"] = d
        return row

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            results = list(ex.map(one, tasks))
    else:
        for i, t in enumerate(tasks, 1):
            results.append(one(t))
            st = results[-1]['pass']
            st = 'resp' if st is None else ('PASS' if st else 'fail')
            print(f"    [{ep.name}/{cat}] {i}/{len(tasks)} {st} "
                  f"({results[-1]['latency']:.1f}s)", flush=True)
    with open(res_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results


def run_rag(ep, rag_tasks, args, outdir, judge_ep, tdir):
    res_path = outdir / ep.name / "rag.jsonl"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for i, t in enumerate(rag_tasks, 1):
        ctx = t.get("context") or fetch_context(args, t["question"])
        prompt = RAG_ANSWER_TMPL.format(ctx=ctx, q=t["question"])
        r = ep.chat(prompt, seed=args.seed)
        if "_mock_pass" in r:
            ok, notes = r["_mock_pass"], "mock"
        else:
            faithful, complete, notes = judge_absolute(
                judge_ep, ctx, t["question"], r["content"], t.get("gold_points"))
            ok = faithful and complete
        with open(tdir / f"{ep.name}_rag_{t['id']}.json", "w") as f:
            json.dump({"id": t["id"], "model": ep.name, "context": ctx,
                       "question": t["question"], "response": r.get("raw_content", ""),
                       "judge_notes": notes}, f, ensure_ascii=False, indent=1)
        results.append({"id": t["id"], "pass": ok, "why": notes,
                        "latency": r["latency"], "ct": r.get("completion_tokens"),
                        "finish": r.get("finish")})
        print(f"    [{ep.name}/rag] {i}/{len(rag_tasks)} "
              f"{'PASS' if ok else 'fail'}", flush=True)
    with open(res_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results


# ---------------------------------------------------------------------------
# Stats + report
# ---------------------------------------------------------------------------

def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def _collect(outdir, name):
    out = {}
    for p in sorted((outdir / name).glob("*.jsonl")):
        out[p.stem] = {r["id"]: r for r in load_jsonl(p)}
    return out


def _pair_lines(data, a, b):
    cats = sorted((set(data[a]) & set(data[b])) - UNSCORED)
    lines = [f"-- {a} vs {b} " + "-" * max(1, 62 - len(a) - len(b)),
             f"{'category':<10}{'n':>4} {'L pass':>8} {'R pass':>8} {'delta':>9}"
             f" {'L>R':>4} {'R>L':>4} {'p(McNemar)':>11}"]
    tb = tc = tn = tA = tB = 0
    for cat in cats:
        A, B = data[a][cat], data[b][cat]
        ids = sorted(set(A) & set(B))
        d1 = sum(1 for i in ids if A[i]["pass"] and not B[i]["pass"])
        d2 = sum(1 for i in ids if B[i]["pass"] and not A[i]["pass"])
        pa = sum(1 for i in ids if A[i]["pass"])
        pb = sum(1 for i in ids if B[i]["pass"])
        n = len(ids)
        lines.append(f"{cat:<10}{n:>4} {pa:>5}/{n:<3}{pb:>5}/{n:<3}"
                     f"{(pa-pb)/n*100 if n else 0:>+8.1f}pp {d1:>4} {d2:>4} "
                     f"{mcnemar_exact(d1, d2):>11.4f}")
        tb += d1; tc += d2; tn += n; tA += pa; tB += pb
    lines.append(f"{'OVERALL':<10}{tn:>4} {tA:>5}/{tn:<3}{tB:>5}/{tn:<3}"
                 f"{(tA-tB)/tn*100 if tn else 0:>+8.1f}pp {tb:>4} {tc:>4} "
                 f"{mcnemar_exact(tb, tc):>11.4f}")
    lines.append("")
    return lines


CAT_ABBR = {"math": "math", "code": "code", "json_tool": "json",
            "instruct": "instr", "bizarre": "bizar", "grounded": "grnd",
            "longctx": "lctx", "rag": "rag",
            "math_hard": "mhard", "grounded_v2": "grnd2",
            "deepctx": "dctx", "agentic": "agent"}


def build_showcase(outdir, names, cat="nonsense"):
    tdir = outdir / "transcripts"
    items = {}
    for name in names:
        for p in sorted(tdir.glob(f"{name}_{cat}_*.json")):
            tr = json.loads(p.read_text())
            items.setdefault(tr["id"], {"q": tr["prompt"], "a": {}})
            items[tr["id"]]["a"][name] = strip_think(tr.get("response", "")) \
                or "(empty response)"
    if not items:
        return None
    lines = [f"# {cat} showcase", "",
             "Unscored novelty round: no correct answers exist. "
             "Full raw output (thinking included) lives in transcripts/.", ""]
    for tid in sorted(items):
        lines.append(f"## {tid}: {items[tid]['q']}")
        lines.append("")
        for name in names:
            lines.append(f"**{name}:**")
            lines.append("")
            lines.append(items[tid]["a"].get(name, "(missing)"))
            lines.append("")
    path = outdir / f"{cat}_showcase.md"
    path.write_text("\n".join(lines))
    return path


def make_report(outdir, names, baseline=None):
    data = {n: _collect(outdir, n) for n in names}
    all_cats = sorted(set().union(*[set(d) for d in data.values()]))
    cats = [c for c in all_cats if c not in UNSCORED]
    w = max(max(len(n) for n in names), 8) + 2

    lines = ["netterm-eval report  |  models: " + ", ".join(names), "=" * 78, ""]
    hdr = f"{'model':<{w}}" + "".join(f"{CAT_ABBR.get(c, c)[:5]:>8}" for c in cats) \
          + f"{'OVERALL':>10}{'tok/s':>7}"
    lines.append(hdr)
    board = []
    for n in names:
        cells, tot_p, tot_n, tps = "", 0, 0, []
        for c in cats:
            D = data[n].get(c, {})
            p = sum(1 for r in D.values() if r["pass"])
            cells += f"{f'{p}/{len(D)}':>8}"
            tot_p += p; tot_n += len(D)
            for r in D.values():
                if r.get("ct") and r["latency"] > 0:
                    tps.append(r["ct"] / r["latency"])
        rate = tot_p / tot_n if tot_n else 0
        board.append((rate, n, cells, tot_p, tot_n,
                      statistics.median(tps) if tps else None))
    for rate, n, cells, tp, tn, med in sorted(board, reverse=True):
        lines.append(f"{n:<{w}}" + cells + f"{f'{tp}/{tn}':>10}"
                     + (f"{med:>7.0f}" if med else f"{'-':>7}"))
    lines.append("")

    # deepctx per-depth breakdown (lost-in-the-middle visibility)
    if "deepctx" in cats:
        buckets = sorted({round(r["depth"], 1) for n in names
                          for r in data[n].get("deepctx", {}).values()
                          if r.get("depth") is not None})
        if buckets:
            lines.append("deepctx by needle depth (fraction into haystack):")
            hdr2 = f"{'model':<{w}}" + "".join(f"{f'd={b}':>10}" for b in buckets)
            lines.append(hdr2)
            for n in names:
                D = data[n].get("deepctx", {})
                cells = ""
                for b in buckets:
                    rows_b = [r for r in D.values()
                              if r.get("depth") is not None
                              and round(r["depth"], 1) == b]
                    p = sum(1 for r in rows_b if r["pass"])
                    cells += f"{f'{p}/{len(rows_b)}':>10}" if rows_b else f"{'-':>10}"
                lines.append(f"{n:<{w}}" + cells)
            lines.append("")

    for uc in [c for c in all_cats if c in UNSCORED]:
        n_items = max(len(data[n].get(uc, {})) for n in names)
        sp = build_showcase(outdir, names, uc)
        lines.append(f"{uc}: {n_items} prompts per model, unscored novelty round "
                     f"-> {sp.name if sp else 'transcripts/'}")
        lines.append("")

    if len(names) >= 2:
        base = baseline or names[0]
        others = [n for n in names if n != base]
        pairs = [(base, o) for o in others]
        if baseline is None and len(names) <= 4:
            pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        lines.append("Pairwise (exact two-sided McNemar on discordant items):")
        lines.append("")
        for a, b in pairs:
            lines += _pair_lines(data, a, b)
    else:
        lines += ["Single-model run: no pairwise stats. Benchmark other models "
                  "with the same", "--seed/--ctx-tokens/--longctx-items, then:",
                  "  python3 harness.py compare --runs <dirA> <dirB> "
                  "--out <comparison-dir>", ""]

    lines += ["Reading guide: L>R = items only the left model passed (and vice versa).",
              "p < 0.05 with a delta you care about -> real difference at this n.",
              "High p does NOT prove equality -- widen n before calling it a tie.",
              "Truncated responses (finish=length) count as failures by design; check",
              "transcripts/ if one model shows many. Read transcripts for bizarre --",
              "the score checks the core; the comedy is in the prose."]
    report = "\n".join(lines)
    (outdir / "report.txt").write_text(report)
    print(report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

THINK_LEVELS = {"low": "on", "medium": "on", "med": "on", "minimal": "on",
                "high": "high", "max": "max", "xhigh": "max", "ultra": "max",
                "ultrathink": "max"}
THINK_MAXTOK = {"on": 16384, "high": 24576, "max": 32768}


def detect_thinking(extra):
    """Walk a model's extra request body for reasoning knobs.
    Returns (level, budget): level in {None,'off','on','high','max'}."""
    level, budget = None, 0

    def walk(o):
        nonlocal level, budget
        if isinstance(o, dict):
            for k, v in o.items():
                kl = str(k).lower()
                if kl in ("enable_thinking", "thinking", "enable_reasoning") \
                        and isinstance(v, bool):
                    level = "off" if not v else (level or "on")
                elif kl in ("reasoning_effort", "effort", "thinking_effort",
                            "reasoning_level", "thinking_level") \
                        and isinstance(v, str):
                    level = THINK_LEVELS.get(v.lower(), "on")
                elif kl in ("budget_tokens", "thinking_budget",
                            "max_thinking_tokens") and isinstance(v, (int, float)):
                    budget = max(budget, int(v))
                    level = level or "on"
                elif kl == "type" and v == "enabled":
                    level = level or "on"
                else:
                    walk(v)
        elif isinstance(o, list):
            for it in o:
                walk(it)
    walk(extra or {})
    return level, budget


def build_endpoints(args):
    specs = []
    if args.models_file:
        specs += json.load(open(args.models_file))
    for m in (args.model or []):
        specs.append(json.loads(m))
    if not specs:
        sys.exit("no models given: use --model '{\"name\":...,\"url\":...,"
                 "\"model\":...}' (repeatable) and/or --models-file models.json")
    names = [s.get("name") for s in specs]
    if len(set(names)) != len(names) or not all(names):
        sys.exit("every model needs a unique non-empty \"name\"")
    eps = []
    for s in specs:
        h = int(hashlib.sha256(s["name"].encode()).hexdigest(), 16)
        mock_p = 0.5 + (h % 40) / 100.0
        level, budget = detect_thinking(s.get("extra"))
        if s.get("max_tokens") is not None:
            mt = int(s["max_tokens"])
            if level in THINK_MAXTOK and mt < 8192:
                print(f"  WARNING {s['name']}: thinking={level} but explicit "
                      f"max_tokens={mt} -- reasoning may truncate (fail).")
        else:
            mt = args.max_tokens
            if level in THINK_MAXTOK:
                mt = max(mt, THINK_MAXTOK[level])
            if budget:
                mt = max(mt, budget + 4096)
        print(f"  {s['name']}: thinking={level or 'unspecified'}"
              + (f" budget={budget}" if budget else "")
              + f" -> max_tokens {mt}")
        eps.append(Endpoint(
            s["name"], s.get("url", "http://mock/v1"), s.get("model", "mock"),
            temp=float(s.get("temp", 0.0)), key=s.get("key"),
            extra=s.get("extra") or {},
            max_tokens=mt,
            timeout=args.timeout,
            mock=mock_p if args.mock else None))
    return eps


def cmd_compare(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cfgs = []
    for rd in args.runs:
        p = Path(rd) / "config.json"
        cfgs.append(json.loads(p.read_text()) if p.exists() else {})
    genkey = lambda c: (c.get("seed"), c.get("ctx_tokens"),
                        c.get("longctx_items"), c.get("deepctx_items"))
    same_gen = len({genkey(c) for c in cfgs}) == 1
    if not same_gen:
        print("WARNING: runs used different --seed/--ctx-tokens/"
              "--longctx-items/--deepctx-items; longctx+deepctx are excluded "
              "(same item ids, different haystacks -- pairing them would be "
              "invalid).")
    names = []
    for rd in args.runs:
        rdp = Path(rd)
        for md in sorted(p for p in rdp.iterdir()
                         if p.is_dir() and p.name != "transcripts"):
            name = md.name if md.name not in names else f"{md.name}@{rdp.name}"
            dst = outdir / name
            dst.mkdir(exist_ok=True)
            for f in md.glob("*.jsonl"):
                if f.stem in ("longctx", "deepctx") and not same_gen:
                    continue
                shutil.copy(f, dst / f.name)
            names.append(name)
    if len(names) < 2:
        sys.exit("compare needs at least two model result sets across the runs")
    make_report(outdir, names, args.baseline)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("report")
    rp.add_argument("--out", required=True)
    rp.add_argument("--baseline", default=None)

    cp = sub.add_parser("compare",
                        help="merge 2+ single-model (or any) runs and produce "
                             "the pairwise report across them")
    cp.add_argument("--runs", nargs="+", required=True,
                    help="two or more results directories from harness.py run")
    cp.add_argument("--out", required=True)
    cp.add_argument("--baseline", default=None)

    r = sub.add_parser("run")
    r.add_argument("--model", action="append",
                   help='JSON per model, repeatable: {"name":"glm52-q3",'
                        '"url":"http://host:8080/v1","model":"glm-5.2",'
                        '"temp":0,"key":null,"extra":{},"max_tokens":6144}')
    r.add_argument("--models-file", default=None,
                   help="JSON file containing a list of model objects")
    r.add_argument("--baseline", default=None,
                   help="model name to anchor pairwise stats (default: all pairs)")
    r.add_argument("--out", required=True)
    r.add_argument("--tasks", default=str(HERE / "tasks"))
    r.add_argument("--only", default=None,
                   help="comma list: math,code,json_tool,instruct,bizarre,"
                        "grounded,nonsense,longctx,rag")
    r.add_argument("--limit", type=int, default=None, help="items per category (smoke)")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--max-tokens", type=int, default=6144)
    r.add_argument("--timeout", type=int, default=900)
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--ctx-tokens", type=int, default=24000)
    r.add_argument("--longctx-items", type=int, default=16)
    r.add_argument("--suite", choices=["v1", "v2"], default="v1",
                   help="v2 = math_hard, grounded_v2, deepctx, agentic, nonsense")
    r.add_argument("--deepctx-items", type=int, default=15)
    r.add_argument("--mock", action="store_true", help="pipeline smoke test, no servers")
    # optional vault-RAG (nothing below is required for a default run)
    r.add_argument("--rag-tasks", default=None)
    r.add_argument("--vault-url-template", default=None,
                   help="e.g. http://vault:8000/search?q={query}&k={k}")
    r.add_argument("--vault-text-path", default="results.*.text")
    r.add_argument("--vault-k", type=int, default=8)
    r.add_argument("--judge-url", default=None)
    r.add_argument("--judge-model", default=None)
    r.add_argument("--judge-key", default=None)

    args = ap.parse_args()

    if args.cmd == "compare":
        cmd_compare(args)
        return

    if args.cmd == "report":
        outdir = Path(args.out)
        names = sorted([p.name for p in outdir.iterdir() if p.is_dir()
                        and p.name != "transcripts"])
        make_report(outdir, names, args.baseline)
        return

    eps = build_endpoints(args)
    outdir = Path(args.out)
    tdir = outdir / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in vars(args).items()
           if k not in ("model", "models_file") and "key" not in k}
    cfg["models"] = [{k: v for k, v in
                      {"name": e.name, "url": e.url, "model": e.model,
                       "temp": e.temp, "extra": e.extra,
                       "max_tokens": e.max_tokens}.items()} for e in eps]
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    if args.only:
        cats = args.only.split(",")
    elif args.suite == "v2":
        cats = list(V2_CATS)
    else:
        cats = ["math", "code", "json_tool", "instruct", "bizarre", "grounded",
                "nonsense", "longctx"] + (["rag"] if args.rag_tasks else [])

    task_sets = {}
    for cat in ("math", "code", "json_tool", "instruct", "bizarre", "grounded",
                "nonsense", "math_hard", "grounded_v2"):
        if cat in cats:
            task_sets[cat] = load_jsonl(Path(args.tasks) / f"{cat}.jsonl")
    if "longctx" in cats:
        task_sets["longctx"] = gen_longctx(args.seed, args.longctx_items,
                                           args.ctx_tokens)
    if "deepctx" in cats:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "gen_deepctx", str(Path(args.tasks) / "gen_deepctx.py"))
        _gdc = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_gdc)
        task_sets["deepctx"] = _gdc.gen(args.seed, args.deepctx_items,
                                        args.ctx_tokens)

    for cat in [c for c in cats if c != "rag"]:
        if cat == "agentic":
            if args.mock:
                print("== agentic: skipped in --mock (needs a live tool loop) ==")
                continue
            print("== agentic ==")
            for ep in eps:
                run_agentic(ep, args.tasks, outdir, args.seed, args.limit, tdir)
            continue
        print(f"== {cat} ({len(task_sets[cat])} items) ==")
        for ep in eps:
            run_category(ep, cat, task_sets[cat], outdir, args.seed,
                         args.concurrency, args.limit, tdir)

    if "rag" in cats:
        rag_tasks = load_jsonl(args.rag_tasks)
        if args.limit:
            rag_tasks = rag_tasks[:args.limit]
        need_ctx = any("context" not in t for t in rag_tasks)
        if need_ctx and not args.vault_url_template and not args.mock:
            sys.exit("rag tasks lack embedded context and no --vault-url-template given")
        if args.mock:
            judge = None
        elif args.judge_url:
            judge = Endpoint("judge", args.judge_url, args.judge_model,
                             key=args.judge_key, timeout=args.timeout)
        else:
            sys.exit("--judge-url/--judge-model required for rag")
        print(f"== rag ({len(rag_tasks)} items) ==")
        for ep in eps:
            run_rag(ep, rag_tasks, args, outdir, judge, tdir)

    make_report(outdir, [e.name for e in eps], args.baseline)


if __name__ == "__main__":
    main()
