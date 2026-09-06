from __future__ import annotations

"""Structure-controlled general reasoning benchmark for General-128 V0.7.

Both systems receive the same task information. General-128 consumes the JSON
structure directly; Qwen receives the exact same JSON serialized into its prompt.
This isolates rule induction / composition / graph reasoning / continual memory
better than a natural-language benchmark, but it is NOT a full general-language
intelligence benchmark.
"""

import argparse
import csv
import json
import math
import random
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

from general128_v07 import General128V07, PRIMITIVES


SYMBOLS = ["zem", "tor", "kav", "pel", "rin", "sud", "mek", "vot"]


def primitive_map() -> Dict[str, Any]:
    return {p.name: p.fn for p in PRIMITIVES}


def clean_num(x: float) -> float | int:
    if abs(x - round(x)) < 1e-12:
        return int(round(x))
    return round(float(x), 6)


def build_suite(seed: int = 20260906) -> Dict[str, Any]:
    rng = random.Random(seed)
    prims = [p.name for p in PRIMITIVES]
    rng.shuffle(prims)
    rule_map = dict(zip(SYMBOLS, prims))
    funcs = primitive_map()

    demos: Dict[str, List[List[float]]] = {}
    numeric_queries: List[Dict[str, Any]] = []
    qid = 0
    for sym in SYMBOLS:
        fn = funcs[rule_map[sym]]
        ds = []
        seen = set()
        while len(ds) < 5:
            a = rng.randint(-8, 9)
            b = rng.randint(-7, 8)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            ds.append([a, b, clean_num(fn(a, b))])
        demos[sym] = ds
        for _ in range(4):
            a = rng.randint(-12, 13)
            b = rng.randint(-10, 11)
            numeric_queries.append({
                "id": f"N{qid}", "symbol": sym, "a": a, "b": b,
                "answer": clean_num(fn(a, b)),
            })
            qid += 1

    composition_queries: List[Dict[str, Any]] = []
    for i in range(20):
        x = float(rng.randint(-5, 6))
        initial = x
        steps = []
        # keep multiplication from exploding too far
        for _ in range(3):
            sym = rng.choice(SYMBOLS)
            b = rng.randint(-4, 5)
            x = funcs[rule_map[sym]](x, b)
            if abs(x) > 500:
                x = math.copysign(500.0, x)
            steps.append([sym, b])
        composition_queries.append({
            "id": f"C{i}", "initial": clean_num(initial), "steps": steps,
            "answer": clean_num(x),
        })

    # Four graph domains with independent vocabularies.
    graph_domains: Dict[str, List[List[str]]] = {}
    graph_queries: List[Dict[str, Any]] = []
    kinds = ["before", "causes", "is_a", "greater"]
    prefixes = {"before": "T", "causes": "K", "is_a": "X", "greater": "G"}
    gid = 0
    for kind in kinds:
        nodes = [f"{prefixes[kind]}{j}" for j in range(8)]
        edges = [[nodes[j], nodes[j + 1]] for j in range(7)]
        graph_domains[kind] = edges
        # true multi-hop queries
        for a_idx, b_idx in [(0, 2), (0, 4), (1, 6), (3, 7)]:
            graph_queries.append({
                "id": f"G{gid}", "kind": kind, "a": nodes[a_idx], "b": nodes[b_idx], "answer": True,
            })
            gid += 1
        # false reverse / disconnected-direction queries
        for a_idx, b_idx in [(5, 1), (7, 2)]:
            graph_queries.append({
                "id": f"G{gid}", "kind": kind, "a": nodes[a_idx], "b": nodes[b_idx], "answer": False,
            })
            gid += 1

    # Continual/no-replay: phase 1 and phase 2 demonstrations are presented once.
    phase1_syms = SYMBOLS[:4]
    phase2_syms = SYMBOLS[4:]
    continual_queries = []
    cq = 0
    for sym in SYMBOLS:
        fn = funcs[rule_map[sym]]
        for _ in range(3):
            a = rng.randint(-11, 12)
            b = rng.randint(-9, 10)
            continual_queries.append({
                "id": f"R{cq}", "symbol": sym, "a": a, "b": b,
                "answer": clean_num(fn(a, b)),
                "origin": "old" if sym in phase1_syms else "new",
            })
            cq += 1

    # Distractors are visible but irrelevant.
    distractors = [
        {"key": f"junk{i}", "value": rng.randint(-999, 999)} for i in range(30)
    ]

    return {
        "seed": seed,
        "rule_symbols_are_arbitrary": True,
        "operator_demos": demos,
        "numeric_queries": numeric_queries,
        "composition_queries": composition_queries,
        "graph_domains": graph_domains,
        "graph_queries": graph_queries,
        "continual": {
            "phase1_symbols": phase1_syms,
            "phase2_symbols": phase2_syms,
            "phase1_demos": {s: demos[s] for s in phase1_syms},
            "phase2_demos": {s: demos[s] for s in phase2_syms},
            "final_queries": continual_queries,
        },
        "distractors": distractors,
        "evaluator_only_rule_map": rule_map,
    }


def run_general128(suite: Dict[str, Any]) -> Dict[str, Any]:
    g = General128V07()
    for sym, demos in suite["operator_demos"].items():
        g.learn_operator(sym, [(float(a), float(b), float(y)) for a, b, y in demos])

    pred: Dict[str, Any] = {}
    traces: Dict[str, Any] = {}
    for q in suite["numeric_queries"]:
        r = g.numeric(q["symbol"], q["a"], q["b"])
        pred[q["id"]] = clean_num(float(r.answer))
        traces[q["id"]] = {"cycles": r.cycles, "confidence": r.confidence, "verified": r.verified}

    for q in suite["composition_queries"]:
        r = g.compose(float(q["initial"]), [(s, float(b)) for s, b in q["steps"]])
        pred[q["id"]] = clean_num(float(r.answer))
        traces[q["id"]] = {"cycles": r.cycles, "confidence": r.confidence, "verified": r.verified}

    for kind, edges in suite["graph_domains"].items():
        for a, b in edges:
            g.remember_relation(kind, a, b)
    for q in suite["graph_queries"]:
        r = g.relation(q["kind"], q["a"], q["b"])
        pred[q["id"]] = bool(r.answer)
        traces[q["id"]] = {"cycles": r.cycles, "confidence": r.confidence, "verified": r.verified}

    # Separate persistent state for explicit continual test.
    gc = General128V07()
    cont = suite["continual"]
    for sym, demos in cont["phase1_demos"].items():
        gc.learn_operator(sym, [(float(a), float(b), float(y)) for a, b, y in demos])
    phase1_snapshot = gc.snapshot()
    # No replay of phase1 examples here.
    for sym, demos in cont["phase2_demos"].items():
        gc.learn_operator(sym, [(float(a), float(b), float(y)) for a, b, y in demos])
    phase2_snapshot = gc.snapshot()
    for q in cont["final_queries"]:
        r = gc.numeric(q["symbol"], q["a"], q["b"])
        pred[q["id"]] = clean_num(float(r.answer))
        traces[q["id"]] = {"cycles": r.cycles, "confidence": r.confidence, "verified": r.verified, "origin": q["origin"]}

    return {
        "model": "General-128-V0.7",
        "predictions": pred,
        "traces": traces,
        "snapshot": g.snapshot(),
        "continual_phase1_snapshot": phase1_snapshot,
        "continual_phase2_snapshot": phase2_snapshot,
    }


def _answers_for_section(suite: Dict[str, Any], section: str) -> Dict[str, Any]:
    if section == "numeric":
        qs = suite["numeric_queries"]
    elif section == "composition":
        qs = suite["composition_queries"]
    elif section == "graph":
        qs = suite["graph_queries"]
    elif section == "continual":
        qs = suite["continual"]["final_queries"]
    else:
        raise KeyError(section)
    return {q["id"]: q["answer"] for q in qs}


def make_qwen_prompt(suite: Dict[str, Any], section: str) -> str:
    instruction = (
        "Solve the benchmark below. Rule symbols such as zem/tor are arbitrary; infer their behavior only from DEMOS. "
        "Return ONLY one flat JSON object mapping every query id to its answer, with no explanation. "
        "Use JSON booleans true/false for relation questions."
    )
    if section == "numeric":
        payload = {
            "DEMOS": suite["operator_demos"],
            "QUERIES": [{k: q[k] for k in ("id", "symbol", "a", "b")} for q in suite["numeric_queries"]],
            "DISTRACTORS": suite["distractors"],
        }
    elif section == "composition":
        payload = {
            "DEMOS": suite["operator_demos"],
            "QUERIES": [{k: q[k] for k in ("id", "initial", "steps")} for q in suite["composition_queries"]],
            "DISTRACTORS": suite["distractors"][:10],
        }
    elif section == "graph":
        payload = {
            "FACTS": suite["graph_domains"],
            "NOTE": "before, causes, is_a, and greater are transitive in this benchmark",
            "QUERIES": [{k: q[k] for k in ("id", "kind", "a", "b")} for q in suite["graph_queries"]],
        }
    elif section == "continual":
        c = suite["continual"]
        payload = {
            "PHASE1_DEMOS_SEEN_ONCE": c["phase1_demos"],
            "PHASE2_DEMOS_SEEN_AFTER_PHASE1": c["phase2_demos"],
            "FINAL_QUERIES_AFTER_PHASE2": [{k: q[k] for k in ("id", "symbol", "a", "b")} for q in c["final_queries"]],
            "NO_PHASE1_DEMO_IS_REPEATED_AFTER_PHASE2": True,
        }
    else:
        raise KeyError(section)
    return instruction + "\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def call_openai_compatible(endpoint: str, prompt: str, timeout: int = 300) -> str:
    body = json.dumps({
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "You are a precise benchmark solver. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 1400,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def extract_flat_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    starts = [m.start() for m in re.finditer(r"\{", text)]
    ends = [m.end() for m in re.finditer(r"\}", text)]
    for s in starts:
        for e in reversed(ends):
            if e <= s:
                continue
            try:
                obj = json.loads(text[s:e])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return {}


def run_qwen(suite: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
    all_pred: Dict[str, Any] = {}
    raw: Dict[str, str] = {}
    latency: Dict[str, float] = {}
    for section in ("numeric", "composition", "graph", "continual"):
        prompt = make_qwen_prompt(suite, section)
        t0 = time.time()
        text = call_openai_compatible(endpoint, prompt)
        latency[section] = time.time() - t0
        raw[section] = text
        all_pred.update(extract_flat_json(text))
    return {
        "model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
        "predictions": all_pred,
        "raw_outputs": raw,
        "latency_seconds": latency,
    }


def equivalent(gold: Any, pred: Any) -> bool:
    if isinstance(gold, bool):
        if isinstance(pred, bool):
            return gold is pred
        if isinstance(pred, str):
            return pred.strip().lower() == str(gold).lower()
        return False
    try:
        return abs(float(gold) - float(pred)) <= 1e-3
    except Exception:
        return False


def score_model(suite: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    pred = result.get("predictions", {})
    sections = {}
    total_ok = total_n = 0
    for section in ("numeric", "composition", "graph", "continual"):
        gold = _answers_for_section(suite, section)
        ok = sum(1 for qid, y in gold.items() if qid in pred and equivalent(y, pred[qid]))
        invalid = sum(1 for qid in gold if qid not in pred)
        sections[section] = {"correct": ok, "total": len(gold), "accuracy": ok / len(gold), "missing": invalid}
        total_ok += ok
        total_n += len(gold)

    # Old/new retention split for continual.
    cq = suite["continual"]["final_queries"]
    for origin in ("old", "new"):
        items = [q for q in cq if q["origin"] == origin]
        ok = sum(1 for q in items if q["id"] in pred and equivalent(q["answer"], pred[q["id"]]))
        sections[f"continual_{origin}"] = {"correct": ok, "total": len(items), "accuracy": ok / len(items), "missing": len(items) - sum(1 for q in items if q["id"] in pred)}

    return {
        "model": result["model"],
        "sections": sections,
        "overall_correct": total_ok,
        "overall_total": total_n,
        "overall_accuracy": total_ok / total_n,
    }


def write_outputs(outdir: Path, suite: Dict[str, Any], model_results: List[Dict[str, Any]], scores: List[Dict[str, Any]]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "suite.json").write_text(json.dumps(suite, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in model_results:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", r["model"])
        (outdir / f"{safe}_raw.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "scores.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")

    with (outdir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "section", "correct", "total", "accuracy", "missing"])
        for s in scores:
            for section, d in s["sections"].items():
                w.writerow([s["model"], section, d["correct"], d["total"], f"{d['accuracy']:.6f}", d["missing"]])
            w.writerow([s["model"], "OVERALL", s["overall_correct"], s["overall_total"], f"{s['overall_accuracy']:.6f}", ""])

    lines = [
        "# General-128 V0.7 vs Qwen2.5-0.5B Q4_K_M",
        "",
        "## Scope",
        "Structure-controlled synthetic reasoning benchmark. Both systems receive the same demonstrations/facts/queries; General-128 consumes structured JSON directly and Qwen receives the same JSON serialized in the prompt. This is not MMLU/MT-Bench and is not proof of general intelligence.",
        "",
        "| Model | Numeric induction | 3-step composition | Graph multi-hop | Continual no-replay | Overall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in scores:
        sec = s["sections"]
        lines.append(
            f"| {s['model']} | {sec['numeric']['accuracy']:.1%} | {sec['composition']['accuracy']:.1%} | "
            f"{sec['graph']['accuracy']:.1%} | {sec['continual']['accuracy']:.1%} | {s['overall_accuracy']:.1%} |"
        )
    lines += ["", "## Continual retention split", "", "| Model | old relations | new relations |", "|---|---:|---:|"]
    for s in scores:
        sec = s["sections"]
        lines.append(f"| {s['model']} | {sec['continual_old']['accuracy']:.1%} | {sec['continual_new']['accuracy']:.1%} |")
    lines += [
        "",
        "## Interpretation guardrail",
        "A General-128 win here means stronger performance on this deliberately structured rule-induction / composition / graph / continual-learning suite. It does not by itself mean the system is a stronger general language model than Qwen2.5-0.5B.",
    ]
    (outdir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--outdir", default="benchmark_results")
    ap.add_argument("--qwen-endpoint", default="")
    args = ap.parse_args()

    suite = build_suite(args.seed)
    results = [run_general128(suite)]
    if args.qwen_endpoint:
        results.append(run_qwen(suite, args.qwen_endpoint))
    scores = [score_model(suite, r) for r in results]
    write_outputs(Path(args.outdir), suite, results, scores)
    print(json.dumps(scores, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
