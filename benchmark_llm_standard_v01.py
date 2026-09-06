from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset

from general128_v07 import General128V07

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def call_chat(endpoint: str, system: str, user: str, max_tokens: int = 80) -> str:
    body = json.dumps({
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def fixed_sample(ds, n: int, seed: int) -> List[Dict[str, Any]]:
    n = min(n, len(ds))
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    return [dict(ds[i]) for i in idx[:n]]


def parse_letter(text: str, nchoices: int) -> int | None:
    t = text.strip().upper()
    # Prefer a standalone answer letter near the end.
    hits = re.findall(r"(?:ANSWER\s*[:=]?\s*)?\b([A-Z])\b", t)
    for h in reversed(hits):
        j = ord(h) - 65
        if 0 <= j < nchoices:
            return j
    # Also accept a bare numeric option 1..N.
    nums = re.findall(r"(?<![\d.])(\d+)(?![\d.])", t)
    for s in reversed(nums):
        j = int(s) - 1
        if 0 <= j < nchoices:
            return j
    return None


def normalize_number(s: str) -> str | None:
    s = s.replace(",", "").replace("$", "").strip()
    m = re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", s)
    return s if m else None


def parse_final_number(text: str) -> str | None:
    t = text.replace(",", "")
    patterns = [
        r"####\s*([-+]?\d+(?:\.\d+)?)",
        r"FINAL(?:\s+ANSWER)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        r"ANSWER\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
    ]
    for p in patterns:
        ms = re.findall(p, t, flags=re.I)
        if ms:
            return normalize_number(ms[-1])
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", t)
    return normalize_number(nums[-1]) if nums else None


def gsm8k_gold(answer: str) -> str | None:
    m = re.search(r"####\s*([^\n]+)", answer)
    if not m:
        return None
    return normalize_number(m.group(1))


def mc_prompt(question: str, choices: List[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"{question}\n\n{opts}\n\nReturn only the letter of the best answer."


def eval_mc(endpoint: str, rows: List[Tuple[str, List[str], int]], task: str) -> Dict[str, Any]:
    correct = 0
    parsed = 0
    raw = []
    t0 = time.time()
    for i, (q, choices, gold) in enumerate(rows):
        out = call_chat(endpoint, "Answer the multiple-choice question. Output only one option letter.", mc_prompt(q, choices), 20)
        pred = parse_letter(out, len(choices))
        parsed += int(pred is not None)
        correct += int(pred == gold)
        raw.append({"i": i, "gold": gold, "pred": pred, "output": out})
    return {
        "task": task,
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "parse_rate": parsed / len(rows) if rows else None,
        "seconds": time.time() - t0,
        "examples": raw,
    }


def load_mmlu(n: int, seed: int):
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = []
    for x in fixed_sample(ds, n, seed):
        rows.append((x["question"], list(x["choices"]), int(x["answer"])))
    return rows


def load_arc(n: int, seed: int):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = []
    for x in fixed_sample(ds, n, seed):
        labels = list(x["choices"]["label"])
        texts = list(x["choices"]["text"])
        key = str(x["answerKey"])
        try:
            gold = labels.index(key)
        except ValueError:
            # Some ARC labels are numeric 1..N.
            gold = int(key) - 1
        rows.append((x["question"], texts, gold))
    return rows


def load_hellaswag(n: int, seed: int):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    rows = []
    for x in fixed_sample(ds, n, seed):
        q = (str(x.get("ctx", "")) + "\nChoose the most plausible continuation.").strip()
        rows.append((q, list(x["endings"]), int(x["label"])))
    return rows


def load_winogrande(n: int, seed: int):
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    rows = []
    for x in fixed_sample(ds, n, seed):
        q = str(x["sentence"]) + "\nWhich option best fills the blank?"
        rows.append((q, [str(x["option1"]), str(x["option2"])], int(x["answer"]) - 1))
    return rows


def load_truthfulqa(n: int, seed: int):
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    rows = []
    for x in fixed_sample(ds, n, seed):
        target = x["mc1_targets"]
        choices = list(target["choices"])
        labels = list(target["labels"])
        gold = labels.index(1)
        rows.append((x["question"], choices, gold))
    return rows


def eval_gsm8k(endpoint: str, n: int, seed: int) -> Dict[str, Any]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    xs = fixed_sample(ds, n, seed)
    correct = 0
    parsed = 0
    raw = []
    t0 = time.time()
    for i, x in enumerate(xs):
        prompt = (
            x["question"]
            + "\nSolve the problem. You may reason briefly, but end with exactly: FINAL: <number>"
        )
        out = call_chat(endpoint, "Solve grade-school math carefully.", prompt, 180)
        pred = parse_final_number(out)
        gold = gsm8k_gold(x["answer"])
        parsed += int(pred is not None)
        correct += int(pred == gold)
        raw.append({"i": i, "gold": gold, "pred": pred, "output": out})
    return {
        "task": "GSM8K",
        "n": len(xs),
        "correct": correct,
        "accuracy": correct / len(xs) if xs else None,
        "parse_rate": parsed / len(xs) if xs else None,
        "seconds": time.time() - t0,
        "examples": raw,
    }


def probe_general128() -> Dict[str, Any]:
    g = General128V07()
    methods = {name for name in dir(g) if not name.startswith("_")}
    text_methods = sorted(methods.intersection({"generate", "chat", "answer", "answer_text", "complete", "predict_text"}))
    return {
        "model": "General-128 V0.7",
        "raw_text_interface": bool(text_methods),
        "text_methods": text_methods,
        "standard_llm_tasks_supported": [],
        "coverage": 0.0,
        "note": "Current source accepts structured numeric/operator/graph inputs but exposes no raw-text generation interface. Standard LLM accuracy is therefore N/A rather than inferred from structured benchmarks.",
    }


def wilson(p: float, n: int, z: float = 1.96) -> List[float]:
    if n <= 0:
        return [0.0, 0.0]
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / den
    return [max(0.0, center-half), min(1.0, center+half)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--out", default="standard_llm_results.json")
    args = ap.parse_args()

    tasks = []
    loaders = [
        ("MMLU", load_mmlu),
        ("ARC-Challenge", load_arc),
        ("HellaSwag", load_hellaswag),
        ("WinoGrande", load_winogrande),
        ("TruthfulQA-MC1", load_truthfulqa),
    ]
    for j, (name, loader) in enumerate(loaders):
        rows = loader(args.samples, args.seed + 101*j)
        tasks.append(eval_mc(args.endpoint, rows, name))
    tasks.append(eval_gsm8k(args.endpoint, args.samples, args.seed + 999))

    for t in tasks:
        t["accuracy_ci95"] = wilson(t["accuracy"], t["n"])

    macro = sum(t["accuracy"] for t in tasks) / len(tasks)
    result = {
        "protocol": {
            "name": "standard-LLM-sampled-v0.1",
            "samples_per_task": args.samples,
            "seed": args.seed,
            "temperature": 0,
            "warning": "Sampled evaluation; not an official full-dataset leaderboard score.",
        },
        "general128": probe_general128(),
        "qwen": {
            "model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
            "tasks": tasks,
            "macro_accuracy": macro,
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
