from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset

from benchmark_llm_standard_v01 import fixed_sample, gsm8k_gold, call_chat, parse_final_number
from general128_v09r34c import General128V09R34C


def eq(a, b):
    try:
        return a is not None and b is not None and abs(float(a) - float(b)) <= 1e-7 * max(1.0, abs(float(b)))
    except Exception:
        return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoint', required=True)
    ap.add_argument('--samples', type=int, default=30)
    ap.add_argument('--seed', type=int, default=20260907)
    ap.add_argument('--out', default='v09r34c_blind30.json')
    ap.add_argument('--summary-out', default='v09r34c_blind30_summary.json')
    a = ap.parse_args()

    ds = load_dataset('openai/gsm8k', 'main', split='test')
    xs = fixed_sample(ds, a.samples, a.seed)

    g = General128V09R34C(a.endpoint)
    rows = []
    base_correct = 0
    hybrid_correct = 0
    t0 = time.time()

    for i, x in enumerate(xs):
        raw = call_chat(
            a.endpoint,
            'Solve grade-school math carefully.',
            x['question'] + '\nSolve the problem. You may reason briefly, but end with exactly: FINAL: <number>',
            180,
        )
        bp = parse_final_number(raw)
        gold = gsm8k_gold(x['answer'])
        base_ok = eq(bp, gold)
        base_correct += int(base_ok)

        ops0 = g.math.ops
        over0 = g.overrides
        route0 = g.routed
        r = g.answer_math(x['question'], baseline_hint=bp)
        hp = str(r.answer) if r.answer is not None else None
        hyb_ok = eq(hp, gold)
        hybrid_correct += int(hyb_ok)

        routed_delta = int(g.routed - route0)
        override_delta = int(g.overrides - over0)
        causal_ops = int(g.math.ops - ops0)
        fam = next((z.split('=', 1)[1] for z in r.trace if z.startswith('family=')), None)

        row = {
            'i': i,
            'question': x['question'],
            'gold': gold,
            'baseline': bp,
            'v09r34c': hp,
            'base_ok': base_ok,
            'v09r34c_ok': hyb_ok,
            'family': fam,
            'causal_ops': causal_ops,
            'routed': routed_delta,
            'override': override_delta,
            'trace': r.trace,
        }
        rows.append(row)
        print(
            i,
            'gold', gold,
            'base', bp,
            'r34c', hp,
            'base_ok', base_ok,
            'r34c_ok', hyb_ok,
            'family', fam,
            'ops', causal_ops,
            'routed', routed_delta,
            'override', override_delta,
            flush=True,
        )

    routed_rows = [r for r in rows if r['routed']]
    fallback_rows = [r for r in rows if not r['routed']]
    beneficial = sum((not r['base_ok']) and r['v09r34c_ok'] for r in rows)
    harmful = sum(r['base_ok'] and (not r['v09r34c_ok']) for r in rows)
    routed_correct = sum(r['v09r34c_ok'] for r in routed_rows)
    fallback_correct = sum(r['v09r34c_ok'] for r in fallback_rows)
    family_counts = {}
    for r in routed_rows:
        family_counts[r['family']] = family_counts.get(r['family'], 0) + 1

    n = len(rows)
    summary = {
        'seed': a.seed,
        'n': n,
        'baseline_correct': base_correct,
        'baseline_accuracy': base_correct / n,
        'r34c_correct': hybrid_correct,
        'r34c_accuracy': hybrid_correct / n,
        'accuracy_delta': (hybrid_correct - base_correct) / n,
        'routed': len(routed_rows),
        'routing_coverage': len(routed_rows) / n,
        'routed_correct': routed_correct,
        'routed_accuracy': routed_correct / len(routed_rows) if routed_rows else None,
        'beneficial_fixes': beneficial,
        'harmful_overrides': harmful,
        'overrides': sum(r['override'] for r in rows),
        'causal_general128_ops': sum(r['causal_ops'] for r in rows),
        'fallback_n': len(fallback_rows),
        'fallback_correct': fallback_correct,
        'fallback_accuracy': fallback_correct / len(fallback_rows) if fallback_rows else None,
        'family_counts': family_counts,
        'seconds': time.time() - t0,
        'snapshot': g.snapshot(),
    }

    result = {'summary': summary, 'rows': rows}
    Path(a.out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    Path(a.summary_out).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('BLIND_SUMMARY')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
