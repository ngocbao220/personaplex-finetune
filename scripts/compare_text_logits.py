#!/usr/bin/env python
"""Locate the FIRST real divergence between two native audit conditions.

Compares `<dir>/text_logits.json` (raw text logits before GT forcing) and
`<dir>/tokens.json` (actual per-step model input) from `audit_personaplex.py probe`.

Question answered: at the first frame where the text prediction differs, were the
model inputs (history) still identical?  If inputs already differed -> forcing/offset
path bug.  If inputs were identical -> the two runs truly saw different context only
because of an earlier own-generated token, or the logits themselves are nondeterministic.

Usage:
  python scripts/compare_text_logits.py <probe>/full_gt <probe>/forced_agent_audio
  python scripts/compare_text_logits.py A B --context 8 --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PAD, EPAD = 3, 0


def load(d: Path):
    t = json.loads((d / "text_logits.json").read_text())
    k = json.loads((d / "tokens.json").read_text()) if (d / "tokens.json").is_file() else {"steps": []}
    frames = {f["dialogue_input_frame"]: f for f in t["frames"]}
    steps = {s["offset"]: s for s in k["steps"]}
    return t, frames, steps, k


def tok(x):
    return {PAD: "PAD", EPAD: "EPAD"}.get(x, str(x)) if x is not None else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="reference condition dir, e.g. full_gt")
    ap.add_argument("b", type=Path, help="other condition dir, e.g. forced_agent_audio")
    ap.add_argument("--context", type=int, default=5, help="frames shown before divergence")
    ap.add_argument("--pad-tol", type=float, default=1e-3, help="pad_probability tolerance")
    ap.add_argument("--json", type=Path, help="write machine-readable report")
    args = ap.parse_args()

    ta, fa, sa, ka = load(args.a)
    tb, fb, sb, kb = load(args.b)
    na, nb = args.a.name, args.b.name
    report: dict = {"a": str(args.a), "b": str(args.b),
                    "completed": [ta.get("completed"), tb.get("completed")],
                    "frames": [len(fa), len(fb)]}

    # 1) First divergence in the actual model INPUT (all streams, incl. system prompt).
    first_input = None
    for off in sorted(set(sa) & set(sb)):
        if sa[off]["input"] != sb[off]["input"]:
            diff = [i for i, (x, y) in enumerate(zip(sa[off]["input"], sb[off]["input"])) if x != y]
            first_input = {"offset": off, "dialogue": sa[off]["dialogue"], "differing_stream_indices": diff,
                           na: sa[off]["input"], nb: sb[off]["input"]}
            break
    report["first_input_divergence"] = first_input

    # 2) First divergence in raw text LOGITS (dialogue only).
    common = sorted(set(fa) & set(fb))
    first_logit = None
    for i in common:
        x, y = fa[i], fb[i]
        if (x["prediction"] != y["prediction"] or x["top_tokens"] != y["top_tokens"]
                or abs(x["pad_probability"] - y["pad_probability"]) > args.pad_tol):
            first_logit = i
            break

    # 3) First frame where B emitted a non-PAD/EPAD prediction (its own-generated deviation).
    first_b_nonpad = next((i for i in common if fb[i]["prediction"] not in (PAD, EPAD)), None)
    # B's own deviation from GT (incl. PAD<->EPAD swaps); its token re-enters the input next step.
    first_b_deviation = next((i for i in common if fa[i]["gt_target"] is not None
                              and fb[i]["prediction"] != fa[i]["gt_target"]), None)
    first_gt_real = next((i for i in common if fa[i]["gt_target"] not in (None, PAD, EPAD)), None)
    first_gt_epad = next((i for i in common if fa[i]["gt_target"] == EPAD), None)
    report.update(first_logit_divergence_frame=first_logit, first_b_nonpad_prediction_frame=first_b_nonpad,
                  first_b_deviation_from_gt_frame=first_b_deviation,
                  first_gt_real_token_frame=first_gt_real, first_gt_epad_frame=first_gt_epad)

    print(f"A={na}  B={nb}  completed={report['completed']}  frames={report['frames']}")
    print(f"first GT EPAD frame        : {first_gt_epad}")
    print(f"first GT real-token frame  : {first_gt_real}")
    print(f"first B deviation from GT  : {first_b_deviation}")
    print(f"first B non-pad prediction : {first_b_nonpad}")
    print(f"first LOGIT divergence     : {first_logit}")
    if first_input:
        print(f"first INPUT divergence     : offset {first_input['offset']} "
              f"(dialogue={first_input['dialogue']}) streams {first_input['differing_stream_indices']}")
        print(f"   {na}: {first_input[na]}\n   {nb}: {first_input[nb]}")
    else:
        print("first INPUT divergence     : none (or tokens.json missing)")

    if first_logit is not None:
        lo = max(common[0], first_logit - args.context)
        print(f"\n{'frame':>6} {'offset':>6} | {'gt':>6} | {na+' pred':>14} {'p_pad':>7} | {nb+' pred':>20} {'p_pad':>7}")
        for i in range(lo, first_logit + 3):
            if i not in fa or i not in fb:
                continue
            x, y = fa[i], fb[i]
            mark = "  <-- first logit divergence" if i == first_logit else ""
            print(f"{i:>6} {x['native_offset']:>6} | {tok(x['gt_target']):>6} | "
                  f"{tok(x['prediction']):>14} {x['pad_probability']:>7.4f} | "
                  f"{tok(y['prediction']):>20} {y['pad_probability']:>7.4f}{mark}")
        x, y = fa[first_logit], fb[first_logit]
        print(f"\ntop5 {na}: {x['top_tokens']}\ntop5 {nb}: {y['top_tokens']}")

    # Verdict.
    if first_logit is None:
        verdict = "Logits identical on all common frames."
    elif first_input and first_input["offset"] <= fa[first_logit]["native_offset"]:
        if first_b_deviation is not None and first_b_deviation < first_logit:
            verdict = ("Inputs diverged after B deviated from GT (its own token fed back): consequence, "
                       "not root cause. Root question: why did B mispredict at first_b_deviation_from_gt_frame "
                       "(EPAD vs PAD?) while A saw identical history.")
        else:
            verdict = ("INPUTS DIFFER BEFORE ANY OWN-GENERATED TOKEN: forcing/offset path bug "
                       "(check differing_stream_indices: 0=text, 1-8=agent audio, 9-16=user).")
    else:
        verdict = ("Inputs identical up to logit divergence, yet logits differ. Candidates: (a) GT leaks into "
                   "A's logits via the provided/target path or untracked state (KV cache, depformer, forced "
                   "audio applied at a different step); (b) nondeterminism (CUDA graph/bf16). Rerun B twice "
                   "with NO_CUDA_GRAPH=1: if B==B but A!=B, it is (a).")
    report["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
