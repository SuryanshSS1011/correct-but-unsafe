"""Does a defective kernel in vLLM return another request's data?

One engine per process, with cbu_vllm_plugin swapping SiluAndMul's kernel.
Victims V1 and V2 alternate before the same request B, short and long, a few
times each. If B's greedy tokens and log-probabilities depend on which victim
ran before it, while repeats with the same victim agree, data flowed from the
victim to B. The plugin's correct kernel is the control: with it, B must not
depend on the victim, except through effects of batching on numerics.

    CBU_KERNEL=defective VLLM_PLUGINS=cbu python gpu/vllm_leak/run_leak.py \
        --mode eager --order sequential --out results.json

--record (eager mode, sequential order) also runs the engine in this process
and compares B's unwritten output elements bit by bit with what the victim
last wrote at the same address in the tensors this op sees, its gate and up
input and its output. That count is a lower bound, because the victim's other
tensors, such as attention outputs, can also have written there.

The victim texts are synthetic.
"""

from __future__ import annotations

import argparse
import json
import os
import time

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
VICTIMS = {
    "V1": "Patient record 7731: Jordan Ellis, born 1984-03-12, has type 2 diabetes "
    "and takes metformin 500 mg twice daily; next visit on 2026-12-02. ",
    "V2": "Invoice 4410 for Orchid Logistics: 32 pallets of ceramic tiles shipped "
    "to Rotterdam on 2026-10-18, payment of 18,250 euros due within 45 days. ",
}
CALLER = "Explain in two sentences how rain forms."
LENGTHS = {"short": 24, "long": 400}


def tokens(tok, text: str, n: int) -> list[int]:
    ids: list[int] = []
    while len(ids) < n:
        ids += tok(text, add_special_tokens=False)["input_ids"]
    return ids[:n]


def compare_records(records: list[dict]) -> dict:
    """Compare the caller's unwritten elements with the victim's last write at each address."""
    import numpy as np

    victim = [r for r in records if r["phase"] == "victim"]
    caller = [r for r in records if r["phase"] == "caller" and r["kind"] == "out"]
    total = covered = equal = equal_shifted = 0
    wrote = []
    for v in victim:
        mask = np.ones(len(v["data"]), dtype=bool)
        mask[v["unwritten"]] = False
        wrote.append(mask)
    for c in caller:
        idx = c["unwritten"]
        if not len(idx):
            continue
        addr = c["ptr"] + idx * c["itemsize"]
        last = np.zeros(len(idx), dtype=np.int16)
        has = np.zeros(len(idx), dtype=bool)
        for v, mask in zip(victim, wrote):
            lo, hi = v["ptr"], v["ptr"] + len(v["data"]) * v["itemsize"]
            sel = (addr >= lo) & (addr < hi)
            if sel.any():
                rel = (addr[sel] - lo) // v["itemsize"]
                ok = mask[rel]
                pos = np.flatnonzero(sel)[ok]
                last[pos] = v["data"][rel[ok]]
                has[pos] = True
        got = c["data"][idx]
        total += len(idx)
        covered += int(has.sum())
        equal += int((has & (got == last)).sum())
        # Chance baseline: the same comparison against the victim's value one element further on.
        shifted = np.roll(last, -1)
        equal_shifted += int((has & np.roll(has, -1) & (got == shifted)).sum())
    return {
        "caller_unwritten": total,
        "at_victim_op_addresses": covered,
        "bit_equal_to_victim": equal,
        "bit_equal_shifted_baseline": equal_shifted,
    }


FIELDS = ("elements", "differ", "zero", "address_moved", "layout_mismatch")


def buffer_diff(a: list, b: list) -> dict:
    """Count elements that differ between two trials' unwritten regions, call by call.

    Calls are matched by order. address_moved counts calls whose buffer sat at a
    different address in the two trials, which happens without the caching
    allocator; their contents are still compared. zero counts unwritten elements
    that were zero in the first trial.
    """
    total = differ = zero = moved = mismatched = 0
    for (pa, da), (pb, db) in zip(a, b):
        if len(da) != len(db):
            mismatched += 1
            continue
        moved += pa != pb
        total += len(da)
        differ += int((da != db).sum())
        zero += int((da == 0).sum())
    return {
        "elements": total,
        "differ": differ,
        "zero": zero,
        "address_moved": moved,
        "layout_mismatch": mismatched,
    }


def differential(snaps: dict, repeats: int) -> dict:
    """Compare B's unwritten prefill buffers between trials with the same victim and with different ones.

    B's input is identical in every trial, so at layer 0 a difference can only come from
    memory that earlier requests left behind.
    """
    out = {}
    for ln in LENGTHS:
        res: dict = {}
        for scope, sl in (("layer0", slice(0, 1)), ("prefill", slice(None))):
            same: list = [
                buffer_diff(snaps[(ln, i, v)][sl], snaps[(ln, j, v)][sl])
                for v in VICTIMS
                for i in range(repeats)
                for j in range(i + 1, repeats)
            ]
            cross = [
                buffer_diff(snaps[(ln, r, "V1")][sl], snaps[(ln, r, "V2")][sl])
                for r in range(repeats)
            ]
            res[scope] = {
                k: {f: sum(x[f] for x in xs) for f in FIELDS}
                for k, xs in (("same_victim", same), ("different_victim", cross))
            }
        out[ln] = res
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eager", "graphs"], required=True)
    ap.add_argument("--order", choices=["sequential", "batched"], required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--record", action="store_true", help="eager and sequential only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument(
        "--chat",
        action="store_true",
        help="wrap B's prompt in the model's chat template; victims stay raw tokens",
    )
    a = ap.parse_args()
    if a.record:
        if a.mode != "eager" or a.order != "sequential":
            ap.error("--record needs --mode eager --order sequential")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=a.model,
        dtype="bfloat16",
        seed=0,
        max_model_len=2048,
        gpu_memory_utilization=a.gpu_memory_utilization,
        enforce_eager=a.mode == "eager",
        enable_prefix_caching=False,
        compilation_config={"custom_ops": ["+silu_and_mul"]},
    )
    startup = time.time() - t0
    plugin = None
    if a.record:
        import cbu_vllm_plugin as plugin
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=16, logprobs=5, seed=0)

    def ids(text: str) -> list[int]:
        if a.chat:
            text = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return tok(text, add_special_tokens=False)["input_ids"]
        return tok(text)["input_ids"]

    caller = {"prompt_token_ids": ids(CALLER)}
    victims = {
        (name, ln): {"prompt_token_ids": tokens(tok, text, n)}
        for name, text in VICTIMS.items()
        for ln, n in LENGTHS.items()
    }

    layers = None
    if plugin:
        from transformers import AutoConfig

        layers = AutoConfig.from_pretrained(a.model).num_hidden_layers
    trials = []
    snaps: dict = {}
    for ln in LENGTHS:
        for rep in range(a.repeats):
            for name in VICTIMS:
                v = victims[(name, ln)]
                if plugin:
                    plugin.RECORD.clear()
                    plugin.PHASE, plugin.RECORDING = "victim", True
                if a.order == "sequential":
                    llm.generate([v], sp, use_tqdm=False)
                    if plugin:
                        plugin.PHASE = "caller"
                    out = llm.generate([caller], sp, use_tqdm=False)[0].outputs[0]
                else:
                    out = llm.generate([v, caller], sp, use_tqdm=False)[1].outputs[0]
                rec = compare_records(plugin.RECORD) if plugin else None
                if plugin:
                    # B's first forward pass is its prefill, one call per layer.
                    calls = [
                        r
                        for r in plugin.RECORD
                        if r["phase"] == "caller" and r["kind"] == "out"
                    ]
                    snaps[(ln, rep, name)] = [
                        (r["ptr"], r["data"][r["unwritten"]].copy())
                        for r in calls[:layers]
                    ]
                    plugin.RECORDING = False
                    plugin.RECORD.clear()
                trials.append(
                    {
                        "length": ln,
                        "repeat": rep,
                        "victim": name,
                        "token_ids": list(out.token_ids),
                        "text": out.text,
                        "top_logprobs": [
                            [[t, lp.logprob] for t, lp in d.items()]
                            for d in (out.logprobs or [])
                        ],
                        "record": rec,
                    }
                )

    summary = {}
    for ln in LENGTHS:
        by = {
            name: [
                tuple(t["token_ids"])
                for t in trials
                if t["length"] == ln and t["victim"] == name
            ]
            for name in VICTIMS
        }
        summary[ln] = {
            "repeats_agree": all(len(set(v)) == 1 for v in by.values()),
            "depends_on_victim": len({v[0] for v in by.values()}) > 1,
            "distinct_outputs": len({x for v in by.values() for x in v}),
        }
    import torch
    import vllm

    result = {
        "config": {
            "kernel": os.environ.get("CBU_KERNEL", "off"),
            "block": os.environ.get("CBU_BLOCK", "256"),
            "skip": os.environ.get("CBU_SKIP", "1"),
            "no_caching": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", "") == "1",
            "mode": a.mode,
            "order": a.order,
            "repeats": a.repeats,
            "record": a.record,
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "model": a.model,
            "chat": a.chat,
            "lengths": LENGTHS,
            "startup_s": round(startup, 1),
        },
        "summary": summary,
        "differential": differential(snaps, a.repeats) if plugin else None,
        "trials": trials,
    }
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(
        json.dumps(
            {
                "config": result["config"],
                "summary": summary,
                "differential": result["differential"],
            }
        )
    )
    for t in trials:
        print(
            t["length"], t["repeat"], t["victim"], t["token_ids"][:8], t["record"] or ""
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
