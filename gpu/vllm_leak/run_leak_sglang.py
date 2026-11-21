"""run_leak.py for SGLang: the same victims, request and output format.

    CBU_KERNEL=defective PYTHONPATH=gpu/vllm_leak/sglang_patch:gpu/vllm_leak \
        python gpu/vllm_leak/run_leak_sglang.py --mode eager --order sequential --out r.json

--mode eager disables SGLang's CUDA graphs. Radix (prefix) caching is off, so B
cannot reuse a victim's KV cache. summarize.py reads the results.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from run_leak import CALLER, LENGTHS, MODEL, VICTIMS, tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eager", "graphs"], required=True)
    ap.add_argument("--order", choices=["sequential", "batched"], required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import sglang
    import torch
    from transformers import AutoTokenizer

    t0 = time.time()
    engine = sglang.Engine(
        model_path=a.model,
        dtype="bfloat16",
        random_seed=0,
        context_length=2048,
        mem_fraction_static=0.5,
        disable_radix_cache=True,
        disable_cuda_graph=a.mode == "eager",
        log_level="error",
    )
    startup = time.time() - t0
    tok = AutoTokenizer.from_pretrained(a.model)
    sp = {"temperature": 0.0, "max_new_tokens": 16}
    caller = tok(CALLER)["input_ids"]
    victims = {
        (name, ln): tokens(tok, text, n)
        for name, text in VICTIMS.items()
        for ln, n in LENGTHS.items()
    }

    def gen(prompts: list[list[int]]) -> dict:
        return engine.generate(
            input_ids=prompts,
            sampling_params=sp,
            return_logprob=True,
            top_logprobs_num=5,
        )[-1]

    trials = []
    for ln in LENGTHS:
        for rep in range(a.repeats):
            for name in VICTIMS:
                v = victims[(name, ln)]
                if a.order == "sequential":
                    gen([v])
                    out = gen([caller])
                else:
                    out = gen([v, caller])
                top = out["meta_info"].get("output_top_logprobs") or []
                trials.append(
                    {
                        "length": ln,
                        "repeat": rep,
                        "victim": name,
                        "token_ids": list(out["output_ids"]),
                        "text": out["text"],
                        "top_logprobs": [[[e[1], e[0]] for e in pos] for pos in top],
                        "record": None,
                    }
                )
    engine.shutdown()

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
    result = {
        "config": {
            "engine": f"sglang {sglang.__version__}",
            "kernel": os.environ.get("CBU_KERNEL", "off"),
            "block": os.environ.get("CBU_BLOCK", "256"),
            "skip": os.environ.get("CBU_SKIP", "1"),
            "no_caching": False,
            "mode": a.mode,
            "order": a.order,
            "repeats": a.repeats,
            "record": False,
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "model": a.model,
            "lengths": LENGTHS,
            "startup_s": round(startup, 1),
        },
        "summary": summary,
        "differential": None,
        "trials": trials,
    }
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"config": result["config"], "summary": summary}))
    for t in trials:
        print(
            t["length"],
            t["repeat"],
            t["victim"],
            t["token_ids"][:8],
            repr(t["text"][:60]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
