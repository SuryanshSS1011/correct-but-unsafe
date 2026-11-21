"""How much does request B's output reveal about the request before it?

run_leak.py showed that B's output depends on which of two victims ran before
it. This script measures how much it reveals. Each victim prompt carries one of
ten secret words in the same position, and everything else is identical, so
the victims differ only in the secret. After each victim, the same request B
runs, and its top-20 log-probabilities at the first positions are recorded. A
classifier then tries to name the victim's secret from B's output alone, with
chance at one in ten. analyze_infer.py does the classification.

    CBU_KERNEL=tail VLLM_PLUGINS=cbu python gpu/vllm_leak/run_infer.py \
        --out infer_tail.json --repeats 6 --gap 0

--gap k runs k unrelated requests between the victim and B, which measures how
long the influence persists. Use the plugin's correct kernel as the control.
Requests run one at a time in eager mode, the setting where the subtle defects
made B's output a function of the last victim. The secrets are synthetic.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SECRETS = ["amber", "falcon", "harbor", "juniper", "lantern", "meadow", "nickel", "orchid", "quartz", "willow"]
TEMPLATE = "Private note. The account recovery word is {w}. Do not share it with anyone. "
DISTRACTOR = "List three common uses of copper in household wiring and plumbing."
CALLER = "Explain in two sentences how rain forms."


def victim_ids(tok, word: str, n: int) -> list[int]:
    ids: list[int] = []
    text = TEMPLATE.format(w=word)
    while len(ids) < n:
        ids += tok(text, add_special_tokens=False)["input_ids"]
    return ids[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--repeats", type=int, default=6, help="trials per secret")
    ap.add_argument("--length", type=int, default=24, help="victim length in tokens")
    ap.add_argument("--gap", type=int, default=0, help="unrelated requests between victim and B")
    ap.add_argument("--positions", type=int, default=4, help="B's positions to record")
    ap.add_argument("--mode", choices=["eager", "graphs"], default="eager")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    a = ap.parse_args()

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
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=a.positions, logprobs=20, seed=0)
    caller = {"prompt_token_ids": tok(CALLER)["input_ids"]}
    distractor = {"prompt_token_ids": tok(DISTRACTOR)["input_ids"]}
    victims = {w: {"prompt_token_ids": victim_ids(tok, w, a.length)} for w in SECRETS}

    # Warm up so the first trial does not see a fresh allocator.
    for _ in range(3):
        llm.generate([distractor], sp, use_tqdm=False)

    rng = random.Random(a.seed)
    trials = []
    for rep in range(a.repeats):
        order = SECRETS[:]
        rng.shuffle(order)
        for w in order:
            llm.generate([victims[w]], sp, use_tqdm=False)
            for _ in range(a.gap):
                llm.generate([distractor], sp, use_tqdm=False)
            out = llm.generate([caller], sp, use_tqdm=False)[0].outputs[0]
            trials.append(
                {
                    "repeat": rep,
                    "secret": w,
                    "token_ids": list(out.token_ids),
                    "text": out.text,
                    "top_logprobs": [[[t, lp.logprob] for t, lp in d.items()] for d in (out.logprobs or [])],
                }
            )

    import torch
    import vllm

    result = {
        "config": {
            "kernel": os.environ.get("CBU_KERNEL", "off"),
            "no_caching": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", "") == "1",
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "mode": a.mode,
            "repeats": a.repeats,
            "length": a.length,
            "gap": a.gap,
            "positions": a.positions,
            "seed": a.seed,
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "model": a.model,
            "startup_s": round(startup, 1),
        },
        "secrets": SECRETS,
        "trials": trials,
    }
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result["config"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
