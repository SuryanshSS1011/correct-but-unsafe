"""Parse Dr. Kernel cold-start trajectories into per-round kernels and harness feedback.

Each row holds five rounds. Assistant turns contain a written analysis followed
by one python block that defines `ModelNew`. User turns 2 to 5 carry the
KernelGYM server feedback for the previous round, so rounds 1 to 4 have a
recorded verdict and round 5 does not.
"""
from __future__ import annotations

import re

CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
BOOL = {"true": True, "false": False}


def extract_code(content: str) -> str | None:
    blocks = CODE_BLOCK.findall(content or "")
    with_model = [b for b in blocks if "class ModelNew" in b]
    if with_model:
        return with_model[-1]
    return None


def parse_feedback(content: str) -> dict | None:
    if not content or "Server feedback" not in content:
        return None
    out: dict = {}
    for key in ("compiled", "correctness", "decoy_kernel"):
        m = re.search(rf'"{key}"\s*:?\s*(true|false)', content)
        out[key] = BOOL[m.group(1)] if m else None
    for key in ("speedup", "reference_runtime", "kernel_runtime"):
        m = re.search(rf'"{key}"\s*:?\s*(-?[0-9.]+(?:[eE][-+]?[0-9]+)?|null)', content)
        out[key] = None if (m is None or m.group(1) == "null") else float(m.group(1))
    m = re.search(r'"status"\s*:?\s*"([a-z_]+)"', content)
    out["status"] = m.group(1) if m else None
    m = re.search(r'"correctness_trials"\s*:\s*"\((\d+) / (\d+)\)"', content)
    out["correct_trials"] = [int(m.group(1)), int(m.group(2))] if m else None
    return out


def rounds(messages: list[dict]) -> list[dict]:
    """Return one entry per round with the kernel source and, for rounds 1 to 4, the feedback."""
    out = []
    n = len(messages) // 2
    for r in range(1, n + 1):
        assistant = messages[2 * r - 1]["content"]
        code = extract_code(assistant)
        fb = parse_feedback(messages[2 * r]["content"]) if 2 * r < len(messages) else None
        cut = assistant.find("```")
        out.append({"round": r, "code": code, "feedback": fb, "analysis": (assistant[:cut] if cut > 0 else "")[:4000]})
    return out
