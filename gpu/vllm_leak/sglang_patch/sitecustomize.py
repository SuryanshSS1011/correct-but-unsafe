"""Swap SGLang's silu_and_mul kernel for the study's Triton kernel in every process.

SGLang's SiluAndMul.forward_cuda allocates its output with torch.empty and
calls silu_and_mul(x, out), imported from sgl_kernel into
sglang.srt.layers.activation. SGLang runs the model in spawned processes, so
this file, put on PYTHONPATH as a sitecustomize, replaces that name as soon as
the module is imported in any of them. It does nothing unless CBU_KERNEL names
one of cbu_vllm_plugin.KINDS, which then selects the kernel as for vLLM.
"""

import importlib.abc
import importlib.machinery
import os
import sys

TARGET = "sglang.srt.layers.activation"


def _patch(module) -> None:
    from cbu_vllm_plugin import launcher

    launch = launcher(os.environ["CBU_KERNEL"])

    def silu_and_mul(x, out):
        launch(out, x)

    module.silu_and_mul = silu_and_mul


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def run(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = run
        return spec


if os.environ.get("CBU_KERNEL"):
    from cbu_vllm_plugin import KINDS

    if os.environ["CBU_KERNEL"] in KINDS:
        sys.meta_path.insert(0, _Hook())
