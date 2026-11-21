#!/bin/bash
# Secret-inference test on Penn State Roar Collab, one GPU: can request B's output name the
# secret word in the request before it? Ten victims differ only in that word.
#
#   sbatch --account=<allocation> gpu/vllm_leak/roar_infer.sbatch
#   KERNELS="tail rowtail correct" GAPS="0 1 4" LENGTHS="24 400" sbatch ...   # the defaults
#   ALLOC=nocache sbatch ...      # allocator cache off
#
# Then: python gpu/vllm_leak/analyze_infer.py $CBU/results/infer-<job>/*.json
#
# Expects /storage/work/$USER/cbu-vllm/env.sh and the venv from roar_vllm.sbatch.
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=cbu-infer
#SBATCH --output=/storage/work/%u/cbu-vllm/logs/infer-%j.out
set -uo pipefail
source /storage/work/$USER/cbu-vllm/env.sh
cd "$CBU/repo"
uv pip install -q -e gpu/vllm_leak
export VLLM_PLUGINS=cbu
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
OUT=$CBU/results/infer-$SLURM_JOB_ID
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nocache=""
[ "${ALLOC:-default}" = nocache ] && nocache=1
for kernel in ${KERNELS:-tail rowtail correct}; do
  for len in ${LENGTHS:-24 400}; do
    for gap in ${GAPS:-0 1 4}; do
      name=${kernel}_len${len}_gap${gap}_${ALLOC:-default}
      echo "== $name $(date +%T)"
      CBU_KERNEL=$kernel CBU_BLOCK=256 PYTORCH_NO_CUDA_MEMORY_CACHING=$nocache \
        timeout 40m python gpu/vllm_leak/run_infer.py --model "$MODEL" --length "$len" --gap "$gap" \
        --repeats "${REPEATS:-6}" --gpu-memory-utilization "${GPU_MEM:-0.5}" --out "$OUT/$name.json" \
        || echo "!! $name failed"
    done
  done
done
python gpu/vllm_leak/analyze_infer.py "$OUT"/*.json
