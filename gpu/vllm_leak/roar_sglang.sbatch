#!/bin/bash
# The vLLM leak test repeated in SGLang on Penn State Roar Collab, one A100.
#
#   sbatch --account=<allocation> gpu/vllm_leak/roar_sglang.sbatch            # 5 kernels x graphs/eager x sequential/batched
#   SMOKE=1 sbatch --account=<allocation> gpu/vllm_leak/roar_sglang.sbatch    # defective and correct, eager, one repeat
#
# Expects /storage/work/$USER/cbu-vllm/env.sh and a venv at $CBU/venv-sglang with sglang==0.5.9.
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=cbu-sglang
#SBATCH --output=/storage/work/%u/cbu-vllm/logs/sglang-%j.out
set -uo pipefail
source /storage/work/$USER/cbu-vllm/env.sh
deactivate 2>/dev/null
source "$CBU/venv-sglang/bin/activate"
cd "$CBU/repo"
export PYTHONPATH=$PWD/gpu/vllm_leak/sglang_patch:$PWD/gpu/vllm_leak
OUT=$CBU/results/sglang-$SLURM_JOB_ID
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

run() {  # kernel mode order [extra args]
  local kernel=$1 mode=$2 order=$3; shift 3
  local name=${kernel}_${mode}_default_${order} block=256
  [ "$kernel" = floor ] && block=1024
  echo "== $name $(date +%T)"
  CBU_KERNEL=$kernel CBU_BLOCK=$block timeout 40m \
    python gpu/vllm_leak/run_leak_sglang.py --mode "$mode" --order "$order" --out "$OUT/$name.json" "$@" \
    > "$OUT/$name.log" 2>&1
  echo "   exit $? $(grep -m1 '"summary"' "$OUT/$name.log" | grep -o '"summary".*' | cut -c1-300)"
}

if [ -n "${SMOKE:-}" ]; then
  for k in defective correct; do run $k eager sequential --repeats 1; done
  exit 0
fi
for k in ${KERNELS:-defective floor tail rowtail correct}; do
  for mode in eager graphs; do
    for order in sequential batched; do run $k $mode $order; done
  done
done
