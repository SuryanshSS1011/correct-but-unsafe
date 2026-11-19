#!/bin/bash
# vLLM cross-request leak test on Penn State Roar Collab, one A100.
#
#   sbatch --account=<allocation> gpu/vllm_leak/roar_vllm.sbatch              # all 16 configurations and the recording runs
#   SMOKE=1 sbatch --account=<allocation> gpu/vllm_leak/roar_vllm.sbatch      # the two recording runs, one repeat
#   RECORD_ONLY=1 sbatch --account=<allocation> gpu/vllm_leak/roar_vllm.sbatch  # the two recording runs, three repeats
#
# Expects /storage/work/$USER/cbu-vllm/env.sh, a copy of gpu/vllm_leak/roar_env.sh, and the venv it
# activates with vllm==0.34.0 installed.
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=cbu-vllm
#SBATCH --output=/storage/work/%u/cbu-vllm/logs/vllm-%j.out
set -uo pipefail
source /storage/work/$USER/cbu-vllm/env.sh
cd "$CBU/repo"
uv pip install -q -e gpu/vllm_leak
export VLLM_PLUGINS=cbu
OUT=$CBU/results/vllm-$SLURM_JOB_ID
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

run() {  # kernel mode alloc order [extra args]
  local kernel=$1 mode=$2 alloc=$3 order=$4; shift 4
  local name=${kernel}_${mode}_${alloc}_${order}
  [[ " $* " == *" --record "* ]] && name=${name}_record
  local conf=""
  [ "$alloc" = expandable ] && conf=expandable_segments:True
  echo "== $name $(date +%T)"
  CBU_KERNEL=$kernel PYTORCH_CUDA_ALLOC_CONF=$conf timeout 40m \
    python gpu/vllm_leak/run_leak.py --mode "$mode" --order "$order" --out "$OUT/$name.json" "$@" \
    > "$OUT/$name.log" 2>&1
  echo "   exit $? $(grep -m1 '"summary"' "$OUT/$name.log" | cut -c1-400)"
}

if [ -n "${RECORD_ONLY:-}" ]; then
  for k in defective correct; do run $k eager default sequential --record; done
  exit 0
fi
if [ -n "${SMOKE:-}" ]; then
  for k in defective correct; do run $k eager default sequential --record --repeats 1; done
  exit 0
fi
for k in defective correct; do run $k eager default sequential --record; done
for k in defective correct; do
  for mode in eager graphs; do
    for alloc in default expandable; do
      for order in sequential batched; do run $k $mode $alloc $order; done
    done
  done
done
