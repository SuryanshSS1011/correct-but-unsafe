#!/bin/bash
# vLLM cross-request leak test on Penn State Roar Collab, one GPU.
#
#   sbatch --account=<allocation> gpu/vllm_leak/roar_vllm.sbatch     # the original 16 configurations and recording runs
#   SET=subtle sbatch ...       # the floor and tail defects, all 16 configurations each, and their recording runs
#   SET=nocache sbatch ...      # allocator cache off, eager only: vLLM cannot capture CUDA graphs without it
#   SET=rowtail sbatch ...      # one element unwritten in every row, all 16 configurations and the recording run
#   SET=shapes sbatch ...       # logs the rows each call sees and whether it runs inside a CUDA graph capture
#   SET=model MODEL=<hf id> sbatch ...                   # defective, tail, rowtail and correct on another model
#   SET=model sbatch --gres=gpu:a40:1 ...                # the same on another GPU type
#   KERNELS="..." overrides the kernels of a set, CHAT=1 wraps B's prompt in the chat template.
#   SMOKE="floor tail" sbatch ...   # recording runs for these kernels, one repeat
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
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
OUT=$CBU/results/vllm-$SLURM_JOB_ID
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "set=${SET:-full} model=$MODEL"

run() {  # kernel mode alloc order [extra args]
  local kernel=$1 mode=$2 alloc=$3 order=$4; shift 4
  local name=${kernel}_${mode}_${alloc}_${order}
  [[ " $* " == *" --record "* ]] && name=${name}_record
  local conf="" nocache="" block=256
  [ "$alloc" = expandable ] && conf=expandable_segments:True
  [ "$alloc" = nocache ] && nocache=1
  [ "$kernel" = floor ] && block=1024
  echo "== $name $(date +%T)"
  local shapes=""
  [ "${SET:-}" = shapes ] && shapes=$OUT/$name.shapes
  CBU_KERNEL=$kernel CBU_BLOCK=$block PYTORCH_CUDA_ALLOC_CONF=$conf PYTORCH_NO_CUDA_MEMORY_CACHING=$nocache \
    CBU_SHAPES=$shapes timeout 40m python gpu/vllm_leak/run_leak.py --model "$MODEL" --mode "$mode" \
    --order "$order" --gpu-memory-utilization "${GPU_MEM:-0.5}" ${CHAT:+--chat} --out "$OUT/$name.json" "$@" \
    > "$OUT/$name.log" 2>&1
  echo "   exit $? $(grep -m1 '"summary"' "$OUT/$name.log" | grep -o '"summary".*' | cut -c1-300)"
}

matrix() {  # kernels allocs
  for k in $1; do
    for mode in eager graphs; do
      for alloc in $2; do
        for order in sequential batched; do run $k $mode $alloc $order; done
      done
    done
  done
}

if [ -n "${SMOKE:-}" ]; then
  for k in $SMOKE; do run $k eager "${ALLOC:-default}" sequential --record --repeats 1; done
  exit 0
fi
case ${SET:-full} in
  full)
    for k in defective correct; do run $k eager default sequential --record; done
    matrix "defective correct" "default expandable" ;;
  subtle)
    for k in floor tail; do run $k eager default sequential --record; done
    matrix "floor tail" "default expandable" ;;
  rowtail)
    for k in ${KERNELS:-rowtail}; do run $k eager default sequential --record; done
    matrix "${KERNELS:-rowtail}" "default expandable" ;;
  shapes)
    for k in ${KERNELS:-tail rowtail}; do
      for mode in eager graphs; do
        for order in sequential batched; do run $k $mode default $order --repeats 1; done
      done
    done ;;
  nocache)
    for k in ${KERNELS:-defective floor rowtail correct}; do run $k eager nocache sequential --record; done
    for k in ${KERNELS:-defective floor rowtail correct}; do
      for order in sequential batched; do run $k eager nocache $order; done
    done ;;
  model)
    for k in ${KERNELS:-defective tail rowtail correct}; do run $k eager default sequential --record; done
    matrix "${KERNELS:-defective tail rowtail correct}" default ;;
esac
