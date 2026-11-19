# Source this in every job and shell for the vLLM leak test on Roar Collab.
# Home is over quota, so every cache and config path points to work or scratch.
export CBU=/storage/work/$USER/cbu-vllm
export CBU_SCRATCH=/scratch/$USER/cbu-vllm
export HF_HOME=$CBU/hf
export XDG_CACHE_HOME=$CBU_SCRATCH/cache
export XDG_CONFIG_HOME=$CBU_SCRATCH/config
export UV_CACHE_DIR=$CBU_SCRATCH/uv
export UV_LINK_MODE=copy  # the uv cache (scratch) and the venv (work) are on different filesystems
export PIP_CACHE_DIR=$CBU_SCRATCH/pip
export VLLM_CACHE_ROOT=$CBU_SCRATCH/vllm
export VLLM_CONFIG_ROOT=$CBU_SCRATCH/vllm-config
export TORCHINDUCTOR_CACHE_DIR=$CBU_SCRATCH/inductor
export CUDA_CACHE_PATH=$CBU_SCRATCH/nv
export FLASHINFER_WORKSPACE_BASE=$CBU_SCRATCH  # FlashInfer ignores XDG_CACHE_HOME and defaults to $HOME
export TRITON_HOME=$CBU_SCRATCH
export TORCH_HOME=$CBU_SCRATCH/torch
export TORCH_EXTENSIONS_DIR=$CBU_SCRATCH/torch_extensions
export XDG_DATA_HOME=$CBU_SCRATCH/data
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# Inside a job, use the node's local disk for temporary files and a fresh Triton cache.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  export TMPDIR=/tmp/cbu-$SLURM_JOB_ID
  export TRITON_CACHE_DIR=$TMPDIR/triton
else
  export TMPDIR=$CBU_SCRATCH/tmp
  export TRITON_CACHE_DIR=$CBU_SCRATCH/triton
fi
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$UV_CACHE_DIR" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT"
module load python/3.12.12 uv/0.7.22 gcc/13.2.0
if [ -f "$CBU/venv/bin/activate" ]; then source "$CBU/venv/bin/activate"; fi
# Inside a job, anything else that writes to ~ lands in work storage, not the full home.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  export HOME=$CBU/jobhome
  mkdir -p "$HOME"
  # A package in the software stack loads gcc 8.5's libstdc++ first, which lacks the
  # GLIBCXX_3.4.32 that FlashInfer's JIT kernels, built with gcc 13.2, need.
  export LD_PRELOAD=/storage/icds/RISE/sw8/gcc/gcc-13.2.0/lib64/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}
fi
