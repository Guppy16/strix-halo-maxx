# setup.sh - source me: `source setup.sh` (repo root) or from a script:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/setup.sh"
# (bash or zsh.) Sets the paths every script uses and creates the scratch workspace on first use. Quiet; no set -e/-u (it is sourced).
#
# Environment variables used in this repo:
#   SHM_REPO        repo root (physical path; set here from the location of this file)
#   SHM_SCRATCH     scratch / working dir, docker mounts it as /work. Default: <repo>/.scratch (gitignored; may be a symlink)
#   TAPNEXT_SRC     the TAPNext++ torch model.py (mounted read-only into containers)
#   TAPNEXT_WEIGHTS tapnextpp_256.safetensors (default: $SHM_SCRATCH/models/tapnextpp_256.safetensors)
#   MD_FRAMES       directory of moondream test frames (private, not in the repo)
#   IGNORE_PROCS    process patterns the GPU/NPU busy check ignores
#
# Layout of $SHM_SCRATCH (created here if missing, existing files and links are never touched):
#   npu-tapnext  -> $SHM_REPO/tapnext/npu        working copies the container sees under /work
#   npu-research -> $SHM_REPO/tapnext/research
#   npu-moondream-> $SHM_REPO/moondream/npu
#   tapnet/data/    tapvid_davis.pkl (TAP-Vid DAVIS), davis_reg.npy (made by tapnext/npu/make_davis_reg.py)
#   models/         model weights, e.g. tapnextpp_256.safetensors
export SHM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
mkdir -p "${SHM_SCRATCH:-$SHM_REPO/.scratch}" 2>/dev/null
export SHM_SCRATCH="$(cd "${SHM_SCRATCH:-$SHM_REPO/.scratch}" && pwd -P)"
mkdir -p "$SHM_SCRATCH/tapnet/data" "$SHM_SCRATCH/models" 2>/dev/null
for _shm_l in npu-tapnext:tapnext/npu npu-research:tapnext/research npu-moondream:moondream/npu; do
  [ -e "$SHM_SCRATCH/${_shm_l%%:*}" ] || [ -L "$SHM_SCRATCH/${_shm_l%%:*}" ] ||
    ln -s "$SHM_REPO/${_shm_l#*:}" "$SHM_SCRATCH/${_shm_l%%:*}"
done
unset _shm_l
