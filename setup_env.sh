module purge
# On the A100 nodes: `JZ_ARCH=a100 source setup_env.sh` (job.slurm does it from the account).
if [ -n "$JZ_ARCH" ]; then
    module load arch/$JZ_ARCH
fi
module load pytorch-gpu/py3/2.8.0

source "$WORK/venvs/tabpfm/bin/activate"

export HF_HOME="$WORK/hf_cache"
