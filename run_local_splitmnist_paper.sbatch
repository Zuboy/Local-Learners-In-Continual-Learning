#!/bin/bash
#SBATCH --job-name=local_splitmnist_paper
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/netscratch/chandel/mammoth-runtime/logs/local_splitmnist_paper_%j.out
#SBATCH --error=/netscratch/chandel/mammoth-runtime/logs/local_splitmnist_paper_%j.err

set -euo pipefail

srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-workdir=/netscratch/chandel/mammoth \
  --container-mounts=/netscratch/chandel:/netscratch/chandel,/ds:/ds:ro \
  bash -lc '
    set -euo pipefail
    cd /netscratch/chandel/mammoth
    export PIP_CACHE_DIR=/netscratch/chandel/mammoth-runtime/pip-cache

    python -m pip install -q -r requirements.txt open-clip-torch==2.32.0 quadprog==0.1.11

    for SEED in $(seq 1 10); do
      python main.py --model local_er --dataset seq-mnist --backbone local-mnistmlp --mlp_hidden_size 1400 --lr 0.001 --n_epochs 2 --batch_size 512 --buffer_size 500 --minibatch_size 512 --seed "$SEED" --permute_classes 1 --enable_other_metrics 1 --device cuda:0 --num_workers 4 --base_path /netscratch/chandel/mammoth-runtime/data --results_path "../results/paper-rq2/seed_${SEED}"

      python main.py --model local_derpp --dataset seq-mnist --backbone local-mnistmlp --mlp_hidden_size 1400 --lr 0.001 --n_epochs 2 --batch_size 512 --buffer_size 500 --minibatch_size 512 --alpha 0.5 --beta 0.5 --seed "$SEED" --permute_classes 1 --enable_other_metrics 1 --device cuda:0 --num_workers 4 --base_path /netscratch/chandel/mammoth-runtime/data --results_path "../results/paper-rq2/seed_${SEED}"

      python main.py --model local_gem --dataset seq-mnist --backbone local-mnistmlp --mlp_hidden_size 1400 --lr 0.001 --n_epochs 2 --batch_size 512 --buffer_size 500 --minibatch_size 20 --gamma 0.5 --seed "$SEED" --permute_classes 1 --enable_other_metrics 1 --device cuda:0 --num_workers 4 --base_path /netscratch/chandel/mammoth-runtime/data --results_path "../results/paper-rq2/seed_${SEED}"
    done
  '
