
CONTAINER_IMAGE=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh
WORKDIR=/netscratch/$USER/mammoth

mkdir -p "$WORKDIR/logs"

args=(
  # --- baselines: standard MNISTMLP backbone
  '--model er --dataset seq-mnist --buffer_size 500 --minibatch_size 32 --lr 0.01 --n_epochs 5'
  '--model er --dataset rot-mnist --buffer_size 500 --minibatch_size 32 --lr 0.01 --n_epochs 5'
  '--model derpp --dataset seq-mnist --buffer_size 500 --minibatch_size 32 --alpha 0.5 --beta 0.5 --lr 0.01 --n_epochs 5'
  '--model derpp --dataset rot-mnist --buffer_size 500 --minibatch_size 32 --alpha 0.5 --beta 0.5 --lr 0.01 --n_epochs 5'
  '--model gem --dataset seq-mnist --buffer_size 500 --minibatch_size 32 --gamma 0.5 --lr 0.01 --n_epochs 5'
  '--model gem --dataset rot-mnist --buffer_size 500 --minibatch_size 32 --gamma 0.5 --lr 0.01 --n_epochs 5'
  # --- local variants: require local-mnistmlp backbone (fc1/fc2/local_head1/local_head2) 
  '--model local_er --dataset seq-mnist --backbone local-mnistmlp --buffer_size 500 --minibatch_size 32 --lr 0.01 --n_epochs 5'
  '--model local_er --dataset rot-mnist --backbone local-mnistmlp --buffer_size 500 --minibatch_size 32 --lr 0.01 --n_epochs 5'
  '--model local_derpp --dataset seq-mnist --backbone local-mnistmlp --buffer_size 500 --minibatch_size 32 --alpha 0.5 --beta 0.5 --lr 0.01 --n_epochs 5'
  '--model local_derpp --dataset rot-mnist --backbone local-mnistmlp --buffer_size 500 --minibatch_size 32 --alpha 0.5 --beta 0.5 --lr 0.01 --n_epochs 5'
  '--model local_gem --dataset seq-mnist --backbone local-mnistmlp --buffer_size 500 --minibatch_size 32 --gamma 0.5 --lr 0.01 --n_epochs 5'
  '--model local_gem --dataset rot-mnist --backbone local-mnistmlp --buffer_size 500 --minibatch_size 32 --gamma 0.5 --lr 0.01 --n_epochs 5'
)

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-workdir="$WORKDIR" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro \
  bash -c "pip install -q -r requirements.txt quadprog==0.1.11 && python main.py ${args[$SLURM_ARRAY_TASK_ID]}"
