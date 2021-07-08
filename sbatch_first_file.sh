#!/bin/bash

#SBATCH -p TeslaV100
#SBATCH -J enas_search
#SBATCH --output=train_darts_nni_search_%j.out
#SBATCH --gres=gpu
#SBATCH -c 6

#srun singularity exec --nv /home/albert/torch_albert.simg python retrain.py

 srun singularity exec --nv /home/albert/torch_albert.simg python search.py 
# srun singularity exec --bind /data_GPU/:/data_GPU/ --nv /data_GPU/albert/Containers/torch_albert.simg python retrain.py --arc-checkpoint ./checkpoint.json

# srun singularity exec --nv /home/albert/torch_albert.simg python dartsFastImplement.py

