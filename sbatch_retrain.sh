#!/bin/bash

#SBATCH -p TeslaV100
#SBATCH -J 0_008Retrain

#SBATCH --output=train_enas_nni_search_%j.out
#SBATCH --gres=gpu
#SBATCH -c 6
##SBATCH -w ns3186000

#srun singularity exec --nv /home/albert/torch_albert.simg python retrain.py

 srun singularity exec --nv /home/albert/torch_albert.simg python retrain.py

# srun singularity exec --bind /data_GPU/:/data_GPU/ --nv /data_GPU/albert/Containers/torch_albert.simg python retrain.py --arc-checkpoint ./checkpoint.json

# srun singularity exec --nv /home/albert/torch_albert.simg python dartsFastImplement.py

