#!/bin/bash

#SBATCH -p TeslaV100
#SBATCH -J list_print
#SBATCH --output=train_enas_nni_search_%j.out
#SBATCH --gres=gpu
#SBATCH -c 6

#srun singularity exec --nv /home/albert/torch_albert.simg python retrain.py
srun singularity exec --nv /home/albert/torch_albert.simg python someTests.py

# srun singularity exec --nv /home/albert/torch_albert.simg python preprocess.py --folder -i ~/BDD/Kaggle/test -o ~/darts_data_nni/preprocessedTest224 --output_size 224
# srun singularity exec --bind /data_GPU/:/data_GPU/ --nv /data_GPU/albert/Containers/torch_albert.simg python retrain.py --arc-checkpoint ./checkpoint.json

# srun singularity exec --nv /home/albert/torch_albert.simg python dartsFastImplement.py

