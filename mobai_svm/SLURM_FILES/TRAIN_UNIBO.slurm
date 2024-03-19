#!/bin/sh
#SBATCH --job-name="unibo svm"
#SBATCH --partition=GPUQ
#SBATCH --account="ie-idi"
#SBATCH --time=1-23:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1  
#SBATCH --mem=30G
#SBATCH --output=../../logs/unibo/log.txt
#SBATCH --mail-user=danvt@stud.ntnu.no
#SBATCH --mail-type=ALL

bonaFideFeatures="../../datasets/unibo/runnable/Feature_Bonafide/"  
morphedAttackFeatures="../../datasets/unibo/runnable/Feature_Morphed/"  
modelOutput="../../models/unibo_morph_2/"

module load Anaconda3/2023.09-0

conda activate mobai-svm

echo "starting unibo svm"

python3 svm_training_pipeline.py --bonaFideFeatures $bonaFideFeatures --morphedAttackFeatures $morphedAttackFeatures --modelOutput $modelOutput

echo "done running svm"




