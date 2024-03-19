#!/bin/sh
#SBATCH --job-name="mordiff svm"
#SBATCH --partition=CPUQ
#SBATCH --account="ie-idi"
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --output=../../logs/mordiff/log3.txt
#SBATCH --mail-user=danvt@stud.ntnu.no
#SBATCH --mail-type=ALL

bonaFideFeatures="../../datasets/mordiff/runnable3/Feature_Bonafide/"  
morphedAttackFeatures="../../datasets/mordiff/runnable3/Feature_Morphed/"  
modelOutput="../../models/mordiff_morph_3/"

module load Anaconda3/2023.09-0

conda activate mobai-svm

echo "starting mordiff svm"

python3 svm_training_pipeline.py --bonaFideFeatures $bonaFideFeatures --morphedAttackFeatures $morphedAttackFeatures --modelOutput $modelOutput

echo "done running svm"




