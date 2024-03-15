#!/bin/sh
#SBATCH --job-name="svm"
#SBATCH --partition=GPUQ
#SBATCH --account="ie-idi"
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --output=../logs/mobai-svm/log.txt
#SBATCH --mail-user=<ADD email here>
#SBATCH --mail-type=ALL

bonaFideFeatures="../datasets/unibo/runnable/Feature_Bonafide/"
morphedAttackFeatures="../datasets/mordiff/runnable/Feature_Morphed/"
modelOutput="../models/mordiff_morph_2/"

module load Anaconda3/2023.09-0

conda activate mobai-svm

echo "starting svm"

bonaFideFeatures="../datasets/unibo/runnable/Feature_Bonafide/"
morphedAttackFeatures="../datasets/mordiff/runnable/Feature_Morphed/"
modelOutput="../models/mordiff_morph_2/"

python3 svm_training_pipeline.py --bonaFideFeatures $bonaFideFeatures --morphedAttackFeatures $morphedAttackFeatures --modelOutput $modelOutput

echo "done running svm"
