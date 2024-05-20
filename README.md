# Code repository for bachelor thesis 
This repository contains code and scripts for the bachelor thesis "Detect manipulated face Images using deep learning tools". This repository will mostly contain code for training and testing for the different models in the thesis. The other 
part will contain various utility scripts for preprocessing, and modifying data.

It is to be noted that the code in this repository is created for the purpose of
achieving results quick and fast.

## Directory structure of repository
This repository has 4 main directories: classifiers, dataset_utils, multiple_feature_shapes
and morph_utils


### Classifiers
Classifiers is the directory for the different training, testing code for 
classifiers used in the thesis.

Like the modified SVM from MOBAI AS, dualsvm, dual neural network and the neural network.

This directory has an additional README.md for how to train and test.

### dataset_utils

The datset_utils folder is for utilities and scripts used for preparing dataset.
This directory includes files like renaming scripts and dataset validation scripts.

### multiple_feat_shapes 

### morph utils

The morph utils directory contains various scripts used to make the 
morphing algorithms run with the desired format.


## Using the modified SVM classifier 

### How to train with a new dataset 
Step 1: 

Run create_dataset.py in mobai_dataset_utils:


Example:
```bash
python create_dataset.py \
  --original datasets/unibo/ \
  --new datasets/mordiff/ \
  --output datasets/mordiff_runnable/

```

Step 2:

Run the Svm with the dataset created in previous step. 

Usage:
```bash
python svm_training_pipeline.py --bonaFideFeatures BONAFIDEFEATURES \
                                --morphedAttackFeatures MORPHEDATTACKFEATURES \
                                --modelOutput MODELOUTPUT \
```




If assert errors occur try removing all the .DS_store files.

You can do this by moving to the desired folder and running:
```bash
trash **/.DS_store
```


### How to test with another dataset 

Step 1: 

Copy the model folder, since the results will be overwritten.

Step 2:

Example usage:
```bash

python3 svm_testing_pipeline.py --bonaFideFeatures unibo/Feature_Bonafide \
	--morphedAttackFeatures  unibo/Feature_Morphed \
	--modelOutput mordiff_runnable/model
```


### How to merge/combine datasets

<!-- These instructions are for merging 2 or more datasets, whilst keeping the size -->
<!-- proportional.  -->

Step 1:

If the datasets does not have any probe images in it run create_dataset.py 
from mobai_dataset_utils with unibo dataset as original.

Example:
```bash
python create_dataset.py \
  --original datasets/unibo/ \
  --new datasets/mordiff/ \
  --output datasets/mordiff_runnable/

```

Do this for each dataset that is gonna be merged/combined.


Step 2:


To merge the datasets whilst keeping the size proportional run. 

Run mege_dataset.py in mobai_dataset_utils


Example usage: 
```bash
python merge_dataset.py --datasets UNIBO MORDIFF MIPGAN --ouput MERGED
```

To combine the datasets fully 

Run combine_dataset.py in mobai_dataset_utils

Example usage:

```bash
python combine_dataset.py --datasets UNIBO MORDIFF MIPGAN --ouput COMBINED
```

## Citations

### mordiff

This repository has utilities and code for running among other things the MORDIFF morphing algorithm.

The github repository for mordiff: [mordiff](https://github.com/naserdamer/MorDIFF)

MORDIFF paper: [mordiff_paper](https://arxiv.org/abs/2302.01843)

Mordiff SYN-mad benchmark: [syn-mad](https://doi.org/10.1109/IJCB54206.2022.10007950)

### MIPGAN


This repository has utilities and code for running among other things the MIPGAN morphing algorithm.

Mipgan github repository: [mipgan](https://github.com/ZHYYYYYYYYYYYY/MIPGAN-face-morphing-algorithm)
