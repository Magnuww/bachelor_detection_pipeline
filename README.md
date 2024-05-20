# Directory structure of repository
This repository has 4 main directories: classifiers, dataset_utils and 
multiple_feat_shapes. 


## Classifiers
Classifiers is the directory for the different training, testing code for 
classifiers used in the thesis.

Like the modified SVM from MOBAI AS, dualsvm, dual neural network and the neural network.

This directory has an additional README.md for how to train and test.

## dataset_utils

The datset_utils folder is for utilities and scripts used for preparing dataset.
This directory includes files like renaming scripts and dataset validation scripts.

## multiple_feat_shapes 

# Using the modified SVM classifier 

## How to train with a new dataset 
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


## How to test with another dataset 

Step 1: 

Copy the model folder, since the results will be overwritten.

Step 2:

Example usage:
```bash

python3 svm_testing_pipeline.py --bonaFideFeatures unibo/Feature_Bonafide \
	--morphedAttackFeatures  unibo/Feature_Morphed \
	--modelOutput mordiff_runnable/model
```


## How to merge/combine datasets

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

