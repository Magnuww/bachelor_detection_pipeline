#!/usr/bin/sh

originalDataset="datasets/unibo/runnable"
newDataset="datasets/mordiff/full_morphed_features"

outputDataset="datasets/mordiff/runnable3"

python create_dataset.py --original $originalDataset --new $newDataset --output $outputDataset
