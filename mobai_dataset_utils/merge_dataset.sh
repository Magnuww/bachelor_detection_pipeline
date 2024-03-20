#!/bin/sh

uniboDataset="../datasets/unibo_2/"
mordiffDataset="../datasets/mordiff_2"
datasetsToCombine="$uniboDataset $mordiffDataset"

output="../datasets/balanced_mordiff_w_unibo"

echo "Starting merge"

python3 combine_datasets.py --datasets $datasetsToCombine \
	--output $output

echo "End merge"
