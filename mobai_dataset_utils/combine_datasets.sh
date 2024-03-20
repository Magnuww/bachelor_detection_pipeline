#!/bin/sh

uniboDataset="./datasets/unibo_2/"
mordiffDataset="./datasets/mordiff_2"
datasetsToCombine="$uniboDataset $mordiffDataset"

output="./datasets/mordiff_w_unibo"

echo "Starting combine"

python3 combine_datasets.py --datasets $datasetsToCombine \
	--output $output

echo "End combine"
