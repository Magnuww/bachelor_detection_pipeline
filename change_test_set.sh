#!/usr/bin/sh
testDataset="/home/dan/school/6th_semester/bachelor/bachelor_detection_pipeline/datasets/mordiff/runnable3"

output="/home/dan/school/6th_semester/bachelor/bachelor_detection_pipeline/datasets/unibo_w_mordiff_test/runnable"

python change_test_set.py --original $testDataset --output $output
