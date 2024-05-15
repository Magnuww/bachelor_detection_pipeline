import os
import sys

sys.path.append("../mobai_svm")
from argparse import ArgumentParser

from data_loader import data_loader


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="The dataset that gets checked",
        required=True,
    )

    args = parser.parse_args()
    dataset = args.dataset

    return dataset


if __name__ == "__main__":
    datasets = get_arguments()
    bonafideFeatureFolder = os.path.join(datasets, "Feature_Bonafide")
    morphedFeatureFolder = os.path.join(datasets, "Feature_Morphed")

    train_loader = data_loader(
        morphedFeatureFolder, bonafideFeatureFolder, flag="1_training_set"
    )

    test_loader = data_loader(
        morphedFeatureFolder, bonafideFeatureFolder, flag="3_test_set"
    )

    print(train_loader.paths_morphed)
