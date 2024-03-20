import os
from utils import build_traversal_array
from utils import create_data_full
from utils import symlink_dataset
from argparse import ArgumentParser

import random


# TODO: Make general
def get_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="The datasets that will be combined",
        required=True,
    )
    parser.add_argument(
        "--output", type=str, help="Path to the output dataset", required=True
    )

    args = parser.parse_args()
    datasets = args.datasets
    output = args.output

    # NOTE:  could probably use a proper path handler instead
    for i in range(0, len(datasets)):
        datasets[i] = os.path.abspath(datasets[i])
        if datasets[i][-1] != "/":
            datasets[i] += "/"

    output = os.path.abspath(output)
    if output[-1] != "/":
        output += "/"

    return datasets, output


def symlink_dataset_balanced(original, output, traversal_array, probability):
    original_name = os.path.basename(original[:-1])
    for path in traversal_array:
        original_path = os.path.join(original, path)
        output_path = os.path.join(output, path)

        if not os.path.exists(original_path):
            continue

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for file in os.listdir(original_path):
            if random.uniform(0, 1) > probability:
                continue

            if os.path.exists(os.path.join(output_path, file)):
                continue
            if "probe" in file:
                new_filename = file
            else:
                new_filename = file.split(".")
                new_filename = (
                    new_filename[0] + "_" + original_name + "." + new_filename[1]
                )

            source = os.path.join(original_path, file)
            destination = os.path.join(output_path, new_filename)
            print("Symlinking from: " + source + ", to: " + destination)
            os.symlink(source, destination)


def create_traversal_array():
    subdir1 = ["Feature_Bonafide", "Feature_Morphed"]
    subdir2 = ["AGE", "FERET", "FRGC", "TUF"]
    subdir3 = ["1_training_set", "2_dev_set", "3_test_set"]
    subdir4 = ["1_male_source", "2_female_source"]

    subdirs = [subdir1, subdir2, subdir3, subdir4]

    current_dir = ""
    traversal_array = build_traversal_array(current_dir, subdirs)
    return traversal_array


if __name__ == "__main__":
    datasets, output = get_arguments()

    traversal_array = create_traversal_array()

    probability_for_each_dataset = 1 / len(datasets)
    for dataset in datasets:
        symlink_dataset_balanced(
            dataset, output, traversal_array, probability_for_each_dataset
        )

    create_data_full(output)
