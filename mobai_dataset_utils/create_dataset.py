import os
from argparse import ArgumentParser
import shutil


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--original", type=str, help="Path to original features", required=True
    )
    parser.add_argument("--new", type=str, help="path to new feature", required=True)
    parser.add_argument("--output", type=str, help="path to ouput", required=True)

    args = parser.parse_args()

    original = args.original
    new = args.new
    output = args.output

    original = os.path.abspath(original)
    new = os.path.abspath(new)
    output = os.path.abspath(output)

    if original[-1] != "/":
        original += "/"

    if new[-1] != "/":
        new += "/"

    if output[-1] != "/":
        output += "/"

    return original, new, output


def build_traversal_array():
    subdir1 = ["AGE", "FERET", "FRGC", "TUF"]
    subdir2 = ["1_training_set", "2_dev_set", "3_test_set"]
    subdir3 = ["1_male_source", "2_female_source"]
    traversal_array = []
    for dir1 in subdir1:
        for dir2 in subdir2:
            for dir3 in subdir3:
                traversal_array.append(dir1 + "/" + dir2 + "/" + dir3 + "/")
    return traversal_array


if __name__ == "__main__":
    original, new, output = get_arguments()
    traversal_array = build_traversal_array()

    data_full_name = "data_full"
    feature_bonafide_name = "Feature_Bonafide"
    feature_morphed_name = "Feature_Morphed"

    shutil.copytree(original + data_full_name, output + data_full_name)

    # copying bonafide over to output dataset
    shutil.copytree(original + feature_bonafide_name, output + feature_bonafide_name)

    for path in traversal_array:
        original_path = original + feature_morphed_name + "/" + path
        new_path = new + feature_morphed_name + "/" + path
        output_path = output + feature_morphed_name + "/" + path

        if not os.path.exists(original_path):
            continue

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for file_name in os.listdir(original_path):
            original_file_path = original_path + file_name
            new_file_path = new_path + file_name

            # if file is probe
            if file_name.find("probe") != -1:
                shutil.copy2(original_file_path, output_path)
                continue

            if os.path.exists(new_file_path) and os.path.exists(original_file_path):
                shutil.copy2(new_file_path, output_path)
