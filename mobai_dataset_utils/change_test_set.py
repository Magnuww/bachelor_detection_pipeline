import os
from argparse import ArgumentParser
import shutil

# WARNING: this script will remove the old test set


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--original", type=str, help="Path to dataset with test", required=True
    )
    parser.add_argument(
        "--output",
        type=str,
        help="path to dataset that gets its test set swapped",
        required=True,
    )

    args = parser.parse_args()

    original = args.original
    output = args.output

    original = os.path.abspath(original)
    output = os.path.abspath(output)

    if original[-1] != "/":
        original += "/"

    if output[-1] != "/":
        output += "/"

    return original, output


def build_traversal_array():
    subdir1 = ["Feature_Morphed", "Feature_Bonafide"]
    subdir2 = ["AGE", "FERET", "FRGC", "TUF"]

    traversal_array = []
    for dir1 in subdir1:
        for dir2 in subdir2:
            traversal_array.append(dir1 + "/" + dir2 + "/")
    return traversal_array


if __name__ == "__main__":
    original, output = get_arguments()
    traversal_array = build_traversal_array()

    test_set_name = "3_test_set"
    for path in traversal_array:
        original_path = original + "/" + path
        output_path = output + "/" + path

        if not os.path.exists(original_path + test_set_name):
            continue

        if os.path.exists(
            output_path + test_set_name
        ):  # remove the old symlink/directory
            print("removing: ", output + test_set_name)
            shutil.rmtree(output_path + test_set_name)
        elif not os.path.exists(output_path):
            print("creating direcotry: ", output_path)
            os.makedirs(output_path)

        # make symlink
        source = original_path + test_set_name
        destination = output_path + test_set_name

        print("Making symlink \nFrom: " + source + "\nTo: " + destination)
        os.symlink(source, destination)
