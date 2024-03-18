from argparse import ArgumentParser
import os

# TODO: make os agnostic


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


def build_traversal_array():
    subdir1 = ["Feature_Bonafide", "Feature_Morphed"]
    subdir2 = ["AGE", "FERET", "FRGC", "TUF"]
    subdir3 = ["1_training_set", "2_dev_set", "3_test_set"]
    subdir4 = ["1_male_source", "2_female_source"]
    traversal_array = []
    for dir1 in subdir1:
        for dir2 in subdir2:
            for dir3 in subdir3:
                for dir4 in subdir4:
                    traversal_array.append(os.path.join(dir1, dir2, dir3, dir4))
    return traversal_array


def symlink_dataset(original, output, traversal_array):
    original_name = os.path.basename(original)
    for path in traversal_array:
        original_path = os.path.join(original, path)
        output_path = os.path.join(output, path)

        if not os.path.exists(original_path):
            continue

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for file in os.listdir(original_path):
            new_filename = file.split(".")
            new_filename = new_filename[0] + original_name + "." + new_filename[1]
            source = os.path.join(original_path, file)
            destination = os.path.join(output_path, new_filename)
            print("Symlinking from: " + source + ", to: " + destination)
            os.symlink(source, destination)


def create_data_full(output):
    print("Creating data full folders")

    data_full_path = os.path.join(output, "data_full")
    os.makedirs(data_full_path)
    files = [
        "1_training_set_full_labels.txt",
        "1_training_set_full_paths.txt",
        "3_test_set_full_labels.txt",
        "3_test_set_full_paths.txt",
    ]
    for file in files:
        with open(file, "w") as file:
            pass


if __name__ == "__main__":
    datasets, output = get_arguments()
    traversal_array = build_traversal_array()

    for dataset in datasets:
        symlink_dataset(dataset, output, traversal_array)

    create_data_full(output)
