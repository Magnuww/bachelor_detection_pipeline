import os


def build_traversal_array(current_dir, subdirs):
    if len(subdirs) == 0:
        return [current_dir]
    subdir1 = subdirs[0]
    traversal_array = []
    for entry in subdir1:
        new_current_dir = os.path.join(current_dir, entry)
        traversal_array.extend(build_traversal_array(new_current_dir, subdirs[1:]))
    return traversal_array


def symlink_dataset(original, output, traversal_array):
    original_name = os.path.basename(original[:-1])
    for path in traversal_array:
        original_path = os.path.join(original, path)
        output_path = os.path.join(output, path)

        if not os.path.exists(original_path):
            continue

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for file in os.listdir(original_path):
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


def create_data_full(output):
    print("Creating data full folders")

    data_full_path = os.path.join(output, "data_full")
    os.makedirs(data_full_path)
