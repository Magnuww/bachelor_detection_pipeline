import os

# count the number of subjects in dataset


def get_system_output(command):
    stream = os.popen(command)
    return stream.read().rstrip("\n")


def tree(dir_path, depth=0, max_depth=None):
    if not max_depth or depth <= max_depth:
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                # amount_of_ref = get_system_output(f"find {item_path} -type f -name 'ref*' | wc -l")
                # amount_of_probe = get_system_output(f"find {item_path} -type f -name 'probe*' | wc -l")
                amount_of_subjects = get_system_output(
                    f"find {item_path} -type f -name '*.jpg' | sed 's/.....$//' | uniq | wc -l"
                )
                # fileCountString = f"         (Ref: {amount_of_ref} Probe: {amount_of_probe})"
                # subjects = get_system_output(f"ls -1 {amount_of_pngs}| wc -l")
                fileCountString = f"         (amount_of_subjects: {amount_of_subjects})"
                print("│  " * depth + "├── " + item + fileCountString)
                tree(item_path, depth + 1, max_depth)
            else:
                print("│  " * depth + "├── " + item)


# Example usage:
tree("./Data_Bonafide/", max_depth=2)
