import os

def get_system_output(command):
    stream = os.popen(command)
    return stream.read().rstrip('\n')


def tree(dir_path, depth=0, max_depth=None):
    if not max_depth or depth <= max_depth:
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                amount_of_ref = get_system_output(f"find {item_path} -type f -name 'ref*' | wc -l")
                amount_of_probe = get_system_output(f"find {item_path} -type f -name 'probe*' | wc -l")
                #fileCountString = f"         (Ref: {amount_of_ref} Probe: {amount_of_probe})"
                fileCountString = f"         (Document features: {amount_of_ref})"
                print("│  " * depth + "├── " + item +fileCountString)
                tree(item_path, depth + 1, max_depth)
            else:
                print("│  " * depth + "├── " + item)

# Example usage:
tree("./tmp/", max_depth=1)

