import os
import shutil
from argparse import ArgumentParser

# copies all files from new features that is also in the original
# also copies the probe files from original
# assumes new has more images than original
if __name__ == "__main__":
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

    subdir1 = ["AGE", "FERET", "FRGC", "TUF"]
    subdir2 = ["1_training_set", "2_dev_set", "3_test_set"]
    subdir3 = ["1_male_source", "2_female_source"]

    # TODO: REWRITE if using for proper script in the future
    for dir1 in subdir1:
        path1 = dir1
        if not os.path.exists(original + path1):
            continue
        for dir2 in subdir2:
            path2 = path1 + "/" + dir2
            if not os.path.exists(original + path2):
                continue
            for dir3 in subdir3:
                path3 = path2 + "/" + dir3
                if not os.path.exists(original + path3):
                    continue
                for file in os.listdir(original + path3):
                    # copy if file exists in original
                    file_in_new = new + path3 + "/" + file
                    file_in_original = original + path3 + "/" + file
                    output_folder = output + path3 + "/"
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    if os.path.exists(file_in_original) and file.find("probe") == -1:
                        if os.path.exists(file_in_new):
                            shutil.copy2(file_in_new, output_folder)
                        else:
                            print("file not found" + file_in_new)

                    # if file is probe
                    if file.find("probe") != -1:
                        shutil.copy2(file_in_original, output_folder)
