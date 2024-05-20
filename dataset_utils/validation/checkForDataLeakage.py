import os
from argparse import ArgumentParser

if __name__ == "__main__":
    # NOTE: required arguments are usually bad form, but we dont care in this case
    # https://docs.python.org/3/library/argparse.html#required
    parser = ArgumentParser()
    parser.add_argument("--feature", type=str, help="Path to feature", required=True)

    args = parser.parse_args()

    feature_path = args.feature

    feature_path = os.path.abspath(feature_path)

    if feature_path[-1] != "/":
        feature_path += "/"

    allFileNames = []
    fileNameLeakage = []
    # AGE, FERET, FRGC, TUF
    for root, dirs, files in os.walk(feature_path):
        for name in files:
            name = name.split(".")[0]
            root_name = root + "\\" + name
            root_name_split = root_name.split("\\")
            database = root_name_split[-4].split("/")[-1]
            database_gender_name = (
                database + "/" + root_name_split[-2] + "/" + root_name_split[-1]
            )

            if database_gender_name in allFileNames and (name != ""):
                fileNameLeakage.append(database_gender_name)
            else:
                # print(gender_name)
                allFileNames.append(database_gender_name)

    if len(fileNameLeakage) == 0:
        print("No leakage found")
    else:
        print("Leakage found")
        print(fileNameLeakage)

