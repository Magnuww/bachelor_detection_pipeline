import os
import shutil
from typing import List

results_path = "../results3/"

test_sets_path = "../datasets/tests/"

models_path = "../models/tests/"

tests = os.listdir(test_sets_path)

models = os.listdir(models_path)

RESULT_NAME = "plot_metrics.png"


def split_path(path) -> List[str]:
    # Split the path into its individual directories
    directories = []
    while True:
        path, directory = os.path.split(path)
        if directory != "":
            directories.append(directory)
        else:
            if path != "":
                directories.append(path)
            break

    # Reverse the list to get the directories in the correct order
    directories.reverse()
    return directories


if __name__ == "__main__":
    for test in tests:
        test_path = os.path.join(test_sets_path, test)

        for model in models:
            model_path = os.path.join(models_path, model)

            bonafideFeaturePath = os.path.join(test_path, "Feature_Bonafide")
            morphedFeaturePath = os.path.join(test_path, "Feature_Morphed")

            test_name = os.path.basename(test_path)

            result_name = model + "_test_" + test_name
            new_result_path = os.path.join(results_path, result_name)

            if not os.path.exists(new_result_path):
                os.mkdir(new_result_path)

            os.system(
                f"""
                python3 svm_testing_pipeline.py  \
                --bonaFideFeatures {bonafideFeaturePath} \
                --morphedAttackFeatures {morphedFeaturePath} \
                --modelOutput {model_path} \
                --resultOutput {new_result_path} \
                --loadPreds True
                """
            )
