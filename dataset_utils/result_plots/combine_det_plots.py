import pickle
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List
import os

PLOT_VAR_FILE_NAME = "det_plot_vars.pkl"


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


def plot_det(det_plot_vars, fig, axs, model_name):
    apcr = det_plot_vars["apcr"]
    bpcr = det_plot_vars["bpcr"]
    eer = det_plot_vars["eer"]

    label = f"{model_name} EER: {np.round(eer*100, 2)}% "
    axs.loglog(apcr, bpcr, label=label)

    return fig, axs


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 20})
    parser = ArgumentParser()
    parser.add_argument(
        "--resultsPath",
        help="The results that will be combined",
        required=True,
    )

    args = parser.parse_args()

    resultFolders = os.listdir(args.resultsPath)

    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    figName = ""

    for folder in resultFolders:
        resultPath = os.path.join(
            args.resultsPath,
            folder,
            # "concatenate",
            # "model_-s 0 -t 0 -c 10 -b 1 -q_os_True",
            PLOT_VAR_FILE_NAME,
        )
        with open(resultPath, "rb") as file:
            det_plot_vars = pickle.load(file)
            fig, axs = plot_det(det_plot_vars, fig, axs, folder)
        if figName != "":
            figName += "+"

        figName += folder

    x = np.logspace(-3, 0, 10)
    axs.loglog(x, x, ls="--", label="EER")
    axs.legend()
    axs.set_title("plot_apcr_bpcr")
    figName += ".png"
    fig.savefig(os.path.join(args.resultsPath, figName))
