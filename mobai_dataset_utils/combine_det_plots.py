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

    label = f"{model_name} eer: {np.round(eer*100, 2)}% "
    axs.loglog(apcr, bpcr, label=label)

    return fig, axs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--resultPaths",
        help="The results that will be combined",
        required=True,
    )
    # parser.add_argument(
    #     "--output", type=str, help="Path to the result outputs", required=True
    # )

    args = parser.parse_args()

    resultFolders = os.listdir(args.resultPaths)

    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    for folder in resultFolders:
        resultPath = os.path.join(
            args.resultPaths,
            folder,
            "concatenate",
            "model_-s 0 -t 0 -c 10 -b 1 -q_os_True",
            PLOT_VAR_FILE_NAME,
        )
        with open(resultPath, "rb") as file:
            det_plot_vars = pickle.load(file)
            fig, axs = plot_det(det_plot_vars, fig, axs, folder)

    x = np.logspace(-3, 0, 10)
    axs.loglog(x, x, ls="--")
    axs.legend()
    axs.set_title("plot_apcr_bpcr")
    figName = "combined_plots.png"
    fig.savefig(figName)
