import os
import numpy as np
from argparse import ArgumentParser
import re
import shutil
from ..dataset_utils import build_traversal_array

# Different regex for different databases naming conventions
tuf_rename = re.compile(r"M_(\d+)_(\d+).*")
age_rename = re.compile(r"M_(\d+)_.*[M|F].*_(\d+)_.*_W0.*")
feret_rename = re.compile(r"M_(\d+)_1_(\d+).*")
frgc_rename = re.compile(r"M_(\d+)d.*_(\d+)d.*")


mipgan_rename = re.compile(r"(\d+).*-vs-(\d+).*")


mordiff_age_feret_rename = re.compile(r"_(\d+)_.+_(\d+)_.*")
mordiff_tuf_frgc_rename = re.compile(r"_(\d+).*_(\d+).*")


def format_name(root: str, file: str, morphalg: str) -> str:
    path = root.split("/")
    argspassed = len(args.out.split("/"))
    database = path[argspassed]

    match = ""
    if morphalg == None:
        raise ValueError("No morphalg passed")
    if morphalg.lower() == "mipgan":
        print(file)
        match = mipgan_rename.match(file)

    elif morphalg.lower() == "mordiff":
        print(database)
        match database:
            case "FRGC" | "TUF":
                print(file)
                match = mordiff_tuf_frgc_rename.match(file)
            case "AGE" | "FERET":
                print(file)
                match = mordiff_age_feret_rename.match(file)
    elif morphalg.lower() == "ubo":
        match database:
            case "FERET":
                match = feret_rename.match(file)
            case "FRGC":
                match = frgc_rename.match(file)
            case "TUF":
                match = tuf_rename.match(file)
            case "AGE":
                match = age_rename.match(file)
    if match == "" or match is None:
        raise ValueError("No match found")
    new_name = "ref_{}_{}.txt".format(*match.groups())

    return new_name


def traverse(path, out: str, rename=False, morphalg=None):
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                path = root.split("/")
                outpath = out + "/" + "/".join(path[-5:]) + "/"
                if not os.path.exists(outpath):
                    os.makedirs(outpath)

                name = file[:-4] + ".txt"

                if rename:
                    name = format_name(root, file, morphalg)

                source = os.path.join(root, file)
                destination = os.path.join(outpath, name)
                shutil.copy2(source, destination)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--path", type=str, help="Path to the folder containing images")
    parser.add_argument("--out", type=str, help="output path")
    parser.add_argument("--rename", type=bool, help="rename files")
    parser.add_argument("--morphalg", type=str, help="morphing algorithm")
    args = parser.parse_args()

    if args.rename:
        traverse(args.path, args.out, True, morphalg=args.morphalg)
    else:
        traverse(args.path, args.out)
