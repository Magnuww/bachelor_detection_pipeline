import os
import argparse
import shutil as sh

# script that generates a folder with images correspoinding to the provided list of feature files

argparser = argparse.ArgumentParser()

argparser.add_argument("--dir", help="Directory containing the jpg files")
argparser.add_argument(
    "--list", type=str, help="List of feature files to find matching pngs"
)
argparser.add_argument(
    "--out", type=str, help="Output file to write the list of pngs to"
)

args = argparser.parse_args()


# REF IMAGES ARE IDENTIFIED BY 0 AND PROBE IMAGES ARE IDENTIFIED BY 1
#
# Creates a folder for the new path if it does not exist, then copies the image over to the new path.
def copy_image(path, file, dest, probe: bool):
    print(os.path.exists(dest))
    if not os.path.exists(dest + "/"):
        os.makedirs(dest)
    if probe:
        outputname = "probe_" + file[:-6] + ".jpg"

    else:
        outputname = "ref_" + file[:-6] + ".jpg"
    print(path + file)
    sh.copy(path + file, dest + "/" + outputname)


print("start")
for root, dirs, files in os.walk(args.dir):
    for file in files:
        rsplt = root.split("/")
        if len(rsplt) > 1:
            # AGE images start with 1. therefore use 1 as ref and 2 as probe.
            if rsplt[1] == "AGE":
                outpath = args.out + "/" + "/".join(rsplt[:-1]) + "/"
                print(outpath)

                if file.endswith("1.jpg"):
                    copy_image(
                        root + "/",
                        file,
                        args.out + "/" + "/".join(rsplt[:-1]) + "/",
                        False,
                    )
                if file.endswith("2.jpg"):
                    copy_image(
                        root + "/",
                        file,
                        args.out + "/" + "/".join(rsplt[:-1]) + "/",
                        True,
                    )
            else:
                if file.endswith("0.jpg"):
                    copy_image(root + "/", file, args.out + "/" + root, False)
                if file.endswith("1.jpg"):
                    copy_image(root + "/", file, args.out + "/" + root, True)
