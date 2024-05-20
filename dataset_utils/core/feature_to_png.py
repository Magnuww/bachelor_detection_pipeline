import os
import re
import argparse

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

with open(args.list, "r") as f, open(f"{args.out}.txt", "w") as out:
    features = f.readlines()
    for i in features:
        line = i.split()
        img1 = line[0]
        img2 = line[1]
        split = img1.split("/")
        id1 = re.findall(r"\d+", split[-1])
        regexstr = f".*{split[2]}.*{id1[-1]}_1.jpg"
        probe = ""
        refrence = ""
        for root, dirs, files in os.walk(args.dir):
            for file in files:
                fullpath = os.path.join(root, file)
                if re.match(regexstr, fullpath):
                    probe = fullpath
                    break
        split = img2.split("/")
        id2 = re.findall(r"\d+", split[-1])
        print(id2)
        if len(id2) == 2:
            regexstr = f".*{split[2]}.*{id2[-2]}.*{id2[-1]}.*.png"
        else:
            regexstr = f".*{split[2]}.*{id2[-1]}_0.jpg"
        for root, dirs, files in os.walk(args.dir):
            for file in files:
                fullpath = os.path.join(root, file)
                if re.match(regexstr, fullpath):
                    refrence = fullpath
                    break
        if refrence == "" or probe == "":
            print(regexstr)
            print(f"Could not find {img1} or {img2}")
        else:
            out.write(f"{probe} {refrence}\n")
