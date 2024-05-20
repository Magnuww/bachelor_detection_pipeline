from argparse import ArgumentParser


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--test",
        help="test paths file",
        required=True,
    )

    parser.add_argument(
        "--train",
        help="train paths file",
        required=True,
    )

    args = parser.parse_args()
    test = args.test
    train = args.train

    return train, test


# TODO: make os agnostic
def create_set(dataset):
    final_set = set()
    for line in open(dataset, "r"):
        path1, path2 = line.split(" ")
        path1 = "".join(path1.split("/")[-4:])
        path2 = "".join(path2.split("/")[-4:])

        final_set.add(path1)
        final_set.add(path2)

    return final_set


if __name__ == "__main__":
    train, test = get_arguments()
    training_set = create_set(train)
    test_set = create_set(test)

    print(test_set & training_set)
