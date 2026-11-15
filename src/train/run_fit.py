"""Run one training fit using a generated holdout."""
import argparse
import sys

import numpy as np

from src.train import train_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", required=True)
    args, training_arguments = parser.parse_known_args()
    with np.load(args.indices, allow_pickle=False) as saved:
        indices = tuple(saved[name] for name in ("train", "val", "test"))

    def fixed_split(frame, *_):
        joined = np.concatenate(indices)
        if not np.array_equal(np.sort(joined), np.arange(len(frame))):
            raise ValueError("Split indices do not partition the input rows")
        return indices

    train_base.make_split = fixed_split
    train_base.PROTOCOLS += ("RANDOM_IN_RANGE",)
    train_base.GRID_PROTOCOLS += ("RANDOM_IN_RANGE",)
    sys.argv = [sys.argv[0], *training_arguments]
    train_base.main()


if __name__ == "__main__":
    main()
