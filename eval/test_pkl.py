"""Inspect the contents of a motion PKL from the command line."""

import argparse
import pickle

import numpy as np


def inspect_motion(path):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    summary = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            summary[key] = "shape={}, dtype={}".format(value.shape, value.dtype)
        else:
            summary[key] = type(value).__name__
    return summary


def main():
    parser = argparse.ArgumentParser(description="Inspect an EDGE/AIST++ motion PKL")
    parser.add_argument("path")
    options = parser.parse_args()
    for key, value in inspect_motion(options.path).items():
        print("{}: {}".format(key, value))


if __name__ == "__main__":
    main()
