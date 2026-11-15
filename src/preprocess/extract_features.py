"""Build Morgan features for the two reference datasets before grid selection."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("references", nargs="+", type=Path)
    args = parser.parse_args()
    for reference in args.references:
        subprocess.run([sys.executable, "-m", "src.preprocess.make_drug_reprs",
                        "--data", str(reference)], check=True)


if __name__ == "__main__":
    main()
