"""Convenience entry point for the isolated Gaussian JSCC codec benchmark."""

import sys

from gaussian_jscc.cli import main


if __name__ == "__main__":
    sys.argv.insert(1, "benchmark-codec")
    main()
