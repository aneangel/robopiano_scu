from pathlib import Path
import sys

import numpy as np


def main() -> None:
    path = Path(sys.argv[1])
    with np.load(path, allow_pickle=True) as data:
        for key in sorted(data.files):
            value = data[key]
            print(key, value.shape, value.dtype)


if __name__ == "__main__":
    main()
