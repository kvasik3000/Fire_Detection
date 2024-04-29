import argparse
import os
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--images_dir",
        type=str,
        dest="images_dir",
        help="Путь до директории с изображениями",
    )

    parser.add_argument(
        "--labels_dir",
        type=str,
        dest="labels_dir",
        help="Путь до директории, в которую будут записаны пустые аннотации.",
    )

    args = parser.parse_args()
    os.makedirs(args.labels_dir, exist_ok=True)

    for image in tqdm(os.listdir(args.images_dir), leave=False):
        with open(os.path.join(args.labels_dir, Path(image).with_suffix(".txt")), "w") as file:
            file.write("")
