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
        help="Путь до директории с аннотациями",
    )

    args = parser.parse_args()
    labels = os.listdir(args.labels_dir)

    for image in tqdm(os.listdir(args.images_dir), leave=False):
        if str(Path(image).with_suffix(".txt")) not in labels:
            os.remove(os.path.join(args.images_dir, image))
