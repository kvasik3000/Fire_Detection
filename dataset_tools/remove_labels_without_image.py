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
    images = os.listdir(args.images_dir)

    for label in tqdm(os.listdir(args.labels_dir), leave=False):
        if all([str(Path(label).with_suffix(ext)) not in images for ext in [".jpg", ".png", ".jpeg"]]):
            os.remove(os.path.join(args.labels_dir, label))
