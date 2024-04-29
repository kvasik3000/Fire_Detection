import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--images_dir",
        type=str,
        dest="images_dir",
        help="Путь до директории с изображениями.",
    )

    parser.add_argument(
        "--target_dir",
        type=str,
        dest="target_dir",
        help="Путь до директории, в которую будут перемещены аннотации.",
    )

    parser.add_argument(
        "--source_dir",
        type=str,
        dest="source_dir",
        help="Путь до директории, из которой будут перемещены аннотации.",
    )

    parser.add_argument(
        "-c",
        "--copy",
        action="store_true",
        dest="copy",
        help="Копировать аннотации вместо перемещения.",
    )

    args = parser.parse_args()

    for image in tqdm(os.listdir(args.images_dir), leave=False):
        label = str(Path(image).with_suffix(".txt"))

        if label not in os.listdir(args.target_dir):
            if args.copy:
                shutil.copy(os.path.join(args.source_dir, label), os.path.join(args.target_dir, label))
            else:
                os.replace(os.path.join(args.source_dir, label), os.path.join(args.target_dir, label))
