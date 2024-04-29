import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--labels_dir",
        type=str,
        dest="labels_dir",
        help="Путь до директории с аннотациями.",
    )

    parser.add_argument(
        "--target_dir",
        type=str,
        dest="target_dir",
        help="Путь до директории, в которую будут перемещены изображения.",
    )

    parser.add_argument(
        "--source_dir",
        type=str,
        dest="source_dir",
        help="Путь до директории, из которой будут перемещены изображения.",
    )

    parser.add_argument(
        "-c",
        "--copy",
        action="store_true",
        dest="copy",
        help="Копировать изображения вместо перемещения.",
    )

    args = parser.parse_args()

    source_images_paths = list(Path(args.source_dir).glob("*/**/*"))
    source_images_names = list(map(os.path.basename, source_images_paths))
    target_images = os.listdir(args.target_dir)

    for label in tqdm(os.listdir(args.labels_dir), leave=False):
        for ext in [".jpg", ".png", ".jpeg"]:

            image = str(Path(label).with_suffix(ext))

            if image not in source_images_names:
                continue

            if image not in target_images:

                id = source_images_names.index(image)

                if args.copy:
                    shutil.copy(source_images_paths[id].resolve(), os.path.join(args.target_dir, image))
                else:
                    os.replace(source_images_paths[id].resolve(), os.path.join(args.target_dir, image))

            break
