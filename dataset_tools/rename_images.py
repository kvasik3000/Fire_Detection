import argparse
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--images_dir",
        type=str,
        dest="images_dir",
        help="Путь до директории с изображениями/поддиректориями",
    )

    parser.add_argument(
        "--labels_dir",
        type=str,
        dest="labels_dir",
        help="Путь до директории с аннотациями",
    )

    parser.add_argument(
        "-r",
        "--recursive",
        dest="recursive",
        action="store_true",
        help="Переименовать изображения во всех поддиректориях."
        "Ожидается, что структура директории с аннотациями будет повторять структуру директории с изображениями.",
    )

    parser.add_argument("-n", "--new_name", dest="new_name", default=None, help="Новое имя для изображений (с индексом)")

    args = parser.parse_args()
    index = 0

    if args.recursive:
        for address, dirs, files in os.walk(args.images_dir):
            for name in files:

                if Path(name).suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                    continue

                path = os.path.normpath(os.path.relpath(address, args.images_dir))
                prefix = path.split(os.sep)
                prefix = "_".join(prefix)

                if prefix == ".":
                    prefix = os.path.basename(args.images_dir)

                labels_subdir = os.path.join(args.labels_dir, path)

                if args.new_name is not None:
                    new_name = f"{args.new_name}_{index}{Path(name).suffix}"
                    index += 1

                else:
                    new_name = f"{prefix}_{name}"

                os.rename(os.path.join(address, name), os.path.join(address, new_name))

                os.rename(
                    os.path.join(labels_subdir, Path(name).with_suffix(".txt")),
                    os.path.join(labels_subdir, Path(new_name).with_suffix(".txt")),
                )

    else:
        for name in os.listdir(args.images_dir):

            if Path(name).suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                continue

            if args.new_name is not None:
                new_name = f"{args.new_name}_{index}{Path(name).suffix}"
                index += 1

            else:
                new_name = f"{os.path.basename(args.images_dir)}_{name}"

            os.rename(os.path.join(args.images_dir, name), os.path.join(args.images_dir, new_name))

            os.rename(
                os.path.join(args.labels_dir, Path(name).with_suffix(".txt")),
                os.path.join(args.labels_dir, Path(new_name).with_suffix(".txt")),
            )
