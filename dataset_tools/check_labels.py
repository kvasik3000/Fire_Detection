import argparse
import os
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--labels_dir",
        type=str,
        help="Директория с аннотациями.",
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        dest="images_dir",
        help="Путь до директории с изображениями.",
    )

    parser.add_argument(
        "-r",
        "--remove",
        action="store_true",
        help="Нужно ли удалить файлы с ошибками.",
    )

    args = parser.parse_args()

    for label in tqdm(os.listdir(args.labels_dir), leave=False):

        error = False

        with open(os.path.join(args.labels_dir, label)) as file:
            data = file.readlines()

            for line in data:
                if int(line.split()[0]) != 0:
                    print("Номер класса не равен 0")
                    error = True

                if any([float(value) > 1 for value in line.split()[1:]]):
                    print(f"Значение больше 1 в {label}")
                    error = True
                    break

        if error:
            if args.remove:
                os.remove(os.path.join(args.images_dir, Path(label).with_suffix(".jpg")))
                os.remove(os.path.join(args.labels_dir, label))
