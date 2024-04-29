import argparse
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--images_path",
        type=str,
        dest="images_path",
        help="Путь до изображений",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Выводить информацию об удаленных файлах")

    args = parser.parse_args()

    files = [p.resolve() for p in Path(args.images_path).glob("**/*") if p.suffix not in {".png", ".jpg", ".jpeg"}]

    for file in files:
        os.remove(file)

        if args.verbose:
            print(f"Удалено: {os.path.basename(file)}")
