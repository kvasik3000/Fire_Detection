import argparse
import os
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--images_dir",
        type=str,
        dest="images_dir",
        help="Путь до директории с изображениями.",
    )

    parser.add_argument(
        "--save_cnt",
        type=int,
        dest="save_cnt",
        help="Количество изображений, которые необходимо оставить.",
    )

    args = parser.parse_args()

    images = os.listdir(args.images_dir)
    extra = random.sample(images, len(images) - args.save_cnt)

    for image in extra:
        os.remove(os.path.join(args.images_dir, image))
