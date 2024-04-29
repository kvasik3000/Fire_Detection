import argparse
import os
from pathlib import Path

import cv2
from pycocotools.coco import COCO
from tqdm import tqdm


def convert_bbox(x1, y1, w, h, image_w, image_h):
    return [(2 * x1 + w) / (2 * image_w), (2 * y1 + h) / (2 * image_h), w / image_w, h / image_h]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--coco_path",
        type=str,
        dest="coco_path",
        help="Путь до аннотаций в формате .json",
    )

    parser.add_argument(
        "--images_path",
        type=str,
        dest="images_path",
        help="Путь до изображений",
        default="images",
    )

    parser.add_argument(
        "--labels_path",
        type=str,
        dest="labels_path",
        help="Путь до директории, в которую будут записаны аннотации",
        default="labels",
    )

    args = parser.parse_args()

    coco = COCO(args.coco_path)

    for id in tqdm(coco.imgs.keys(), leave=False):
        img_name = coco.imgs[id]["file_name"]
        annotations = coco.loadAnns(coco.getAnnIds([id]))

        os.makedirs(os.path.join(args.labels_path, os.path.split(img_name)[0]), exist_ok=True)

        with open(os.path.join(args.labels_path, Path(img_name).with_suffix(".txt")), "w") as file:
            for annotation in annotations:

                bbox = annotation["bbox"]
                area = annotation["area"]

                img = cv2.imread(os.path.join(args.images_path, img_name))
                height, width = img.shape[:-1]

                yolo_bbox = convert_bbox(*bbox, width, height)

                file.write(f"0 {' '.join(list(map(str, yolo_bbox)))}\n")
