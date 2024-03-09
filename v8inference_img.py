import argparse
import os

import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model-path",
    type=str,
    required=True,
    help="Path to model",
)

parser.add_argument(
    "-imgs",
    "--images-path",
    type=str,
    required=True,
    help="Path to folder with input images",
)

parser.add_argument(
    "-o",
    "--output-path",
    type=str,
    required=True,
    help="Path to folder for saving images",
)

parser.add_argument(
    "-conf",
    "--conf-th",
    type=float,
    default=0.3,
    help="Model confidence threshold",
)

args = parser.parse_args()

# Load a model
model = YOLO(args.model_path)

IMAGE_FOLDER = args.images_path
IMAGE_OUTPUT_FOLDER = args.output_path

os.makedirs(IMAGE_OUTPUT_FOLDER, exist_ok=True)

images = sorted(os.listdir(IMAGE_FOLDER))
images = [image for image in images if image.endswith((".jpg", ".jpeg", ".png"))]

for image in images:
    
    IMAGE_PATH = os.path.join(IMAGE_FOLDER, image)
    image_output = os.path.join(IMAGE_OUTPUT_FOLDER, image)

    rgb_image = cv2.cvtColor(cv2.imread(IMAGE_PATH),cv2.COLOR_BGR2RGB)

    results = model(rgb_image, conf=args.conf_th)
    annotated_frame = results[0].plot()
    print(results)
    
    name, ext = os.path.splitext(image_output)
    txt_output = name + ".txt"
    results[0].save_txt(txt_output, save_conf=False)
    
    if not os.path.isfile(txt_output):
        with open(txt_output, 'w') as f:
            f.write("No objects detected")
    
    cv2.imwrite(image_output, annotated_frame)