import argparse
import os
from pathlib import Path
from typing import Literal
from typing import Union
from typing import Optional
import contextlib
from subprocess import CalledProcessError
from subprocess import check_output
from subprocess import run

from tqdm.auto import tqdm

import cv2
from ultralytics import YOLO

SUPPORTED_VIDEO_EXTS = [".mp4"]
SUPPORTED_IMAGE_EXTS = [".jpg", ".png", ".jpeg"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        required=True,
        help="Path to model",
    )

    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        required=True,
        help="Path to folder with data",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        required=True,
        help="Path to folder for saving images",
    )

    parser.add_argument(
        "-conf",
        "--model-conf",
        type=float,
        default=0.3,
        help="Model confidence threshold",
    )

    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="all",
        choices=[
            "all",
            "video",
            "img",
        ],
        help="Which files process",
    )

    parser.add_argument(
        "-cv",
        "--compress-video",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compress video after inference using h264 codec and ffmpeg",
    )

    parser.add_argument(
        "--compress-overwrite",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite compressed video",
    )

    return parser.parse_args()


def folder_inferece(
    model_path: str,
    model_conf: float,
    input_path: Path,
    output_path: Path,
    target: Literal["all", "video", "img"],
    **kwargs,
):
    model = YOLO(model_path)

    os.makedirs(output_path, exist_ok=True)

    processing_extensions = []
    processing_extensions += SUPPORTED_VIDEO_EXTS if target in ["all", "video"] else []
    processing_extensions += SUPPORTED_IMAGE_EXTS if target in ["all", "img"] else []

    if processing_extensions == []:
        raise AttributeError(
            f'The required file types are not specified, possibly an error in: "{target}"'
        )

    for ext in processing_extensions:
        for file in tqdm(input_path.rglob(f"*{ext}"), desc=f"Parsing: \"{ext}\""):
            if ext in SUPPORTED_VIDEO_EXTS:
                video_inference(
                    model,
                    model_conf,
                    file,
                    output_path / (str(file.stem) + file.suffix),
                    compress=kwargs["compress_video"],
                    compress_overwrite=kwargs["compress_overwrite"],
                )
            elif ext in SUPPORTED_IMAGE_EXTS:
                image_inference(
                    model, model_conf, file, output_path / (str(file.stem) + file.suffix)
                )


def compress_function(input: Path, output: Path, overwrite: bool = False):
    with open("/dev/null", "w") as dummy_f:
        with contextlib.redirect_stdout(dummy_f):
            arguments = [
                "ffmpeg",
                "-i",
                input,
                "-vcodec",
                "h264",
                "-hide_banner",
                "-loglevel",
                "error",
                "-crf",
                "28",
                output,
            ]
            if overwrite:
                arguments.append("-y")
            run(arguments)


def video_inference(
    model,
    model_conf: float,
    video_path: Path,
    output_path: Path,
    compress: bool,
    compress_overwrite: bool,
):

    output_path_root = output_path.parent
    os.makedirs(output_path_root, exist_ok=True)

    if compress:
        original_video_path = video_path
        video_path = str(video_path).rsplit(".", maxsplit=1)[0]
        video_path += "_.mp4"

    video_stream_in = cv2.VideoCapture(str(video_path))

    video_width = int(video_stream_in.get(3))
    video_height = int(video_stream_in.get(4))
    framerate = int(video_stream_in.get(5))
    total_frames = int(video_stream_in.get(7))

    video_stream_out = cv2.VideoWriter(
        filename=str(output_path),
        fourcc=0x7634706D,
        fps=framerate,
        frameSize=(video_width, video_height),
    )

    while True:
        success, frame = video_stream_in.read()
        if success:
            annotated_frame, _ = image_inference(model, model_conf, image=frame)
            video_stream_out.write(annotated_frame)
        else:
            break

    video_stream_in.release()
    video_stream_out.release()

    if compress:
        compress_function(str(video_path), str(original_video_path), compress_overwrite)
        os.remove(str(video_path))


def image_inference(
    model,
    model_conf: float,
    image: Union[Path, cv2.Mat],
    output_path: Optional[Path] = None,
):

    if output_path:
        output_path_root = output_path.parent
        os.makedirs(output_path_root, exist_ok=True)

    if isinstance(image, Path):
        image = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)

    results = model(image, conf=model_conf, verbose=False)
    annotated_frame = results[0].plot()

    if output_path:
        cv2.imwrite(str(output_path), annotated_frame)

        txt_output = str(output_path).rsplit(".", maxsplit=1)[0]
        txt_output += ".txt"

        results[0].save_txt(txt_output, save_conf=False)
        
        if not os.path.isfile(txt_output):
            with open(txt_output, 'w') as f:
                f.write("")

    return annotated_frame, results


if __name__ == "__main__":

    args = parse_args()

    args = args.__dict__

    if args["compress_video"]:
        try:
            check_output(["which", "ffmpeg"])
        except CalledProcessError:
            print(f"ffmpeg was not found. Compression cannot be done.")
            args["compress_video"] = False

    folder_inferece(**args)
