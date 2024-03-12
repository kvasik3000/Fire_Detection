import argparse
import contextlib
from subprocess import CalledProcessError
from subprocess import check_output
from subprocess import run


def compress(in_video_path: str, out_video_path: str, overwrite: bool = False):
    """
    Compresses video file using ffmpeg and h264 codec.

    Parameters
    ----------
    in_video_path : str
        Path to the video file to be compressed

    out_video_path : str
        Path for saving the result video file

    overwrite : bool, optional
        Flag to allow overwriting of the result video file

    """

    try:
        check_output(["which", "ffmpeg"])
    except CalledProcessError:
        print(f"ffmpeg was not found. Compression cannot be done. Leaving the source file unchanged {in_video_path}")

    with open("/dev/null", "w") as dummy_f:
        with contextlib.redirect_stdout(dummy_f):
            arguments = [
                "ffmpeg",
                "-i",
                in_video_path,
                "-vcodec",
                "h264",
                "-hide_banner",
                "-loglevel",
                "error",
                "-crf",
                "28",
                out_video_path,
            ]
            if overwrite:
                arguments.append("-y")
            run(arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default="")
    parser.add_argument(
        "--overwrite",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args()

    video_file = args.input
    if args.output == "":
        extension = video_file.split(".")[-1]
        output_filename = video_file.replace(f".{extension}", f"-compressed.{extension}")
    else:
        output_filename = args.output

    compress(video_file, output_filename, overwrite=args.overwrite)
