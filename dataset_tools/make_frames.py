import argparse
import os
from pathlib import Path

import cv2
from dataset_handler import Dataset_Handler
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--videos_dir",
        type=str,
        default="videos",
        help="Директория с видео.",
    )

    parser.add_argument(
        "--frames_dir",
        type=str,
        default="frames",
        help="Директория, в которую будут записаны кадры." "Для каждого видео создается отдельная поддиректория.",
    )

    parser.add_argument("-f", "--frames_per_video", type=int, default=None, help="Количество кадров, взятых с одного видео.")

    parser.add_argument("-ts", "--timestep", type=int, default=None, help="Временной шаг для сохранения кадра (в секундах).")

    parser.add_argument(
        "-d", "--delete_duplicates", action="store_true", help="Удалить дубликаты сразу после нарезки всех видео."
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Выводить информацию об удаленных дубликатах.")

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        dest="treshold",
        default=0.99,
        help="Порог косинусного расстояния для определения дубликатов",
    )

    args = parser.parse_args()
    os.makedirs(args.frames_dir, exist_ok=True)

    for video in tqdm(os.listdir(args.videos_dir), leave=False):

        file_name = Path(video).with_suffix("")
        save_path = os.path.join(args.frames_dir, file_name)

        cap = cv2.VideoCapture(os.path.join(args.videos_dir, video))

        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        duration = frames / fps

        if args.timestep is None:
            frames_per_video = args.frames_per_video
            timestep = duration / frames_per_video

        else:
            timestep = args.timestep
            frames_per_video = round(duration / timestep)

        index = 0
        count = 0

        with tqdm(total=frames_per_video, desc=str(file_name), leave=False) as pbar:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if count % int(fps * timestep) == 0:

                    if index >= frames_per_video:
                        break

                    Path.mkdir(Path(save_path), exist_ok=True)
                    cv2.imwrite(os.path.join(save_path, f"{file_name}_{index}.jpg"), frame)

                    pbar.update()
                    index += 1

                count += 1

        cap.release()

    if args.delete_duplicates:
        for frames_folder in tqdm(os.listdir(args.frames_dir)):
            dataset = Dataset_Handler(os.path.join(args.frames_dir, frames_folder))
            dataset.find_duplicates(cosine_treshold=args.treshold)
            dataset.delete_duplicates(verbose=args.verbose)
