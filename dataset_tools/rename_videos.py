import argparse
import os
import random
import string
from pathlib import Path

from deep_translator import GoogleTranslator
from keybert import KeyBERT
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--target_videos_dir",
        type=str,
        dest="target_videos_dir",
        help="Путь до директории с видео, которые нужно переименовать.",
    )

    parser.add_argument(
        "--renamed_videos_dir",
        type=str,
        dest="renamed_videos_dir",
        default=None,
        help="Путь до директории с видео, которые уже переименованны.",
    )

    args = parser.parse_args()
    translator = GoogleTranslator(source="auto", target="en")

    names = os.listdir(args.renamed_videos_dir)

    if args.target_videos_dir is not None:
        names += os.listdir(args.target_videos_dir)

    for video_name in tqdm(os.listdir(args.target_videos_dir), leave=False):

        new_video_name = str(Path(video_name).with_suffix(""))
        new_video_name = translator.translate(new_video_name)

        kw_model = KeyBERT()
        new_video_name = kw_model.extract_keywords(new_video_name, keyphrase_ngram_range=(1, 4), stop_words=None)[0][0]

        new_video_name = new_video_name.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
        new_video_name = new_video_name.strip("_ ")
        new_video_name = str(Path(new_video_name).with_suffix(".mp4"))
        new_video_name = new_video_name.capitalize().replace(" ", "_")

        while new_video_name in names:
            new_video_name = str(Path(new_video_name).with_suffix(""))
            new_video_name += "_" + str(random.randint(0, 9))
            new_video_name = str(Path(new_video_name).with_suffix(".mp4"))

        names.append(new_video_name)
        os.rename(os.path.join(args.target_videos_dir, video_name), os.path.join(args.target_videos_dir, new_video_name))
